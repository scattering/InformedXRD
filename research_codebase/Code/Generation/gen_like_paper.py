#!/usr/bin/env python3
# gen_like_paper.py
from __future__ import annotations

import os
import warnings
import traceback
import re
import sys
import tempfile
import subprocess
import hashlib
from functools import lru_cache
from multiprocessing import Pool
import random

import numpy as np
import h5py
from tqdm import tqdm
import pandas as pd

from pyxtal import pyxtal
from pyxtal.symmetry import get_wyckoffs
import spglib
from ase.io import write as ase_write

try:
    from pycrysfml import cfml_py_utilities
except ImportError:
    cfml_py_utilities = None

try:
    import CFML_api
except ImportError:
    CFML_api = None

from scipy.linalg import cholesky
from scipy.stats import truncnorm

warnings.filterwarnings("ignore")

# ==============================
# 0. Config / environment
# ==============================

os.environ["FULLPROF"] = os.environ.get("FULLPROF", "/storage/ejb12/FullProf")

TOTAL_SAMPLES = int(os.environ.get("TOTAL_SAMPLES", "10000000"))
MAX_INTENSITY = 1000.0
EG_SAMPLING_MODE = os.environ.get("EG_SAMPLING_MODE", "rruff").strip().lower()

# Cu Kα doublet (2:1 weights)
KALPHA1 = 1.54056
KALPHA2 = 1.54439
W_KALPHA1 = 2.0 / 3.0
W_KALPHA2 = 1.0 / 3.0

INSTRUMENT_PARAMS = {"twotheta_min": 5.0, "twotheta_max": 90.0, "step_size": 0.01}
NUM_POINTS = int((INSTRUMENT_PARAMS["twotheta_max"] - INSTRUMENT_PARAMS["twotheta_min"]) / INSTRUMENT_PARAMS["step_size"] + 1)
TWO_THETA = np.linspace(INSTRUMENT_PARAMS["twotheta_min"], INSTRUMENT_PARAMS["twotheta_max"], NUM_POINTS, dtype=np.float32)

USE_EXACT_GP = bool(int(os.environ.get("USE_EXACT_GP", "0")))
RFF_FEATURES = int(os.environ.get("RFF_FEATURES", "1024"))

NUM_SHARDS = int(os.environ.get("GENMAP_NUM_SHARDS", "1"))
SHARD_ID = int(os.environ.get("GENMAP_SHARD_ID", "0"))

_base = TOTAL_SAMPLES // NUM_SHARDS
_rem = TOTAL_SAMPLES % NUM_SHARDS
LOCAL_SAMPLES = _base + (1 if SHARD_ID < _rem else 0)

print(f"[INFO] Shard {SHARD_ID}/{NUM_SHARDS} -> LOCAL_SAMPLES={LOCAL_SAMPLES:,} (TOTAL_SAMPLES={TOTAL_SAMPLES:,})")
print(f"[INFO] Background mode: {'EXACT_GP' if USE_EXACT_GP else f'RFF (features={RFF_FEATURES})'}")

BASE_OUTPUT_HDF5 = os.environ.get("BASE_OUTPUT_HDF5", "rruff_dist_shard")
BATCH_ID = os.environ.get("BATCH_ID", "0")
OUTPUT_HDF5 = f"{BASE_OUTPUT_HDF5}_batch{BATCH_ID}_shard{SHARD_ID:04d}_02182025.hdf5"
RESUME_GENERATION = bool(int(os.environ.get("RESUME_GENERATION", "1")))
BATCH_FLUSH_SIZE = int(os.environ.get("BATCH_FLUSH_SIZE", "8"))
MAX_TASK_MULTIPLIER = int(os.environ.get("MAX_TASK_MULTIPLIER", "20"))
SAMPLE_RETRY_LIMIT = int(os.environ.get("SAMPLE_RETRY_LIMIT", "3"))

EXT_MAP_CSV = os.environ.get("EXT_MAP_CSV", "/storage/ejb12/Work/Database/FINAL_SPG_ExtG_CrysS_Table.csv")

def get_num_processes():
    override = os.environ.get("NUM_PROCESSES_OVERRIDE")
    if override:
        try:
            n = int(override)
            if n > 0:
                return n
        except ValueError:
            pass
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        try:
            n = int(slurm_cpus)
            if n > 0:
                return n
        except ValueError:
            pass
    return os.cpu_count() or 1

NUM_PROCESSES = get_num_processes()
print(f"[INFO] Using {NUM_PROCESSES} worker processes")

# ==============================
# 1. Build SG <-> EG mapping
# ==============================

df_ext = pd.read_csv(EXT_MAP_CSV)
sg_to_ext = {int(sg): int(eg) for sg, eg in zip(df_ext["Space Group"], df_ext["Extinction Group"])}

ext_to_sg = {}
for sg, eg in sg_to_ext.items():
    ext_to_sg.setdefault(int(eg), []).append(int(sg))

ALL_SPACE_GROUPS = np.array(sorted(sg_to_ext.keys()), dtype=np.int16)
if ALL_SPACE_GROUPS.size == 0:
    raise RuntimeError("EXT_MAP_CSV produced an empty SG list; check the file path and columns.")

# ==============================
# 2. RRUFF EG distribution (Option A)
# ==============================

RRUFF_EG_COUNTS = {
    4: 153, 17: 121, 5: 112, 99: 107, 6: 98, 75: 75, 2: 55, 77: 41, 73: 40, 71: 34,
    29: 32, 22: 23, 10: 22, 76: 21, 21: 21, 93: 20, 63: 20, 3: 19, 97: 17, 18: 14,
    12: 13, 80: 12, 54: 12, 16: 9, 85: 9, 68: 8, 33: 8, 14: 7, 52: 7, 74: 6,
    89: 6, 45: 6, 62: 5, 30: 5, 94: 5, 55: 4, 86: 4, 72: 4, 24: 4, 34: 4,
    1: 4, 31: 4, 69: 4, 60: 4, 11: 4, 65: 3, 35: 3, 32: 3, 70: 3, 40: 3,
    38: 3, 26: 3, 81: 2, 39: 2, 53: 2, 19: 2, 95: 2, 51: 2, 49: 1, 42: 1,
    91: 1, 58: 1, 41: 1, 82: 1, 47: 1, 44: 1, 87: 1, 9: 1, 15: 1, 23: 1,
}

valid_egs = [eg for eg in RRUFF_EG_COUNTS.keys() if eg in ext_to_sg and len(ext_to_sg[eg]) > 0]
if not valid_egs:
    raise RuntimeError("No overlap between RRUFF_EG_COUNTS and EXT_MAP_CSV extinction groups.")

EGS = np.array(valid_egs, dtype=np.int16)
EG_WEIGHTS = np.array([RRUFF_EG_COUNTS[int(eg)] for eg in EGS], dtype=np.float64)
EG_WEIGHTS /= EG_WEIGHTS.sum()

missing_egs = sorted(set(RRUFF_EG_COUNTS.keys()) - set(valid_egs))
if missing_egs:
    print(f"[WARN] Ignoring {len(missing_egs)} EGs not present in EXT_MAP_CSV mapping: {missing_egs}")

ALL_EGS = np.array(sorted(ext_to_sg.keys()), dtype=np.int16)
if ALL_EGS.size != 99:
    raise RuntimeError(f"Expected 99 extinction groups from EXT_MAP_CSV, found {ALL_EGS.size}")

if EG_SAMPLING_MODE not in {"rruff", "uniform"}:
    raise RuntimeError(f"Unsupported EG_SAMPLING_MODE='{EG_SAMPLING_MODE}'. Use 'rruff' or 'uniform'.")

if EG_SAMPLING_MODE == "uniform":
    SAMPLE_EGS = ALL_EGS
    SAMPLE_EG_WEIGHTS = np.full(ALL_EGS.shape, 1.0 / ALL_EGS.size, dtype=np.float64)
    print(f"[INFO] Sampling {len(SAMPLE_EGS)} extinction groups with UNIFORM balanced block schedule.")
else:
    SAMPLE_EGS = EGS
    SAMPLE_EG_WEIGHTS = EG_WEIGHTS
    print(f"[INFO] Sampling {len(SAMPLE_EGS)} extinction groups with RRUFF-weighted probabilities.")

# ==============================
# 3. Crystal system helper
# ==============================

def get_crystal_system_code(space_group: int) -> int:
    if 1 <= space_group <= 2: return 0
    if 3 <= space_group <= 15: return 1
    if 16 <= space_group <= 74: return 2
    if 75 <= space_group <= 142: return 3
    if 143 <= space_group <= 167: return 4
    if 168 <= space_group <= 194: return 5
    if 195 <= space_group <= 230: return 6
    return -1

# ==============================
# 4. Structure generation
# ==============================

ELEMENTS = [
    "H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Cl","Ar","K","Ca",
    "Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr","Rb","Sr",
    "Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I","Xe","Cs","Ba",
    "La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","Hf","Ta","W",
    "Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn","Fr","Ra","Ac","Th","Pa","U",
]

@lru_cache(maxsize=None)
def get_wyckoffs_cached(space_group):
    return tuple(get_wyckoffs(space_group, organized=True))

def sample_valid_composition(space_group):
    wyckoff_data = get_wyckoffs_cached(space_group)
    valid_counts = []
    for positions in wyckoff_data:
        total_atoms = sum(len(pos) for pos in positions)
        if total_atoms > 1:
            valid_counts.append(total_atoms)

    if not valid_counts:
        return ["C"], [2]

    num_atoms = random.choice(valid_counts)
    num_elements = np.random.randint(2, 4)
    elements = random.sample(ELEMENTS, num_elements)

    valid_ratios = [r for r in range(2, num_atoms + 1) if num_atoms % r == 0]
    if not valid_ratios:
        ratios = [num_atoms // num_elements] * num_elements
    else:
        ratios = [random.choice(valid_ratios) for _ in range(num_elements)]
    return elements, ratios

def generate_p1_structure():
    structure = pyxtal()
    num_elements = random.randint(1, 3)
    selected_elements = random.sample(ELEMENTS, num_elements)
    num_atoms = random.randint(2, 10)
    atom_distribution = [num_atoms // num_elements] * num_elements
    remainder = num_atoms % num_elements
    for i in range(remainder):
        atom_distribution[i] += 1
    structure.from_random(dim=3, group=1, species=selected_elements, numIons=atom_distribution)
    return structure

def validate_structure(structure, expected_space_group):
    atoms = structure.to_ase()
    lattice = atoms.get_cell()
    positions = atoms.get_scaled_positions()
    numbers = atoms.get_atomic_numbers()
    cell = (lattice, positions, numbers)
    detected_group = spglib.get_spacegroup(cell, symprec=1e-2)
    if not detected_group:
        return False
    m = re.search(r"\((\d+)\)", detected_group)
    if m:
        return int(m.group(1)) == expected_space_group
    return False

def get_valid_structure(space_group, max_attempts=50):
    attempts = 0
    last_err = "Unknown"
    while attempts < max_attempts:
        try:
            if space_group == 1:
                structure = generate_p1_structure()
            else:
                elements, ratios = sample_valid_composition(space_group)
                structure = pyxtal()
                structure.from_random(dim=3, group=space_group, species=elements, numIons=ratios)
            if validate_structure(structure, space_group):
                return structure
        except Exception as e:
            last_err = str(e)
        attempts += 1
    print(f"[WARN] Failed to generate structure for SG {space_group} after {max_attempts} attempts: {last_err}")
    return None

# ==============================
# 5. PyCrysFML pattern + paper-style augmentation
# ==============================

def extract_structural_info(structure):
    atoms_obj = structure.to_ase()
    a, b, c, alpha, beta, gamma = atoms_obj.get_cell_lengths_and_angles()
    sg = structure.group.number
    positions = atoms_obj.get_scaled_positions()
    symbols = atoms_obj.get_chemical_symbols()
    atom_list = [{"symbol": s, "position": p} for s, p in zip(symbols, positions)]
    return sg, a, b, c, alpha, beta, gamma, atom_list

def sample_paper_uvw(rng):
    u = float(rng.uniform(0.0, 0.01))
    v = 0.0
    w = float(rng.uniform(0.0, 0.01))
    return u, v, w

def create_pattern_dict(space_group, a, b, c, alpha, beta, gamma, atom_list, u, v, w, wavelength):
    sg_type = spglib.get_spacegroup_type(space_group)
    sg_symbol = sg_type["international_short"]
    return {
        "phases": [
            {
                "placeholder": {
                    "_space_group_name_H-M_alt": sg_symbol,
                    "_cell_length_a": float(a),
                    "_cell_length_b": float(b),
                    "_cell_length_c": float(c),
                    "_cell_angle_alpha": float(alpha),
                    "_cell_angle_beta": float(beta),
                    "_cell_angle_gamma": float(gamma),
                    "_atom_site": [
                        {
                            "_label": atom["symbol"],
                            "_type_symbol": atom["symbol"],
                            "_fract_x": float(atom["position"][0]),
                            "_fract_y": float(atom["position"][1]),
                            "_fract_z": float(atom["position"][2]),
                            "_occupancy": 1.0,
                            "_adp_type": "iso",
                            "_B_iso_or_equiv": 0.0,
                        }
                        for atom in atom_list
                    ],
                }
            }
        ],
        "experiments": [
            {
                "NPD": {
                    "_diffrn_radiation_probe": "xray",
                    "_diffrn_radiation_wavelength": float(wavelength),
                    "_pd_instr_resolution_u": float(u),
                    "_pd_instr_resolution_v": float(v),
                    "_pd_instr_resolution_w": float(w),
                    "_pd_instr_resolution_x": 0.0015,
                    "_pd_instr_resolution_y": 0.0,
                    "_pd_instr_reflex_asymmetry_p1": 0.0,
                    "_pd_instr_reflex_asymmetry_p2": 0.0,
                    "_pd_instr_reflex_asymmetry_p3": 0.0,
                    "_pd_instr_reflex_asymmetry_p4": 0.0,
                    "_pd_meas_2theta_offset": 0.0,
                    "_pd_meas_2theta_range_min": float(INSTRUMENT_PARAMS["twotheta_min"]),
                    "_pd_meas_2theta_range_max": float(INSTRUMENT_PARAMS["twotheta_max"]),
                    "_pd_meas_2theta_range_inc": float(INSTRUMENT_PARAMS["step_size"]),
                    "_phase": [{"_label": "placeholder", "_scale": 1.0}],
                    "_pd_background": [
                        {"_2theta": float(INSTRUMENT_PARAMS["twotheta_min"]), "_intensity": 0.0},
                        {"_2theta": float(INSTRUMENT_PARAMS["twotheta_max"]), "_intensity": 0.0},
                    ],
                }
            }
        ],
    }


def cw_powder_pattern_backend(pattern_dict, structure):
    if cfml_py_utilities is not None:
        return cfml_py_utilities.cw_powder_pattern_from_dict(pattern_dict)

    if CFML_api is None:
        raise RuntimeError("Neither pycrysfml.cfml_py_utilities nor CFML_api is available")

    atoms = structure.to_ase()
    with tempfile.NamedTemporaryFile(suffix=".cif", delete=False) as handle:
        cif_path = handle.name
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as handle:
        out_path = handle.name

    try:
        ase_write(cif_path, atoms, format="cif")
        npd = pattern_dict["experiments"][0]["NPD"]
        helper = os.path.join(os.path.dirname(__file__), "tacc", "cfml_api_single_pattern.py")
        subprocess.run(
            [
                sys.executable,
                helper,
                "--cif",
                cif_path,
                "--out",
                out_path,
                "--theta-min",
                str(float(npd["_pd_meas_2theta_range_min"])),
                "--theta-max",
                str(float(npd["_pd_meas_2theta_range_max"])),
                "--step",
                str(float(npd["_pd_meas_2theta_range_inc"])),
                "--u-resolution",
                str(float(npd["_pd_instr_resolution_u"])),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        with np.load(out_path) as data:
            x = data["x"].astype(np.float32, copy=False)
            y = data["y"].astype(np.float32, copy=False)
        if x.shape[0] != TWO_THETA.shape[0] or not np.allclose(x, TWO_THETA, atol=1e-6):
            y = np.interp(TWO_THETA, x, y, left=0.0, right=0.0).astype(np.float32)
            x = TWO_THETA.copy()
        return x, y
    finally:
        try:
            os.remove(cif_path)
        except OSError:
            pass
        try:
            os.remove(out_path)
        except OSError:
            pass

def _truncnorm_scale(rng, mean=0.0, std=75.0, low=0.0, high=150.0) -> float:
    a = (low - mean) / std
    b = (high - mean) / std
    return float(truncnorm.rvs(a, b, loc=mean, scale=std, random_state=rng))

def _rbf_gp_exact(twotheta: np.ndarray, l: float, rng: np.random.Generator, jitter: float = 1e-8) -> np.ndarray:
    x = twotheta.astype(np.float64)
    d2 = (x[:, None] - x[None, :]) ** 2
    K = np.exp(-d2 / (2.0 * l * l))
    K[np.diag_indices_from(K)] += jitter
    L = cholesky(K, lower=True, check_finite=False)
    return (L @ rng.standard_normal(x.shape[0])).astype(np.float64)

def _rbf_gp_rff(twotheta: np.ndarray, l: float, rng: np.random.Generator, m_features: int) -> np.ndarray:
    x = twotheta.astype(np.float64)
    m = int(m_features)
    omega = rng.normal(loc=0.0, scale=1.0 / l, size=m)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=m)
    w = rng.normal(0.0, 1.0, size=m)
    return (np.sqrt(2.0 / m) * (np.cos(x[:, None] * omega[None, :] + phase[None, :]) @ w)).astype(np.float64)

def add_paper_background_noise(intensity_01: np.ndarray, twotheta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    I = intensity_01.astype(np.float64, copy=True)

    l = float(rng.uniform(7.0, 40.0))
    bg = _rbf_gp_exact(twotheta, l, rng) if USE_EXACT_GP else _rbf_gp_rff(twotheta, l, rng, RFF_FEATURES)

    bg -= bg.min()
    s = bg.sum()
    if s > 0:
        bg /= s
    else:
        bg[:] = 0.0

    bg *= _truncnorm_scale(rng, mean=0.0, std=75.0, low=0.0, high=150.0)
    I += bg

    sigma_add = float(rng.uniform(0.0, 0.02))
    I += rng.normal(0.0, sigma_add, size=I.shape)

    sigma_mult = float(rng.uniform(0.0, 0.16))
    I *= rng.normal(1.0, sigma_mult, size=I.shape)

    return I.astype(np.float32)

def generate_pure_pattern_01(structure, rng: np.random.Generator) -> tuple[int, np.ndarray]:
    sg, a, b, c, alpha, beta, gamma, atom_list = extract_structural_info(structure)
    u, v, w = sample_paper_uvw(rng)

    p_k1 = create_pattern_dict(sg, a, b, c, alpha, beta, gamma, atom_list, u, v, w, KALPHA1)
    p_k2 = create_pattern_dict(sg, a, b, c, alpha, beta, gamma, atom_list, u, v, w, KALPHA2)

    _, y1 = cw_powder_pattern_backend(p_k1, structure)
    _, y2 = cw_powder_pattern_backend(p_k2, structure)

    I1 = np.asarray(y1, dtype=np.float32)
    I2 = np.asarray(y2, dtype=np.float32)
    intensity = (W_KALPHA1 * I1 + W_KALPHA2 * I2).astype(np.float32)

    max_val = float(intensity.max())
    if max_val <= 0:
        return sg, intensity
    return sg, (intensity / max_val).astype(np.float32)

def mix_impurity(main_01: np.ndarray, imp_01: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    a = float(rng.uniform(0.0, 0.05))
    return ((1.0 - a) * main_01 + a * imp_01).astype(np.float32)

# ==============================
# 6. EG->SG sampling
# ==============================

def sample_main_sg(rng: np.random.Generator) -> tuple[int, int]:
    eg = int(rng.choice(SAMPLE_EGS, p=SAMPLE_EG_WEIGHTS))
    sgs = ext_to_sg[eg]
    sg = int(sgs[int(rng.integers(0, len(sgs)))])
    return eg, sg


def sample_main_sg_balanced(task_id: int, seed: int) -> tuple[int, int]:
    block_size = ALL_EGS.size
    block_id = task_id // block_size
    block_offset = task_id % block_size
    block_rng = np.random.default_rng(np.random.SeedSequence([seed, block_id, 17]))
    eg_perm = block_rng.permutation(ALL_EGS)
    eg = int(eg_perm[block_offset])

    sg_rng = np.random.default_rng(np.random.SeedSequence([seed, task_id, 23]))
    sgs = ext_to_sg[eg]
    sg = int(sgs[int(sg_rng.integers(0, len(sgs)))])
    return eg, sg

def sample_impurity_sg_uniform(rng: np.random.Generator) -> int:
    return int(ALL_SPACE_GROUPS[int(rng.integers(0, ALL_SPACE_GROUPS.size))])

# ==============================
# 7. HDF5 helpers
# ==============================

def compute_stream_seed() -> int:
    key = f"{OUTPUT_HDF5}|{EG_SAMPLING_MODE}|{SHARD_ID}|{NUM_SHARDS}|{TOTAL_SAMPLES}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def write_progress_attrs(root, tasks_processed: int):
    root.attrs["tasks_processed"] = int(tasks_processed)
    root.attrs["successes_written"] = int(root["intensity"].shape[0]) if "intensity" in root else 0


def init_or_resume_hdf5(out_path, stream_seed: int):
    if os.path.exists(out_path):
        if not RESUME_GENERATION:
            os.remove(out_path)
        else:
            with h5py.File(out_path, "a") as f:
                root = f["/"]
                expected = {
                    "eg_sampling_mode": EG_SAMPLING_MODE,
                    "shard_id": SHARD_ID,
                    "num_shards": NUM_SHARDS,
                    "total_samples": TOTAL_SAMPLES,
                    "local_samples_target": LOCAL_SAMPLES,
                    "stream_seed": stream_seed,
                }
                for key, value in expected.items():
                    if key in root.attrs and root.attrs[key] != value:
                        raise RuntimeError(f"Existing shard metadata mismatch for {key}: {root.attrs[key]} != {value}")
                if "2theta" not in f:
                    f.create_dataset("2theta", data=TWO_THETA, dtype="float32")
                root.attrs["twotheta_min"] = float(INSTRUMENT_PARAMS["twotheta_min"])
                root.attrs["twotheta_max"] = float(INSTRUMENT_PARAMS["twotheta_max"])
                root.attrs["step_size"] = float(INSTRUMENT_PARAMS["step_size"])
                root.attrs["description"] = f"{EG_SAMPLING_MODE.upper()} EG dist synthetic XRD (shard {SHARD_ID}/{NUM_SHARDS}) with impurity + paper aug + Kalpha splitting"
                root.attrs["version"] = f"v6_{EG_SAMPLING_MODE}_eg_dist_impurity_resume"
                root.attrs["eg_sampling_mode"] = EG_SAMPLING_MODE
                root.attrs["eg_schedule"] = "balanced_blocks_of_99" if EG_SAMPLING_MODE == "uniform" else "rruff_weighted_random"
                root.attrs["shard_id"] = SHARD_ID
                root.attrs["num_shards"] = NUM_SHARDS
                root.attrs["total_samples"] = TOTAL_SAMPLES
                root.attrs["local_samples_target"] = LOCAL_SAMPLES
                root.attrs["stream_seed"] = stream_seed
                tasks_processed = int(root.attrs.get("tasks_processed", 0))
                write_progress_attrs(root, tasks_processed)
                f.flush()
                print(f"[INFO] Resuming shard from task {tasks_processed:,}, existing samples={root.attrs['successes_written']:,}")
                return tasks_processed, int(root.attrs["successes_written"])

    with h5py.File(out_path, "w") as f:
        f.create_dataset("2theta", data=TWO_THETA, dtype="float32")
        root = f["/"]
        root.attrs["twotheta_min"] = float(INSTRUMENT_PARAMS["twotheta_min"])
        root.attrs["twotheta_max"] = float(INSTRUMENT_PARAMS["twotheta_max"])
        root.attrs["step_size"] = float(INSTRUMENT_PARAMS["step_size"])
        root.attrs["description"] = f"{EG_SAMPLING_MODE.upper()} EG dist synthetic XRD (shard {SHARD_ID}/{NUM_SHARDS}) with impurity + paper aug + Kalpha splitting"
        root.attrs["version"] = f"v6_{EG_SAMPLING_MODE}_eg_dist_impurity_resume"
        root.attrs["eg_sampling_mode"] = EG_SAMPLING_MODE
        root.attrs["eg_schedule"] = "balanced_blocks_of_99" if EG_SAMPLING_MODE == "uniform" else "rruff_weighted_random"
        root.attrs["shard_id"] = SHARD_ID
        root.attrs["num_shards"] = NUM_SHARDS
        root.attrs["total_samples"] = TOTAL_SAMPLES
        root.attrs["local_samples_target"] = LOCAL_SAMPLES
        root.attrs["stream_seed"] = stream_seed
        write_progress_attrs(root, 0)
    return 0, 0

def append_batch(f, batch):
    if not batch:
        return
    task_ids, sg_list, eg_list, intensity_list = zip(*batch)
    intensities = np.stack(intensity_list, axis=0).astype("float32")

    if "intensity" not in f:
        num_points = intensities.shape[1]
        f.create_dataset("intensity", data=intensities, maxshape=(None, num_points), dtype="float32")
        f.create_dataset("task_id", data=np.array(task_ids, dtype="int64"), maxshape=(None,), dtype="int64")
        f.create_dataset("space_group", data=np.array(sg_list, dtype="int16"), maxshape=(None,), dtype="int16")
        f.create_dataset("extinction_group", data=np.array(eg_list, dtype="int16"), maxshape=(None,), dtype="int16")
        f.create_dataset(
            "crystal_system",
            data=np.array([get_crystal_system_code(int(sg)) for sg in sg_list], dtype="int8"),
            maxshape=(None,),
            dtype="int8",
        )
    else:
        ds_int = f["intensity"]
        ds_task = f["task_id"]
        ds_sg = f["space_group"]
        ds_eg = f["extinction_group"]
        ds_cs = f["crystal_system"]

        n_old = ds_int.shape[0]
        b = intensities.shape[0]

        ds_int.resize((n_old + b, ds_int.shape[1]))
        ds_int[n_old : n_old + b, :] = intensities

        ds_task.resize((n_old + b,))
        ds_task[n_old : n_old + b] = np.array(task_ids, dtype="int64")

        ds_sg.resize((n_old + b,))
        ds_sg[n_old : n_old + b] = np.array(sg_list, dtype="int16")

        ds_eg.resize((n_old + b,))
        ds_eg[n_old : n_old + b] = np.array(eg_list, dtype="int16")

        ds_cs.resize((n_old + b,))
        ds_cs[n_old : n_old + b] = np.array([get_crystal_system_code(int(sg)) for sg in sg_list], dtype="int8")

# ==============================
# 8. Worker + task stream
# ==============================

def worker(task):
    task_id, eg, sg_main, sg_imp, seed = task
    rng = np.random.default_rng(seed)

    last_error = None
    for attempt in range(SAMPLE_RETRY_LIMIT):
        try:
            st_main = get_valid_structure(sg_main)
            if st_main is None:
                continue

            st_imp = get_valid_structure(sg_imp)
            if st_imp is None:
                continue

            sg_out, main_01 = generate_pure_pattern_01(st_main, rng)
            _, imp_01 = generate_pure_pattern_01(st_imp, rng)

            mixed_01 = mix_impurity(main_01, imp_01, rng)
            mixed_aug_01 = add_paper_background_noise(mixed_01, TWO_THETA, rng)

            intensity_out = (mixed_aug_01 * MAX_INTENSITY).astype(np.float32)
            if not np.all(np.isfinite(intensity_out)):
                raise RuntimeError("Generated intensity contains non-finite values")
            return (task_id, sg_out, eg, intensity_out)

        except Exception as e:
            last_error = e
            continue

    if last_error is not None:
        print(
            f"[WARN] Worker dropped task after {SAMPLE_RETRY_LIMIT} attempts "
            f"(SG_main={sg_main}, SG_imp={sg_imp}): {last_error}"
        )
    return (task_id, None)

def build_task(task_id: int, seed: int):
    rng = np.random.default_rng(np.random.SeedSequence([seed, task_id]))
    if EG_SAMPLING_MODE == "uniform":
        eg, sg_main = sample_main_sg_balanced(task_id, seed)
    else:
        eg, sg_main = sample_main_sg(rng)
    sg_imp = sample_impurity_sg_uniform(rng)
    wseed = int(rng.integers(0, 2**63 - 1))
    return (task_id, eg, sg_main, sg_imp, wseed)

def task_stream(max_tasks: int, seed: int, start_task_id: int = 0):
    for task_id in range(start_task_id, max_tasks):
        yield build_task(task_id, seed)

# ==============================
# 9. Main
# ==============================

def main():
    print(f"\n[INFO] Starting generation ({EG_SAMPLING_MODE.upper()} EG dist + impurity)...\n")
    stream_seed = compute_stream_seed()
    start_task_id, successes_written = init_or_resume_hdf5(OUTPUT_HDF5, stream_seed)
    if successes_written >= LOCAL_SAMPLES:
        print(f"[INFO] Shard already complete: {OUTPUT_HDF5}")
        return
    max_tasks = max(LOCAL_SAMPLES, LOCAL_SAMPLES * MAX_TASK_MULTIPLIER)

    with h5py.File(OUTPUT_HDF5, "a") as f:
        batch = []
        batch_size = BATCH_FLUSH_SIZE

        with tqdm(total=LOCAL_SAMPLES, initial=successes_written, desc=f"Shard {SHARD_ID} generating", dynamic_ncols=True) as pbar:
            if NUM_PROCESSES <= 1:
                task_id = start_task_id
                while successes_written < LOCAL_SAMPLES and task_id < max_tasks:
                    res = worker(build_task(task_id, stream_seed))
                    task_id = int(res[0])
                    write_progress_attrs(f["/"], task_id + 1)
                    if res[1] is not None:
                        _, sg_out, eg, intensity = res
                        if intensity.shape[0] == NUM_POINTS:
                            batch.append((task_id, sg_out, eg, intensity))
                            successes_written += 1
                            pbar.update(1)
                    if ((task_id + 1) % batch_size) == 0:
                        if batch:
                            append_batch(f, batch)
                            batch = []
                        write_progress_attrs(f["/"], task_id + 1)
                        f.flush()
                    task_id += 1

                if successes_written < LOCAL_SAMPLES:
                    raise RuntimeError(
                        f"Shard {SHARD_ID} exhausted max task budget ({max_tasks}) before reaching {LOCAL_SAMPLES} samples"
                    )
            else:
                pool = Pool(processes=NUM_PROCESSES)
                iterator = pool.imap_unordered(worker, task_stream(max_tasks, stream_seed, start_task_id=start_task_id), chunksize=8)

                try:
                    for res in iterator:
                        task_id = int(res[0])
                        write_progress_attrs(f["/"], task_id + 1)
                        if ((task_id + 1) % batch_size) == 0:
                            f.flush()
                        if res[1] is not None:
                            _, sg_out, eg, intensity = res
                            if intensity.shape[0] == NUM_POINTS:
                                batch.append((task_id, sg_out, eg, intensity))
                                successes_written += 1
                                pbar.update(1)
                        if len(batch) >= batch_size:
                            append_batch(f, batch)
                            write_progress_attrs(f["/"], task_id + 1)
                            f.flush()
                            batch = []
                        if successes_written >= LOCAL_SAMPLES:
                            break
                finally:
                    pool.close()
                    pool.join()

        if batch:
            append_batch(f, batch)
        write_progress_attrs(f["/"], int(f["/"].attrs["tasks_processed"]))
        f.flush()

    print(f"\n[INFO] Shard {SHARD_ID} complete. Output: {OUTPUT_HDF5}")

if __name__ == "__main__":
    main()
