#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
import warnings
from collections import Counter
from functools import lru_cache
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import spglib
from ase.io import write as ase_write
from pyxtal import pyxtal
from pyxtal.symmetry import Group, get_wyckoffs
from scipy.linalg import cholesky
from tqdm import tqdm
try:
    from pymatgen.symmetry.groups import SpaceGroup as PmgSpaceGroup
except ImportError:
    PmgSpaceGroup = None

try:
    from pycrysfml import cfml_py_utilities
except ImportError:
    cfml_py_utilities = None

try:
    import CFML_api
except ImportError:
    CFML_api = None

FORMULA_TOKEN_RE = re.compile(r"([A-Z][a-z]?)([0-9]*\.?[0-9]*)")
VALID_ELEMENT_SYMBOLS = {
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
}
EXCLUDED_UNSTABLE_ELEMENTS = {
    "Tc", "Pm", "Po", "At", "Rn", "Fr", "Ra", "Ac",
    "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs",
    "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv",
    "Ts", "Og",
}
PYXTAL_UNSUPPORTED_RADIUS_ELEMENTS = {"Bk", "Cf"}
ALLOWED_CHEMISTRY_SYMBOLS = VALID_ELEMENT_SYMBOLS - EXCLUDED_UNSTABLE_ELEMENTS - PYXTAL_UNSUPPORTED_RADIUS_ELEMENTS

warnings.filterwarnings("ignore")


os.environ["FULLPROF"] = os.environ.get("FULLPROF", "/storage/ejb12/FullProf")

TOTAL_SAMPLES = int(os.environ.get("TOTAL_SAMPLES", "1000000"))
MAX_INTENSITY = 1000.0
EG_SAMPLING_MODE = os.environ.get("EG_SAMPLING_MODE", "rruff").strip().lower()
PRIOR_MODE = os.environ.get("PRIOR_MODE", "robust").strip().lower()
ENABLE_PREFERRED_ORIENTATION = bool(int(os.environ.get("ENABLE_PREFERRED_ORIENTATION", "0")))

KALPHA1 = 1.54056
KALPHA2 = 1.54439
W_KALPHA1 = 2.0 / 3.0
W_KALPHA2 = 1.0 / 3.0

INSTRUMENT_PARAMS = {"twotheta_min": 5.0, "twotheta_max": 90.0, "step_size": 0.01}
NUM_POINTS = int((INSTRUMENT_PARAMS["twotheta_max"] - INSTRUMENT_PARAMS["twotheta_min"]) / INSTRUMENT_PARAMS["step_size"] + 1)
TWO_THETA = np.linspace(INSTRUMENT_PARAMS["twotheta_min"], INSTRUMENT_PARAMS["twotheta_max"], NUM_POINTS, dtype=np.float32)
X_SCALED = np.linspace(-1.0, 1.0, NUM_POINTS, dtype=np.float64)

USE_EXACT_GP = bool(int(os.environ.get("USE_EXACT_GP", "0")))
RFF_FEATURES = int(os.environ.get("RFF_FEATURES", "1024"))

NUM_SHARDS = int(os.environ.get("GENMAP_NUM_SHARDS", "1"))
SHARD_ID = int(os.environ.get("GENMAP_SHARD_ID", "0"))

_base = TOTAL_SAMPLES // NUM_SHARDS
_rem = TOTAL_SAMPLES % NUM_SHARDS
LOCAL_SAMPLES = _base + (1 if SHARD_ID < _rem else 0)

BASE_OUTPUT_HDF5 = os.environ.get("BASE_OUTPUT_HDF5", "rruff_conditioned_shard")
BATCH_ID = os.environ.get("BATCH_ID", "0")
OUTPUT_HDF5 = f"{BASE_OUTPUT_HDF5}_batch{BATCH_ID}_shard{SHARD_ID:04d}_03282026.hdf5"
RESUME_GENERATION = bool(int(os.environ.get("RESUME_GENERATION", "1")))
BATCH_FLUSH_SIZE = int(os.environ.get("BATCH_FLUSH_SIZE", "8"))
MAX_TASK_MULTIPLIER = int(os.environ.get("MAX_TASK_MULTIPLIER", "20"))
SAMPLE_RETRY_LIMIT = int(os.environ.get("SAMPLE_RETRY_LIMIT", "3"))
FAILURE_LOG_LIMIT = int(os.environ.get("FAILURE_LOG_LIMIT", "200"))

EXT_MAP_CSV = os.environ.get("EXT_MAP_CSV", "/storage/ejb12/Work/Database/FINAL_SPG_ExtG_CrysS_Table.csv")
RRUFF_PRIOR_CSV = os.environ.get(
    "RRUFF_PRIOR_CSV",
    "/work2/ALLOC3/UserE/rruff-benchmark/option1_metadata_qc_full.csv",
)
ICSD_CHEMISTRY_CSV = os.environ.get(
    "ICSD_CHEMISTRY_CSV",
    "/work2/ALLOC3/UserE/reference_data/ICSD_With_Bravais.csv",
)
FAILURE_LOG_PATH = os.environ.get("FAILURE_LOG_PATH", f"{OUTPUT_HDF5}.failures.jsonl")
_PO_BACKEND_LOGGED = False


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
_failure_log_count = 0

print(f"[INFO] Shard {SHARD_ID}/{NUM_SHARDS} -> LOCAL_SAMPLES={LOCAL_SAMPLES:,} (TOTAL_SAMPLES={TOTAL_SAMPLES:,})")
print(f"[INFO] Using {NUM_PROCESSES} worker processes")
print(f"[INFO] EG mode={EG_SAMPLING_MODE} prior_mode={PRIOR_MODE}")
print(f"[INFO] Preferred orientation enabled={ENABLE_PREFERRED_ORIENTATION}")
print(f"[INFO] RRUFF prior CSV={RRUFF_PRIOR_CSV}")
print(f"[INFO] ICSD chemistry CSV={ICSD_CHEMISTRY_CSV}")


df_ext = pd.read_csv(EXT_MAP_CSV)
sg_to_ext = {int(sg): int(eg) for sg, eg in zip(df_ext["Space Group"], df_ext["Extinction Group"])}

ext_to_sg = {}
for sg, eg in sg_to_ext.items():
    ext_to_sg.setdefault(int(eg), []).append(int(sg))

ALL_SPACE_GROUPS = np.array(sorted(sg_to_ext.keys()), dtype=np.int16)
ALL_EGS = np.array(sorted(ext_to_sg.keys()), dtype=np.int16)
if ALL_SPACE_GROUPS.size == 0:
    raise RuntimeError("EXT_MAP_CSV produced an empty SG list; check the file path and columns.")


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

if EG_SAMPLING_MODE not in {"rruff", "uniform"}:
    raise RuntimeError(f"Unsupported EG_SAMPLING_MODE='{EG_SAMPLING_MODE}'. Use 'rruff' or 'uniform'.")
if PRIOR_MODE not in {"clean", "robust"}:
    raise RuntimeError(f"Unsupported PRIOR_MODE='{PRIOR_MODE}'. Use 'clean' or 'robust'.")

if EG_SAMPLING_MODE == "uniform":
    SAMPLE_EGS = ALL_EGS
    SAMPLE_EG_WEIGHTS = np.full(ALL_EGS.shape, 1.0 / ALL_EGS.size, dtype=np.float64)
else:
    SAMPLE_EGS = EGS
    SAMPLE_EG_WEIGHTS = EG_WEIGHTS


def get_crystal_system_code(space_group: int) -> int:
    if 1 <= space_group <= 2:
        return 0
    if 3 <= space_group <= 15:
        return 1
    if 16 <= space_group <= 74:
        return 2
    if 75 <= space_group <= 142:
        return 3
    if 143 <= space_group <= 167:
        return 4
    if 168 <= space_group <= 194:
        return 5
    if 195 <= space_group <= 230:
        return 6
    return -1


def get_family_key(space_group: int) -> str:
    code = get_crystal_system_code(space_group)
    if code == 6:
        return "cubic"
    if code in {2, 3}:
        return "ortho_tetra"
    if code == 1:
        return "monoclinic"
    if code in {4, 5}:
        return "trig_hex"
    return "triclinic_other"


def sample_preferred_orientation(space_group: int, rng: np.random.Generator) -> dict:
    family = get_family_key(space_group)

    random_powder_prob = {
        "cubic": 0.65,
        "ortho_tetra": 0.40,
        "monoclinic": 0.40,
        "trig_hex": 0.40,
        "triclinic_other": 0.40,
    }[family]
    if rng.random() < random_powder_prob:
        return {"r": 1.0, "h": 0.0, "k": 0.0, "l": 1.0}

    if family == "cubic":
        plane_options = [(1, 0, 0), (1, 1, 0), (1, 1, 1)]
        plane_weights = np.array([0.4, 0.3, 0.3], dtype=np.float64)
        platy_range = (0.80, 0.95)
        needle_range = (1.05, 1.25)
    elif family == "ortho_tetra":
        plane_options = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        plane_weights = np.array([0.35, 0.25, 0.40], dtype=np.float64)
        platy_range = (0.60, 0.95)
        needle_range = (1.05, 1.60)
    elif family == "monoclinic":
        plane_options = [(0, 0, 1), (1, 0, 0), (1, 1, 0)]
        plane_weights = np.array([0.70, 0.15, 0.15], dtype=np.float64)
        platy_range = (0.55, 0.95)
        needle_range = (1.05, 1.80)
    elif family == "trig_hex":
        plane_options = [(0, 0, 1), (1, 0, 0), (1, 1, 0)]
        plane_weights = np.array([0.75, 0.15, 0.10], dtype=np.float64)
        platy_range = (0.55, 0.95)
        needle_range = (1.05, 1.80)
    else:
        plane_options = [(0, 0, 1)]
        plane_weights = np.array([1.0], dtype=np.float64)
        platy_range = (0.55, 0.95)
        needle_range = (1.05, 1.80)

    plane_weights /= plane_weights.sum()
    h, k, l = plane_options[int(rng.choice(len(plane_options), p=plane_weights))]
    if rng.random() < 0.5:
        r = float(rng.uniform(*platy_range))
    else:
        r = float(rng.uniform(*needle_range))
    return {"r": r, "h": float(h), "k": float(k), "l": float(l)}


def classify_severity_from_rwp(rwp: float) -> str:
    if rwp > 0.5:
        return "catastrophic"
    if rwp > 0.22:
        return "poor"
    if rwp > 0.12:
        return "recoverable"
    return "usable_or_better"


def _as_float(row: dict, key: str, default: float = 0.0) -> float:
    raw = row.get(key, "")
    if raw in ("", "nan", "NaN"):
        return default
    try:
        return float(raw)
    except Exception:
        return default


@lru_cache(maxsize=None)
def resolve_space_group_number(raw_sg: str) -> int:
    if raw_sg in ("", "nan", "NaN", None):
        return -1
    try:
        return int(float(raw_sg))
    except Exception:
        pass

    sg_text = str(raw_sg).strip()
    if PmgSpaceGroup is not None:
        try:
            return int(PmgSpaceGroup(sg_text).int_number)
        except Exception:
            pass

    compact = sg_text.replace(" ", "")
    crystal_guess = compact.lower()
    if any(tag in crystal_guess for tag in ["-3m", "m-3", "23", "432", "4̅3m"]):
        return 221
    if compact.startswith(("P6", "P3", "R3", "R-3", "P-3", "P63", "R32")):
        return 168
    if compact.startswith(("P4", "I4", "P-4", "I-4")):
        return 75
    if compact.startswith(("P2", "C2", "A2", "B2", "I2")):
        return 3
    if compact.startswith(("P1", "P-1")):
        return 1
    if compact.startswith(("Pn", "Pb", "Pc", "Cm", "Cc", "Pm", "Im", "Fm", "Fd", "Ia", "Ib", "Ic")):
        return 16
    return -1


def load_rruff_priors(path: str, prior_mode: str) -> dict:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"RRUFF prior CSV not found: {csv_path}")

    allowed = {"usable_or_better"} if prior_mode == "clean" else {"usable_or_better", "recoverable"}
    by_family: dict[str, list[dict]] = {k: [] for k in ["cubic", "ortho_tetra", "monoclinic", "trig_hex", "triclinic_other"]}
    global_rows: list[dict] = []

    with csv_path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sg = resolve_space_group_number(row.get("space_group", ""))
            if sg <= 0:
                continue
            severity = classify_severity_from_rwp(_as_float(row, "rwp", 1.0))
            if severity not in allowed:
                continue
            prior_row = {
                "space_group": sg,
                "family": get_family_key(sg),
                "severity": severity,
                "U": max(0.0, _as_float(row, "U")),
                "V": _as_float(row, "V"),
                "W": max(0.0, _as_float(row, "W")),
                "X": max(0.0, _as_float(row, "X")),
                "Y": max(0.0, _as_float(row, "Y")),
                "eta": float(np.clip(_as_float(row, "eta"), 0.0, 1.0)),
                "zero": _as_float(row, "zero"),
                "bkg": np.array([_as_float(row, f"bkg_c{i}") for i in range(4)], dtype=np.float64),
                "rwp": _as_float(row, "rwp"),
            }
            by_family[prior_row["family"]].append(prior_row)
            global_rows.append(prior_row)

    if not global_rows:
        raise RuntimeError(f"No usable prior rows loaded from {csv_path} for PRIOR_MODE={prior_mode}")

    print("[INFO] Loaded empirical priors:")
    for family, rows in by_family.items():
        print(f"  - {family}: {len(rows)}")
    print(f"  - global: {len(global_rows)}")
    return {"by_family": by_family, "global": global_rows}


PRIORS = load_rruff_priors(RRUFF_PRIOR_CSV, PRIOR_MODE)


def reduce_integer_ratios(values: list[int]) -> list[int]:
    out = [max(1, int(v)) for v in values]
    g = 0
    for value in out:
        g = math.gcd(g, value)
    if g > 1:
        out = [max(1, v // g) for v in out]
    return out


def parse_formula_species_counts(formula: str) -> tuple[list[str], list[int]] | None:
    if not formula or not isinstance(formula, str):
        return None
    tokens = FORMULA_TOKEN_RE.findall(formula.strip())
    if not tokens:
        return None

    elements: list[str] = []
    counts: list[float] = []
    for symbol, raw_count in tokens:
        if symbol not in VALID_ELEMENT_SYMBOLS:
            continue
        if symbol not in ALLOWED_CHEMISTRY_SYMBOLS:
            continue
        try:
            count = float(raw_count) if raw_count else 1.0
        except ValueError:
            count = 1.0
        if count <= 0:
            continue
        elements.append(symbol)
        counts.append(count)

    if not elements:
        return None

    min_count = min(counts)
    ratios = [max(1, int(round(c / min_count))) for c in counts]
    ratios = reduce_integer_ratios(ratios)

    total = sum(ratios)
    if total > 16:
        scale = max(1, int(round(total / 12)))
        ratios = [max(1, int(round(v / scale))) for v in ratios]
        ratios = reduce_integer_ratios(ratios)

    return elements, ratios


def load_chemistry_priors(path: str) -> dict:
    csv_path = Path(path)
    families = ["cubic", "ortho_tetra", "monoclinic", "trig_hex", "triclinic_other"]
    by_family = {
        k: {"element_counts": Counter(), "pair_counts": Counter(), "num_elements_counts": Counter(), "ratio_bank": []}
        for k in families
    }
    global_stats = {"element_counts": Counter(), "pair_counts": Counter(), "num_elements_counts": Counter(), "ratio_bank": []}

    if not csv_path.exists():
        print(f"[WARN] Chemistry prior CSV not found: {csv_path}. Falling back to broad element sampling.")
        return {"by_family": by_family, "global": global_stats}

    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sg = resolve_space_group_number(row.get("sym_group", "")) or resolve_space_group_number(row.get("Space Group", ""))
            if sg <= 0:
                continue
            parsed = parse_formula_species_counts(row.get("name", ""))
            if parsed is None:
                continue
            elements, ratios = parsed
            if len(elements) > 4:
                continue
            family = get_family_key(sg)
            stats = by_family[family]
            unique_elements = tuple(sorted(set(elements)))
            stats["num_elements_counts"][len(unique_elements)] += 1
            global_stats["num_elements_counts"][len(unique_elements)] += 1
            stats["ratio_bank"].append(tuple(ratios))
            global_stats["ratio_bank"].append(tuple(ratios))
            for element in unique_elements:
                stats["element_counts"][element] += 1
                global_stats["element_counts"][element] += 1
            for pair in combinations(unique_elements, 2):
                stats["pair_counts"][tuple(sorted(pair))] += 1
                global_stats["pair_counts"][tuple(sorted(pair))] += 1

    print("[INFO] Loaded chemistry priors:")
    for family, stats in by_family.items():
        print(f"  - {family}: {sum(stats['num_elements_counts'].values())}")
    print(f"  - global: {sum(global_stats['num_elements_counts'].values())}")
    return {"by_family": by_family, "global": global_stats}


CHEMISTRY_PRIORS = load_chemistry_priors(ICSD_CHEMISTRY_CSV)

CHEMISTRY_MARGINAL_TEMPER = float(os.environ.get("CHEMISTRY_MARGINAL_TEMPER", "0.5"))
CHEMISTRY_PAIR_TEMPER = float(os.environ.get("CHEMISTRY_PAIR_TEMPER", "0.5"))
CHEMISTRY_SMOOTH = float(os.environ.get("CHEMISTRY_SMOOTH", "1.0"))


def _normalized_element_probs(stats: dict) -> dict[str, float]:
    counts = stats["element_counts"]
    if not counts:
        return {}
    total = float(sum(counts.values()))
    denom = total + CHEMISTRY_SMOOTH * len(counts)
    return {element: (count + CHEMISTRY_SMOOTH) / denom for element, count in counts.items()}


def _pair_affinity(stats: dict, a: str, b: str, element_probs: dict[str, float]) -> float:
    pair = tuple(sorted((a, b)))
    pair_counts = stats["pair_counts"]
    pair_total = float(sum(pair_counts.values()))
    num_pairs = max(1, len(pair_counts))
    p_pair = (pair_counts.get(pair, 0.0) + CHEMISTRY_SMOOTH) / (pair_total + CHEMISTRY_SMOOTH * num_pairs)
    p_a = element_probs.get(a, 0.0)
    p_b = element_probs.get(b, 0.0)
    expected = max(p_a * p_b, 1e-12)
    return max(p_pair / expected, 1e-6)


def sample_empirical_element_set(space_group: int) -> list[str] | None:
    family = get_family_key(space_group)
    stats = CHEMISTRY_PRIORS["by_family"].get(family, {})
    if not stats or not stats["element_counts"]:
        stats = CHEMISTRY_PRIORS["global"]
    if not stats["element_counts"]:
        return None

    num_choices = sorted(stats["num_elements_counts"])
    num_weights = np.array([stats["num_elements_counts"][k] for k in num_choices], dtype=np.float64)
    num_weights /= num_weights.sum()
    target_n = int(np.random.choice(num_choices, p=num_weights))

    element_items = sorted(stats["element_counts"].items())
    elements = [k for k, _ in element_items]
    element_probs = _normalized_element_probs(stats)
    weights = np.array([element_probs[k] ** CHEMISTRY_MARGINAL_TEMPER for k in elements], dtype=np.float64)
    weights /= weights.sum()

    chosen = [str(np.random.choice(elements, p=weights))]
    while len(chosen) < target_n:
        candidate_scores = []
        candidate_elements = []
        for element in elements:
            if element in chosen:
                continue
            score = max(element_probs[element] ** CHEMISTRY_MARGINAL_TEMPER, 1e-12)
            for prior in chosen:
                affinity = _pair_affinity(stats, element, prior, element_probs)
                score *= affinity ** CHEMISTRY_PAIR_TEMPER
            candidate_elements.append(element)
            candidate_scores.append(score)
        if not candidate_elements:
            break
        candidate_scores = np.array(candidate_scores, dtype=np.float64)
        if not np.isfinite(candidate_scores).all() or candidate_scores.sum() <= 0:
            candidate_scores = np.ones(len(candidate_elements), dtype=np.float64)
        candidate_scores /= candidate_scores.sum()
        chosen.append(str(np.random.choice(candidate_elements, p=candidate_scores)))

    return chosen


ELEMENTS = [
    "H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Cl","Ar","K","Ca",
    "Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr","Rb","Sr",
    "Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I","Xe","Cs","Ba",
    "La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","Hf","Ta","W",
    "Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn","Fr","Ra","Ac","Th","Pa","U",
]
ELEMENTS = [el for el in ELEMENTS if el in ALLOWED_CHEMISTRY_SYMBOLS]


@lru_cache(maxsize=None)
def get_wyckoffs_cached(space_group):
    return tuple(get_wyckoffs(space_group, organized=True))


@lru_cache(maxsize=None)
def get_group_base_bounds(space_group: int):
    group = Group(space_group)
    if hasattr(group, "_get_base_and_upper_bounds"):
        base, upper_bounds = group._get_base_and_upper_bounds()
        return tuple(int(x) for x in base), tuple(upper_bounds)

    wyckoff_data = get_wyckoffs_cached(space_group)
    multiplicity_counts = Counter()
    for positions in wyckoff_data:
        mult = int(sum(len(pos) for pos in positions))
        if mult > 0:
            multiplicity_counts[mult] += 1
    base = tuple(sorted(multiplicity_counts))
    upper_bounds = tuple(multiplicity_counts[m] for m in base)
    return base, upper_bounds


@lru_cache(maxsize=None)
def get_group_flexible_multiplicities(space_group: int):
    group = Group(space_group)
    flexible = set()
    for wp in group:
        try:
            if wp.get_dof() > 0:
                flexible.add(int(wp.multiplicity))
        except Exception:
            continue
    return tuple(sorted(flexible))


@lru_cache(maxsize=None)
def get_group_general_multiplicity(space_group: int):
    group = Group(space_group)
    mults = []
    for wp in group:
        try:
            if wp.get_dof() > 0:
                mults.append(int(wp.multiplicity))
        except Exception:
            continue
    if not mults:
        return None
    return max(mults)


def sample_wyckoff_compatible_counts(
    space_group: int,
    n_species: int,
    max_extra_orbits: int = 2,
    max_attempts: int = 100,
    require_general_position: bool = False,
) -> list[int] | None:
    if n_species <= 0:
        return None

    group = Group(space_group)
    base, upper_bounds = get_group_base_bounds(space_group)
    flexible_mults = set(get_group_flexible_multiplicities(space_group))
    general_mult = get_group_general_multiplicity(space_group)
    if not base:
        return None

    weights = np.array([1.0 / max(float(mult), 1.0) for mult in base], dtype=np.float64)
    weights /= weights.sum()

    for _ in range(max_attempts):
        remaining = [None if ub is None else int(ub) for ub in upper_bounds]
        counts = [0] * n_species

        def draw_mult(prefer_flexible: bool = False, require_general: bool = False) -> int | None:
            available_idx = [i for i, ub in enumerate(remaining) if ub is None or ub > 0]
            if not available_idx:
                return None
            if require_general and general_mult is not None:
                general_idx = [i for i in available_idx if int(base[i]) == int(general_mult)]
                if general_idx:
                    available_idx = general_idx
            if prefer_flexible:
                flexible_idx = [i for i in available_idx if int(base[i]) in flexible_mults]
                if flexible_idx:
                    available_idx = flexible_idx
            local_weights = np.array([weights[i] for i in available_idx], dtype=np.float64)
            local_weights /= local_weights.sum()
            choice = int(np.random.choice(available_idx, p=local_weights))
            if remaining[choice] is not None:
                remaining[choice] -= 1
            return int(base[choice])

        ok = True
        for i in range(n_species):
            mult = draw_mult(
                prefer_flexible=(i == 0 and bool(flexible_mults)),
                require_general=(i == 0 and require_general_position and general_mult is not None),
            )
            if mult is None:
                ok = False
                break
            counts[i] += mult
        if not ok:
            continue

        extra_orbits = int(np.random.randint(0, max_extra_orbits + 1))
        for _ in range(extra_orbits):
            mult = draw_mult()
            if mult is None:
                break
            counts[int(np.random.randint(0, n_species))] += mult

        compat, _ = group.check_compatible(counts)
        if compat:
            return counts

    return None


def sample_valid_composition(space_group, require_general_position: bool = False):
    sampled = sample_empirical_element_set(space_group)
    if sampled is not None:
        counts = sample_wyckoff_compatible_counts(
            space_group,
            len(sampled),
            require_general_position=require_general_position,
        )
        if counts is not None:
            return sampled, counts, "empirical_wyckoff"

    wyckoff_data = get_wyckoffs_cached(space_group)
    valid_counts = []
    for positions in wyckoff_data:
        total_atoms = sum(len(pos) for pos in positions)
        if total_atoms > 1:
            valid_counts.append(total_atoms)

    if not valid_counts:
        return ["C"], [2], "fallback_no_wyckoff"

    num_elements = np.random.randint(2, 4)
    elements = random.sample(ELEMENTS, num_elements)
    counts = sample_wyckoff_compatible_counts(
        space_group,
        num_elements,
        require_general_position=require_general_position,
    )
    if counts is not None:
        return elements, counts, "fallback_wyckoff"

    fallback_atoms = max(2, min(valid_counts))
    return elements[:1], [fallback_atoms], "fallback_minimal"


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


def detect_structure_space_group(structure) -> tuple[int | None, str | None]:
    atoms = structure.to_ase()
    lattice = atoms.get_cell()
    positions = atoms.get_scaled_positions()
    numbers = atoms.get_atomic_numbers()
    cell = (lattice, positions, numbers)
    detected_group = spglib.get_spacegroup(cell, symprec=1e-3)
    if not detected_group:
        return None, None
    m = re.search(r"\((\d+)\)", detected_group)
    if not m:
        return None, detected_group
    return int(m.group(1)), detected_group


def validate_structure(structure, expected_space_group):
    detected_number, detected_label = detect_structure_space_group(structure)
    return bool(detected_number == expected_space_group)


def log_failure_event(event: dict):
    global _failure_log_count
    if FAILURE_LOG_LIMIT <= 0 or _failure_log_count >= FAILURE_LOG_LIMIT:
        return
    try:
        Path(FAILURE_LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
        payload = dict(event)
        payload.setdefault("pid", os.getpid())
        with open(FAILURE_LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, sort_keys=True) + "\n")
        _failure_log_count += 1
    except Exception:
        pass


def get_valid_structure(space_group, max_attempts=50, debug_task_id=None, role="main"):
    attempts = 0
    target_eg = sg_to_ext.get(int(space_group))
    while attempts < max_attempts:
        elements = None
        ratios = None
        composition_source = None
        try:
            if space_group == 1:
                structure = generate_p1_structure()
                composition_source = "p1_random"
            else:
                elements, ratios, composition_source = sample_valid_composition(
                    space_group,
                    require_general_position=(role == "main"),
                )
                structure = pyxtal()
                structure.from_random(dim=3, group=space_group, species=elements, numIons=ratios)
            detected_number, detected_label = detect_structure_space_group(structure)
            detected_eg = sg_to_ext.get(int(detected_number)) if detected_number is not None else None
            if role == "impurity":
                return structure
            if detected_eg == target_eg and detected_eg is not None:
                return structure
            log_failure_event(
                {
                    "stage": "validate_structure_false",
                    "task_id": debug_task_id,
                    "role": role,
                    "space_group": int(space_group),
                    "family": get_family_key(space_group),
                    "attempt": attempts + 1,
                    "composition_source": composition_source,
                    "elements": elements,
                    "ratios": ratios,
                    "target_space_group": int(space_group),
                    "target_extinction_group": int(target_eg) if target_eg is not None else None,
                    "detected_space_group": detected_number,
                    "detected_space_group_label": detected_label,
                    "detected_extinction_group": int(detected_eg) if detected_eg is not None else None,
                }
            )
        except Exception as exc:
            log_failure_event(
                {
                    "stage": "pyxtal_from_random" if space_group != 1 else "p1_generation",
                    "task_id": debug_task_id,
                    "role": role,
                    "space_group": int(space_group),
                    "family": get_family_key(space_group),
                    "attempt": attempts + 1,
                    "composition_source": composition_source,
                    "elements": elements,
                    "ratios": ratios,
                    "exc_type": type(exc).__name__,
                    "exc_message": str(exc),
                }
            )
        attempts += 1
    return None


def extract_structural_info(structure):
    atoms_obj = structure.to_ase()
    a, b, c, alpha, beta, gamma = atoms_obj.get_cell_lengths_and_angles()
    sg = structure.group.number
    positions = atoms_obj.get_scaled_positions()
    symbols = atoms_obj.get_chemical_symbols()
    atom_list = [{"symbol": s, "position": p} for s, p in zip(symbols, positions)]
    return sg, a, b, c, alpha, beta, gamma, atom_list


def sample_empirical_nuisance(space_group: int, rng: np.random.Generator) -> dict:
    family = get_family_key(space_group)
    pool = PRIORS["by_family"].get(family) or PRIORS["global"]
    row = pool[int(rng.integers(0, len(pool)))]

    u = max(5e-4, row["U"] * np.exp(rng.normal(0.0, 0.12)))
    w = max(5e-4, row["W"] * np.exp(rng.normal(0.0, 0.15)))
    x = max(0.0, row["X"] * np.exp(rng.normal(0.0, 0.20)))
    y = max(0.0, row["Y"] * np.exp(rng.normal(0.0, 0.30)))
    eta = float(np.clip(row["eta"] + rng.normal(0.0, 0.06), 0.0, 1.0))
    zero = 0.0

    if family == "cubic":
        v_scale = 4e-4
    elif family == "ortho_tetra":
        v_scale = 9e-4
    elif family == "monoclinic":
        v_scale = 1.4e-3
    else:
        v_scale = 2.0e-3
    v = float(np.clip(-abs(rng.normal(0.0, v_scale)), -0.05, 0.0))

    bkg = row["bkg"].copy()
    bkg_jitter = np.array([0.10, 0.15, 0.20, 0.20], dtype=np.float64)
    bkg *= np.exp(rng.normal(0.0, bkg_jitter))

    po_cfg = sample_preferred_orientation(space_group, rng) if ENABLE_PREFERRED_ORIENTATION else {
        "r": 1.0,
        "h": 0.0,
        "k": 0.0,
        "l": 1.0,
    }
    return {
        "u": u,
        "v": v,
        "w": w,
        "x": x,
        "y": y,
        "eta": eta,
        "zero": zero,
        "bkg": bkg,
        "po_r": float(po_cfg["r"]),
        "po_h": float(po_cfg["h"]),
        "po_k": float(po_cfg["k"]),
        "po_l": float(po_cfg["l"]),
    }


def create_pattern_dict(space_group, a, b, c, alpha, beta, gamma, atom_list, nuisance, wavelength):
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
                    "_pd_instr_resolution_u": float(nuisance["u"]),
                    "_pd_instr_resolution_v": float(nuisance["v"]),
                    "_pd_instr_resolution_w": float(nuisance["w"]),
                    "_pd_instr_resolution_x": float(nuisance["x"]),
                    "_pd_instr_resolution_y": float(nuisance["y"]),
                    "_pd_instr_reflex_asymmetry_p1": 0.0,
                    "_pd_instr_reflex_asymmetry_p2": 0.0,
                    "_pd_instr_reflex_asymmetry_p3": 0.0,
                    "_pd_instr_reflex_asymmetry_p4": 0.0,
                    "_pd_meas_2theta_offset": float(nuisance["zero"]),
                    "_pd_meas_2theta_range_min": float(INSTRUMENT_PARAMS["twotheta_min"]),
                    "_pd_meas_2theta_range_max": float(INSTRUMENT_PARAMS["twotheta_max"]),
                    "_pd_meas_2theta_range_inc": float(INSTRUMENT_PARAMS["step_size"]),
                    "_pd_proc_ls_pref_orient_corr": float(nuisance.get("po_r", 1.0)),
                    "_pd_proc_ls_pref_orient_h": float(nuisance.get("po_h", 0.0)),
                    "_pd_proc_ls_pref_orient_k": float(nuisance.get("po_k", 0.0)),
                    "_pd_proc_ls_pref_orient_l": float(nuisance.get("po_l", 1.0)),
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
    global _PO_BACKEND_LOGGED
    npd = pattern_dict["experiments"][0]["NPD"]
    po_r = float(npd.get("_pd_proc_ls_pref_orient_corr", 1.0))
    po_h = float(npd.get("_pd_proc_ls_pref_orient_h", 0.0))
    po_k = float(npd.get("_pd_proc_ls_pref_orient_k", 0.0))
    po_l = float(npd.get("_pd_proc_ls_pref_orient_l", 1.0))
    po_enabled = abs(po_r - 1.0) > 1e-6
    if po_enabled and not _PO_BACKEND_LOGGED:
        print(
            "[INFO] Preferred orientation engaged in CFML_api backend: "
            f"r={po_r:.4f} hkl=({po_h:.0f},{po_k:.0f},{po_l:.0f})"
        )
        _PO_BACKEND_LOGGED = True

    if cfml_py_utilities is not None and not po_enabled:
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
                "--po-r",
                str(po_r),
                "--po-h",
                str(po_h),
                "--po-k",
                str(po_k),
                "--po-l",
                str(po_l),
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
        for path in (cif_path, out_path):
            try:
                os.remove(path)
            except OSError:
                pass


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


def _smooth_field(twotheta: np.ndarray, length_scale: float, rng: np.random.Generator) -> np.ndarray:
    field = _rbf_gp_exact(twotheta, length_scale, rng) if USE_EXACT_GP else _rbf_gp_rff(twotheta, length_scale, rng, RFF_FEATURES)
    field -= field.mean()
    std = float(field.std())
    if std > 0:
        field /= std
    return field


def sample_amplitude_distortion_config(space_group: int, rng: np.random.Generator) -> tuple[bool, float]:
    family = get_family_key(space_group)
    base_prob = {
        "cubic": 0.05,
        "ortho_tetra": 0.10,
        "monoclinic": 0.15,
        "trig_hex": 0.15,
        "triclinic_other": 0.20,
    }[family]
    if PRIOR_MODE == "robust":
        base_prob += 0.05
    if EG_SAMPLING_MODE == "rruff":
        base_prob += 0.03
    enabled = bool(rng.random() < min(base_prob, 0.35))

    strength = {
        "cubic": 0.02,
        "ortho_tetra": 0.04,
        "monoclinic": 0.06,
        "trig_hex": 0.06,
        "triclinic_other": 0.08,
    }[family]
    return enabled, strength


def apply_amplitude_distortion(intensity_01: np.ndarray, twotheta: np.ndarray, rng: np.random.Generator, strength: float) -> np.ndarray:
    if strength <= 0.0:
        return intensity_01.astype(np.float32, copy=True)
    field = _smooth_field(twotheta, float(rng.uniform(2.5, 9.0)), rng)
    envelope = np.exp(strength * field)
    envelope /= np.median(envelope)
    out = intensity_01.astype(np.float64, copy=True) * envelope
    peak = float(out.max())
    if peak > 0:
        out /= peak
    return out.astype(np.float32)


def evaluate_empirical_background(nuisance: dict) -> np.ndarray:
    coeffs = nuisance["bkg"] / MAX_INTENSITY
    bg = np.polynomial.chebyshev.chebval(X_SCALED, coeffs)
    return bg.astype(np.float64)


def add_rruff_background_noise(
    intensity_counts: np.ndarray,
    twotheta: np.ndarray,
    rng: np.random.Generator,
    nuisance: dict,
    space_group: int,
) -> np.ndarray:
    counts = intensity_counts.astype(np.float64, copy=True)
    counts += evaluate_empirical_background(nuisance) * MAX_INTENSITY

    family = get_family_key(space_group)
    gp_scale = {
        "cubic": (0.0, 10.0),
        "ortho_tetra": (5.0, 20.0),
        "monoclinic": (8.0, 30.0),
        "trig_hex": (8.0, 30.0),
        "triclinic_other": (10.0, 40.0),
    }[family]

    field = _smooth_field(twotheta, float(rng.uniform(7.0, 35.0)), rng)
    gp = field - field.min()
    total = float(gp.sum())
    if total > 0:
        gp /= total
        gp *= float(rng.uniform(*gp_scale))
        counts += gp

    sigma_add = float(rng.uniform(0.5, 8.0))
    counts += rng.normal(0.0, sigma_add, size=counts.shape)

    sigma_mult = float(rng.uniform(0.0, 0.10 if PRIOR_MODE == "clean" else 0.14))
    counts *= rng.normal(1.0, sigma_mult, size=counts.shape)

    return np.clip(counts, 0.0, None).astype(np.float32)


def generate_pure_pattern_01(
    structure,
    rng: np.random.Generator,
    nuisance: dict,
    apply_distortion: bool = True,
) -> tuple[int, np.ndarray]:
    sg, a, b, c, alpha, beta, gamma, atom_list = extract_structural_info(structure)

    p_k1 = create_pattern_dict(sg, a, b, c, alpha, beta, gamma, atom_list, nuisance, KALPHA1)
    p_k2 = create_pattern_dict(sg, a, b, c, alpha, beta, gamma, atom_list, nuisance, KALPHA2)

    _, y1 = cw_powder_pattern_backend(p_k1, structure)
    _, y2 = cw_powder_pattern_backend(p_k2, structure)

    intensity = (W_KALPHA1 * np.asarray(y1, dtype=np.float32) + W_KALPHA2 * np.asarray(y2, dtype=np.float32)).astype(np.float32)
    max_val = float(intensity.max())
    if max_val > 0:
        intensity = (intensity / max_val).astype(np.float32)

    distort, strength = sample_amplitude_distortion_config(sg, rng)
    if apply_distortion and distort:
        intensity = apply_amplitude_distortion(intensity, TWO_THETA, rng, strength)

    return sg, intensity


def sample_impurity_fraction(rng: np.random.Generator) -> float:
    p = float(rng.random())
    if p < 0.70:
        return 0.0
    if p < 0.90:
        return float(rng.uniform(0.005, 0.03))
    return float(rng.uniform(0.03, 0.10))


def mix_impurity(main_01: np.ndarray, imp_01: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    frac = sample_impurity_fraction(rng)
    if frac <= 0.0:
        return main_01.astype(np.float32, copy=True)
    return ((1.0 - frac) * main_01 + frac * imp_01).astype(np.float32)


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


def sample_impurity_sg(rng: np.random.Generator) -> int:
    eg = int(rng.choice(SAMPLE_EGS, p=SAMPLE_EG_WEIGHTS))
    sgs = ext_to_sg[eg]
    return int(sgs[int(rng.integers(0, len(sgs)))])


def compute_stream_seed() -> int:
    key = (
        f"{OUTPUT_HDF5}|{EG_SAMPLING_MODE}|{PRIOR_MODE}|po={int(ENABLE_PREFERRED_ORIENTATION)}|"
        f"{SHARD_ID}|{NUM_SHARDS}|{TOTAL_SAMPLES}"
    )
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
                    "prior_mode": PRIOR_MODE,
                    "shard_id": SHARD_ID,
                    "num_shards": NUM_SHARDS,
                    "total_samples": TOTAL_SAMPLES,
                    "local_samples_target": LOCAL_SAMPLES,
                    "stream_seed": stream_seed,
                    "preferred_orientation_enabled": int(ENABLE_PREFERRED_ORIENTATION),
                }
                for key, value in expected.items():
                    if key in root.attrs and root.attrs[key] != value:
                        raise RuntimeError(f"Existing shard metadata mismatch for {key}: {root.attrs[key]} != {value}")
                if "2theta" not in f:
                    f.create_dataset("2theta", data=TWO_THETA, dtype="float32")
                root.attrs["twotheta_min"] = float(INSTRUMENT_PARAMS["twotheta_min"])
                root.attrs["twotheta_max"] = float(INSTRUMENT_PARAMS["twotheta_max"])
                root.attrs["step_size"] = float(INSTRUMENT_PARAMS["step_size"])
                po_desc = " + exact March-Dollase preferred orientation" if ENABLE_PREFERRED_ORIENTATION else ""
                root.attrs["description"] = f"{EG_SAMPLING_MODE.upper()} EG dist synthetic XRD with empirical RRUFF nuisance priors{po_desc}"
                root.attrs["version"] = f"v1_rruff_conditioned_{EG_SAMPLING_MODE}_{PRIOR_MODE}{'_po' if ENABLE_PREFERRED_ORIENTATION else ''}"
                root.attrs["eg_sampling_mode"] = EG_SAMPLING_MODE
                root.attrs["prior_mode"] = PRIOR_MODE
                root.attrs["preferred_orientation_enabled"] = int(ENABLE_PREFERRED_ORIENTATION)
                root.attrs["rruff_prior_csv"] = RRUFF_PRIOR_CSV
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
        po_desc = " + exact March-Dollase preferred orientation" if ENABLE_PREFERRED_ORIENTATION else ""
        root.attrs["description"] = f"{EG_SAMPLING_MODE.upper()} EG dist synthetic XRD with empirical RRUFF nuisance priors{po_desc}"
        root.attrs["version"] = f"v1_rruff_conditioned_{EG_SAMPLING_MODE}_{PRIOR_MODE}{'_po' if ENABLE_PREFERRED_ORIENTATION else ''}"
        root.attrs["eg_sampling_mode"] = EG_SAMPLING_MODE
        root.attrs["prior_mode"] = PRIOR_MODE
        root.attrs["preferred_orientation_enabled"] = int(ENABLE_PREFERRED_ORIENTATION)
        root.attrs["rruff_prior_csv"] = RRUFF_PRIOR_CSV
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


def worker(task):
    task_id, eg, sg_main, sg_imp, seed = task
    rng = np.random.default_rng(seed)

    last_error = None
    for _ in range(SAMPLE_RETRY_LIMIT):
        try:
            st_main = get_valid_structure(sg_main, debug_task_id=task_id, role="main")
            if st_main is None:
                continue

            st_imp = get_valid_structure(sg_imp, debug_task_id=task_id, role="impurity")
            if st_imp is None:
                continue

            nuisance = sample_empirical_nuisance(sg_main, rng)
            sg_out, main_01 = generate_pure_pattern_01(st_main, rng, nuisance, apply_distortion=True)
            _, imp_01 = generate_pure_pattern_01(st_imp, rng, nuisance, apply_distortion=False)

            mixed_01 = mix_impurity(main_01, imp_01, rng)
            intensity_counts = (mixed_01 * MAX_INTENSITY).astype(np.float32)
            intensity_out = add_rruff_background_noise(intensity_counts, TWO_THETA, rng, nuisance, sg_out)

            if not np.all(np.isfinite(intensity_out)):
                raise RuntimeError("Generated intensity contains non-finite values")
            return (task_id, sg_out, eg, intensity_out.astype(np.float32))
        except Exception as exc:
            last_error = exc
            continue

    if last_error is not None:
        print(f"[WARN] Worker dropped task after {SAMPLE_RETRY_LIMIT} attempts (SG_main={sg_main}, SG_imp={sg_imp}): {last_error}")
    return (task_id, None)


def build_task(task_id: int, seed: int):
    rng = np.random.default_rng(np.random.SeedSequence([seed, task_id]))
    if EG_SAMPLING_MODE == "uniform":
        eg, sg_main = sample_main_sg_balanced(task_id, seed)
    else:
        eg, sg_main = sample_main_sg(rng)
    sg_imp = sample_impurity_sg(rng)
    wseed = int(rng.integers(0, 2**63 - 1))
    return (task_id, eg, sg_main, sg_imp, wseed)


def task_stream(max_tasks: int, seed: int, start_task_id: int = 0):
    for task_id in range(start_task_id, max_tasks):
        yield build_task(task_id, seed)


def main():
    print(f"\n[INFO] Starting generation ({EG_SAMPLING_MODE.upper()} EG dist + empirical nuisance priors)...\n")
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
                    raise RuntimeError(f"Shard {SHARD_ID} exhausted max task budget ({max_tasks}) before reaching {LOCAL_SAMPLES} samples")
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
