#!/usr/bin/env python3
"""
Minimal Gate 2 discrete-channel validation on Stampede.

This script uses the existing PyXtal + PyCrysFML generation path to produce a
small simulator-conditioned absence-like channel over a fixed set of low-index
reflection families. It is intentionally minimal:

- one clean condition: narrow peaks, low overlap
- one degraded condition: broader peaks, more overlap -> more "unresolved"

The goal is not a publication-grade ladder. It is only to verify that the
real structure-generation + mask-extraction pipeline yields sane information
numbers with the existing `discrete_channel.py` machinery.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
import random
import subprocess
import sys
import tempfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import spglib
from pyxtal import pyxtal
from pyxtal.symmetry import get_wyckoffs
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.io.ase import AseAtomsAdaptor

try:
    from pycrysfml import cfml_py_utilities  # type: ignore
except ImportError:
    cfml_py_utilities = None

try:
    import CFML_api  # type: ignore
except ImportError:
    CFML_api = None

from discrete_channel import encode_mask_vector, estimate_channel, full_summary

_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[1]
_XRD_GENERATOR_DIR = _REPO_ROOT / "Code" / "ViT_Streaming"
if str(_XRD_GENERATOR_DIR) not in sys.path:
    sys.path.insert(0, str(_XRD_GENERATOR_DIR))

try:
    from xrd_generator import get_valid_structure as production_get_valid_structure  # type: ignore
except Exception:
    production_get_valid_structure = None


ELEMENTS = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si",
    "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni",
    "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb",
    "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho",
    "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U",
]

FAMILY_SPECS = [
    ("100", (1, 0, 0)),
    ("010", (0, 1, 0)),
    ("001", (0, 0, 1)),
    ("110", (1, 1, 0)),
    ("101", (1, 0, 1)),
    ("011", (0, 1, 1)),
]

CLEAN_NUISANCE = {"u": 0.008, "v": -0.004, "w": 0.008, "x": 0.0010, "y": 0.0}
DEGRADED_NUISANCE = {"u": 0.120, "v": -0.030, "w": 0.120, "x": 0.0120, "y": 0.0030}
CFML_HELPER = Path(__file__).resolve().parents[1] / "Code" / "Generation" / "tacc" / "cfml_api_single_pattern.py"


def parse_args():
    parser = argparse.ArgumentParser(description="Run minimal Gate 2 discrete-channel sanity check.")
    parser.add_argument("--mapping-csv", required=True)
    parser.add_argument("--samples-per-eg", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-attempts", type=int, default=150)
    parser.add_argument("--wavelength", type=float, default=1.54030)
    parser.add_argument("--tt-min", type=float, default=10.0)
    parser.add_argument("--tt-max", type=float, default=110.0)
    parser.add_argument("--tt-step", type=float, default=0.1)
    parser.add_argument("--present-threshold-frac", type=float, default=0.03)
    parser.add_argument("--overlap-window-deg-clean", type=float, default=0.12)
    parser.add_argument("--overlap-window-deg-degraded", type=float, default=0.32)
    parser.add_argument("--sample-retry-limit", type=int, default=3)
    parser.add_argument("--num-processes", type=int, default=0)
    parser.add_argument("--eg-index-start", type=int, default=0)
    parser.add_argument("--eg-index-stop", type=int, default=0)
    parser.add_argument("--skip-summary", action="store_true")
    parser.add_argument("--dirichlet-alpha", type=float, default=0.5)
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument("--progress-json", default="")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-npz", default="")
    return parser.parse_args()


def get_num_processes(requested: int) -> int:
    if requested > 0:
        return requested
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        try:
            n = int(slurm_cpus)
            if n > 0:
                return n
        except ValueError:
            pass
    return os.cpu_count() or 1


def load_eg_to_sgs(mapping_csv: str) -> list[tuple[int, list[int]]]:
    rows: list[tuple[int, list[int]]] = []
    with open(mapping_csv, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for eg_idx, row in enumerate(reader):
            sgs = ast.literal_eval(row["Space Group Numbers"])
            if not sgs:
                continue
            rows.append((eg_idx, [int(v) for v in sgs]))
    return rows


def sample_valid_composition(space_group: int):
    wyckoff_data = get_wyckoffs(space_group, organized=True)
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
    ratios = [random.choice(valid_ratios) for _ in range(num_elements)]
    return elements, ratios


def validate_structure(structure, expected_space_group: int) -> bool:
    atoms = structure.to_ase()
    lattice = atoms.get_cell()
    positions = atoms.get_scaled_positions()
    numbers = atoms.get_atomic_numbers()
    detected_group = spglib.get_spacegroup((lattice, positions, numbers), symprec=1e-2)
    return detected_group is not None and f"({expected_space_group})" in detected_group


def get_valid_structure(space_group: int, max_attempts: int):
    if production_get_valid_structure is not None:
        structure = production_get_valid_structure(space_group, max_attempts=max_attempts, role="main")
        if structure is not None:
            return structure
    last_error = None
    for _ in range(max_attempts):
        try:
            structure = pyxtal()
            elements, ratios = sample_valid_composition(space_group)
            structure.from_random(dim=3, group=space_group, species=elements, numIons=ratios)
            if validate_structure(structure, space_group):
                return structure
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise RuntimeError(f"failed to generate SG {space_group}: {last_error}")


def create_pattern_json(structure, wavelength: float, tt_min: float, tt_max: float, tt_step: float, nuisance: dict[str, float]):
    atoms = structure.to_ase()
    a, b, c, alpha, beta, gamma = atoms.get_cell_lengths_and_angles()
    positions = atoms.get_scaled_positions()
    symbols = atoms.get_chemical_symbols()
    atom_list = [{"symbol": sym, "position": pos} for sym, pos in zip(symbols, positions)]
    sg = int(structure.group.number)
    return {
        "phases": [
            {
                "placeholder": {
                    "_space_group_name_H-M_alt": spglib.get_spacegroup_type(sg)["international_short"],
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
                    "_pd_instr_reflex_asymmetry_p1": 0,
                    "_pd_instr_reflex_asymmetry_p2": 0,
                    "_pd_instr_reflex_asymmetry_p3": 0,
                    "_pd_instr_reflex_asymmetry_p4": 0,
                    "_pd_meas_2theta_offset": 0,
                    "_pd_meas_2theta_range_min": float(tt_min),
                    "_pd_meas_2theta_range_max": float(tt_max),
                    "_pd_meas_2theta_range_inc": float(tt_step),
                    "_phase": [{"_label": "placeholder", "_scale": 1}],
                    "_pd_background": [
                        {"_2theta": float(tt_min), "_intensity": 0},
                        {"_2theta": float(tt_max), "_intensity": 0},
                    ],
                }
            }
        ],
    }


def cw_powder_pattern_backend(pattern_dict: dict, structure):
    if cfml_py_utilities is not None:
        return cfml_py_utilities.cw_powder_pattern_from_dict(pattern_dict)

    if CFML_api is None:
        raise RuntimeError("Neither pycrysfml.cfml_py_utilities nor CFML_api is available")
    if not CFML_HELPER.exists():
        raise RuntimeError(f"CFML_api helper not found at {CFML_HELPER}")

    atoms = structure.to_ase()
    npd = pattern_dict["experiments"][0]["NPD"]
    with tempfile.NamedTemporaryFile(suffix=".cif", delete=False) as handle:
        cif_path = handle.name
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as handle:
        out_path = handle.name

    try:
        from ase.io import write as ase_write

        ase_write(cif_path, atoms, format="cif")
        subprocess.run(
            [
                sys.executable,
                str(CFML_HELPER),
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
            x = np.asarray(data["x"], dtype=np.float64)
            y = np.asarray(data["y"], dtype=np.float64)
        return x, y
    finally:
        for path in (cif_path, out_path):
            try:
                os.remove(path)
            except OSError:
                pass


def normalize_hkl(entry) -> tuple[int, int, int] | None:
    if isinstance(entry, dict):
        if "hkl" in entry:
            return normalize_hkl(entry["hkl"])
        return None
    if isinstance(entry, np.ndarray):
        entry = entry.tolist()
    if isinstance(entry, (list, tuple)):
        if len(entry) == 3 and all(np.isscalar(v) for v in entry):
            return tuple(int(v) for v in entry)
        if len(entry) > 0:
            # Some pymatgen payloads wrap a single HKL tuple inside another sequence.
            first = entry[0]
            if isinstance(first, (list, tuple, np.ndarray, dict)):
                return normalize_hkl(first)
    return None


def family_key(hkl: tuple[int, int, int] | None) -> str | None:
    if hkl is None:
        return None
    ah, ak, al = (abs(int(v)) for v in hkl)
    target = (ah, ak, al)
    for name, proto in FAMILY_SPECS:
        if target == proto:
            return name
    return None


def reference_family_peaks(structure, wavelength: float, tt_min: float, tt_max: float) -> dict[str, list[float]]:
    pmg = AseAtomsAdaptor.get_structure(structure.to_ase())
    calc = XRDCalculator(wavelength=wavelength)
    pattern = calc.get_pattern(pmg, two_theta_range=(tt_min, tt_max))
    peaks: dict[str, list[float]] = {name: [] for name, _ in FAMILY_SPECS}
    for tt, hkls in zip(pattern.x, pattern.hkls):
        for entry in hkls:
            fam = family_key(normalize_hkl(entry))
            if fam is not None:
                peaks[fam].append(float(tt))
    return peaks


def observed_pattern(structure, wavelength: float, tt_min: float, tt_max: float, tt_step: float, nuisance: dict[str, float]):
    pattern_json = create_pattern_json(structure, wavelength, tt_min, tt_max, tt_step, nuisance)
    x, y = cw_powder_pattern_backend(pattern_json, structure)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return x, y


def local_peak_height(x: np.ndarray, y: np.ndarray, center: float, half_window: float) -> float:
    mask = (x >= center - half_window) & (x <= center + half_window)
    if not np.any(mask):
        return 0.0
    return float(np.max(y[mask]))


def family_mask_states(reference_peaks: dict[str, list[float]], x: np.ndarray, y: np.ndarray, overlap_window_deg: float, present_threshold_frac: float) -> list[str]:
    states: list[str] = []
    ymax = float(np.max(y)) if y.size else 0.0
    present_threshold = present_threshold_frac * ymax

    family_centers: dict[str, float | None] = {}
    for fam, centers in reference_peaks.items():
        family_centers[fam] = min(centers) if centers else None

    for fam, _proto in FAMILY_SPECS:
        center = family_centers[fam]
        if center is None:
            states.append("absent")
            continue
        unresolved = False
        for other, other_center in family_centers.items():
            if other == fam or other_center is None:
                continue
            if abs(other_center - center) <= overlap_window_deg:
                unresolved = True
                break
        if unresolved:
            states.append("unresolved")
            continue
        height = local_peak_height(x, y, center, overlap_window_deg)
        states.append("present" if height >= present_threshold else "unresolved")
    return states


def build_tasks(eg_to_sgs, samples_per_eg: int, seed: int):
    task_id = 0
    for eg, sgs in eg_to_sgs:
        for rep in range(samples_per_eg):
            seq = np.random.SeedSequence([seed, int(eg), int(rep)])
            rng = np.random.default_rng(seq)
            sg = int(rng.choice(np.asarray(sgs, dtype=np.int64)))
            wseed = int(rng.integers(0, 2**63 - 1))
            yield (task_id, int(eg), sg, int(rep), wseed)
            task_id += 1


WORKER_CFG = {}


def init_worker(cfg):
    global WORKER_CFG
    WORKER_CFG = dict(cfg)


def gate2_worker(task):
    task_id, eg, sg, rep, seed = task
    rng = np.random.default_rng(seed)
    max_attempts = int(WORKER_CFG["max_attempts"])
    wavelength = float(WORKER_CFG["wavelength"])
    tt_min = float(WORKER_CFG["tt_min"])
    tt_max = float(WORKER_CFG["tt_max"])
    tt_step = float(WORKER_CFG["tt_step"])
    present_threshold_frac = float(WORKER_CFG["present_threshold_frac"])
    overlap_window_deg_clean = float(WORKER_CFG["overlap_window_deg_clean"])
    overlap_window_deg_degraded = float(WORKER_CFG["overlap_window_deg_degraded"])
    try:
        structure = get_valid_structure(sg, max_attempts=max_attempts)
        ref = reference_family_peaks(structure, wavelength=wavelength, tt_min=tt_min, tt_max=tt_max)

        x_clean, y_clean = observed_pattern(
            structure,
            wavelength=wavelength,
            tt_min=tt_min,
            tt_max=tt_max,
            tt_step=tt_step,
            nuisance=CLEAN_NUISANCE,
        )
        clean_states = family_mask_states(
            ref,
            x_clean,
            y_clean,
            overlap_window_deg=overlap_window_deg_clean,
            present_threshold_frac=present_threshold_frac,
        )
        x_deg, y_deg = observed_pattern(
            structure,
            wavelength=wavelength,
            tt_min=tt_min,
            tt_max=tt_max,
            tt_step=tt_step,
            nuisance=DEGRADED_NUISANCE,
        )
        degraded_states = family_mask_states(
            ref,
            x_deg,
            y_deg,
            overlap_window_deg=overlap_window_deg_degraded,
            present_threshold_frac=present_threshold_frac,
        )
        return {
            "ok": True,
            "task_id": int(task_id),
            "eg": int(eg),
            "sg": int(sg),
            "rep": int(rep),
            "clean_states": clean_states,
            "clean_symbol": int(encode_mask_vector(clean_states)),
            "degraded_states": degraded_states,
            "degraded_symbol": int(encode_mask_vector(degraded_states)),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "task_id": int(task_id),
            "eg": int(eg),
            "sg": int(sg),
            "rep_target": int(rep),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def summarize_counts(rows: list[dict]) -> dict[str, int]:
    counts = {"present": 0, "absent": 0, "unresolved": 0}
    for row in rows:
        for state in row["mask_states"]:
            counts[state] += 1
    return counts


def to_jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


def write_json(path: str | Path, payload) -> None:
    Path(path).write_text(json.dumps(to_jsonable(payload), indent=2))


def write_progress_json(
    path: str | Path,
    *,
    completed_tasks: int,
    total_tasks: int,
    accepted_examples: int,
    dropped_attempts: int,
    current_task_id: int | None,
    eg_index_start: int,
    eg_index_stop: int,
) -> None:
    payload = {
        "completed_tasks": int(completed_tasks),
        "total_tasks": int(total_tasks),
        "accepted_examples": int(accepted_examples),
        "dropped_attempts": int(dropped_attempts),
        "current_task_id": None if current_task_id is None else int(current_task_id),
        "eg_index_start": int(eg_index_start),
        "eg_index_stop": int(eg_index_stop),
    }
    write_json(path, payload)


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    eg_to_sgs_all = load_eg_to_sgs(args.mapping_csv)
    eg_start = max(0, int(args.eg_index_start))
    eg_stop = int(args.eg_index_stop) if int(args.eg_index_stop) > 0 else len(eg_to_sgs_all)
    eg_to_sgs = eg_to_sgs_all[eg_start:eg_stop]
    n_classes = len(eg_to_sgs_all)
    n_symbols = 3 ** len(FAMILY_SPECS)
    prior = np.full(n_classes, 1.0 / n_classes, dtype=np.float64)
    num_processes = get_num_processes(args.num_processes)
    output = {
        "config": {
            "samples_per_eg": args.samples_per_eg,
            "seed": args.seed,
            "num_processes": num_processes,
            "eg_index_start": eg_start,
            "eg_index_stop": eg_stop,
            "n_classes_global": n_classes,
            "n_egs_in_shard": len(eg_to_sgs),
            "families": [name for name, _ in FAMILY_SPECS],
            "n_symbols": n_symbols,
            "sample_retry_limit": args.sample_retry_limit,
            "skip_summary": bool(args.skip_summary),
            "dirichlet_alpha": args.dirichlet_alpha,
            "present_threshold_frac": args.present_threshold_frac,
            "overlap_window_deg_clean": args.overlap_window_deg_clean,
            "overlap_window_deg_degraded": args.overlap_window_deg_degraded,
            "wavelength": args.wavelength,
            "tt_min": args.tt_min,
            "tt_max": args.tt_max,
            "tt_step": args.tt_step,
        }
    }
    out_npz = args.out_npz.strip() or str(Path(args.out_json).with_suffix(".npz"))
    progress_every = max(1, int(args.progress_every))
    progress_json = args.progress_json.strip() or str(Path(args.out_json).with_suffix(".progress.json"))
    worker_cfg = {
        "max_attempts": args.max_attempts,
        "wavelength": args.wavelength,
        "tt_min": args.tt_min,
        "tt_max": args.tt_max,
        "tt_step": args.tt_step,
        "present_threshold_frac": args.present_threshold_frac,
        "overlap_window_deg_clean": args.overlap_window_deg_clean,
        "overlap_window_deg_degraded": args.overlap_window_deg_degraded,
    }
    tasks = list(build_tasks(eg_to_sgs, samples_per_eg=args.samples_per_eg, seed=args.seed))
    clean_labels = []
    clean_obs = []
    degraded_labels = []
    degraded_obs = []
    example_rows = []
    dropped_examples = []
    clean_state_counts = {"present": 0, "absent": 0, "unresolved": 0}
    degraded_state_counts = {"present": 0, "absent": 0, "unresolved": 0}
    chunksize = 8
    if num_processes <= 1:
        init_worker(worker_cfg)
        iterator = (gate2_worker(task) for task in tasks)
    else:
        pool = Pool(processes=num_processes, initializer=init_worker, initargs=(worker_cfg,))
        iterator = pool.imap_unordered(gate2_worker, tasks, chunksize=chunksize)
    try:
        completed_tasks = 0
        current_task_id = None
        for res in iterator:
            completed_tasks += 1
            current_task_id = int(res["task_id"])
            if res["ok"]:
                clean_labels.append(int(res["eg"]))
                clean_obs.append(int(res["clean_symbol"]))
                degraded_labels.append(int(res["eg"]))
                degraded_obs.append(int(res["degraded_symbol"]))
                for state in res["clean_states"]:
                    clean_state_counts[state] += 1
                for state in res["degraded_states"]:
                    degraded_state_counts[state] += 1
                if len(example_rows) < 8:
                    example_rows.append({
                        "eg": int(res["eg"]),
                        "sg": int(res["sg"]),
                        "rep": int(res["rep"]),
                        "clean_states": res["clean_states"],
                        "degraded_states": res["degraded_states"],
                    })
            else:
                if len(dropped_examples) < 8:
                    dropped_examples.append({
                        "eg": int(res["eg"]),
                        "sg": int(res["sg"]),
                        "rep_target": int(res["rep_target"]),
                        "error_type": res["error_type"],
                        "error": res["error"],
                    })
            if (completed_tasks % progress_every) == 0 or completed_tasks == len(tasks):
                write_progress_json(
                    progress_json,
                    completed_tasks=completed_tasks,
                    total_tasks=len(tasks),
                    accepted_examples=len(clean_labels),
                    dropped_attempts=completed_tasks - len(clean_labels),
                    current_task_id=current_task_id,
                    eg_index_start=eg_start,
                    eg_index_stop=eg_stop,
                )
    finally:
        if num_processes > 1:
            pool.close()
            pool.join()
    clean_labels = np.asarray(clean_labels, dtype=np.int64)
    clean_obs = np.asarray(clean_obs, dtype=np.int64)
    degraded_labels = np.asarray(degraded_labels, dtype=np.int64)
    degraded_obs = np.asarray(degraded_obs, dtype=np.int64)
    output["clean_structure_factor_like"] = {
        "mask_state_counts": clean_state_counts,
        "n_examples": int(len(clean_labels)),
        "n_dropped_attempts": int(len(tasks) - len(clean_labels)),
        "labels_npz_key": "clean_labels",
        "symbols_npz_key": "clean_symbols",
        "dropped_preview": dropped_examples,
        "example_rows_preview": example_rows,
    }
    output["degraded_overlap_like"] = {
        "mask_state_counts": degraded_state_counts,
        "n_examples": int(len(degraded_labels)),
        "n_dropped_attempts": int(len(tasks) - len(degraded_labels)),
        "labels_npz_key": "degraded_labels",
        "symbols_npz_key": "degraded_symbols",
        "dropped_preview": dropped_examples,
        "example_rows_preview": example_rows,
    }

    if not args.skip_summary:
        clean_W, clean_warn = estimate_channel(
            clean_labels,
            clean_obs,
            n_classes=n_classes,
            n_symbols=n_symbols,
            dirichlet_alpha=args.dirichlet_alpha,
        )
        output["clean_structure_factor_like"]["warning"] = clean_warn
        output["clean_structure_factor_like"]["summary"] = full_summary(clean_W, prior, run_capacity=True)
        write_json(args.out_json, output)

        degraded_W, degraded_warn = estimate_channel(
            degraded_labels,
            degraded_obs,
            n_classes=n_classes,
            n_symbols=n_symbols,
            dirichlet_alpha=args.dirichlet_alpha,
        )
        output["degraded_overlap_like"]["warning"] = degraded_warn
        output["degraded_overlap_like"]["summary"] = full_summary(degraded_W, prior, run_capacity=True)
    else:
        output["clean_structure_factor_like"]["warning"] = "summary skipped for shard run"
        output["degraded_overlap_like"]["warning"] = "summary skipped for shard run"

    np.savez_compressed(
        out_npz,
        clean_labels=clean_labels,
        clean_symbols=clean_obs,
        degraded_labels=degraded_labels,
        degraded_symbols=degraded_obs,
    )
    output["artifacts"] = {
        "out_json": str(args.out_json),
        "out_npz": str(out_npz),
        "progress_json": str(progress_json),
    }

    write_json(args.out_json, output)
    print(json.dumps(to_jsonable(output), indent=2))


if __name__ == "__main__":
    main()
