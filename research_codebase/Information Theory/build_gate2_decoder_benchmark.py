#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import random
import subprocess
import sys
import tempfile
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
import spglib
from ase import Atoms
from pyxtal import pyxtal
from pyxtal.symmetry import get_wyckoffs
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.io.ase import AseAtomsAdaptor


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

RUNG_NAMES = [
    "ideal_mask",
    "structure_factor_like",
    "broadening_overlap_like",
    "rruff_like",
    "background_noise_like",
]

MASK_STATES = ("present", "absent", "unresolved")
_STATE_TO_INT = {s: i for i, s in enumerate(MASK_STATES)}

DEFAULT_RRUFF473_COUNTS = {
    1: 16, 2: 12, 3: 61, 4: 37, 5: 43, 9: 5, 10: 3, 11: 12, 13: 4, 15: 2, 16: 51,
    17: 5, 20: 6, 21: 13, 23: 2, 28: 8, 29: 5, 34: 2, 37: 4, 38: 1, 39: 2, 44: 4,
    45: 2, 51: 2, 52: 2, 53: 1, 59: 1, 61: 4, 62: 2, 67: 5, 69: 4, 70: 2, 71: 9,
    72: 21, 74: 19, 75: 13, 76: 12, 79: 7, 81: 2, 85: 2, 88: 1, 92: 3, 93: 2,
    94: 2, 96: 6, 98: 51,
}

_LOCAL_CFML_HELPER = Path(__file__).with_name("cfml_api_single_pattern.py")
_REPO_CFML_HELPER = Path(__file__).resolve().parents[1] / "Code" / "Generation" / "tacc" / "cfml_api_single_pattern.py"
CFML_HELPER = _LOCAL_CFML_HELPER if _LOCAL_CFML_HELPER.exists() else _REPO_CFML_HELPER
WORKER_CFG: dict = {}


class SimpleGeneratedStructure:
    def __init__(self, atoms: Atoms, sg_number: int):
        self._atoms = atoms
        self.group = type("Group", (), {"number": int(sg_number)})()

    def to_ase(self):
        return self._atoms


def parse_args():
    p = argparse.ArgumentParser(description="Build a decoder-facing benchmark on a Gate-3-style degraded channel.")
    p.add_argument("--mapping-csv", required=True)
    p.add_argument("--output-h5", required=True)
    p.add_argument("--output-npz", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--rung", default="background_noise_like", choices=RUNG_NAMES)
    p.add_argument("--counts-json", default="")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--max-attempts", type=int, default=150)
    p.add_argument("--wavelength", type=float, default=1.54030)
    p.add_argument("--two-theta-min", type=float, default=5.0)
    p.add_argument("--two-theta-max", type=float, default=90.0)
    p.add_argument("--two-theta-step", type=float, default=0.01)
    p.add_argument("--present-threshold-frac", type=float, default=0.03)
    p.add_argument("--overlap-window-deg-clean", type=float, default=0.12)
    p.add_argument("--overlap-window-deg-degraded", type=float, default=0.32)
    p.add_argument("--background-frac", type=float, default=0.03)
    p.add_argument("--noise-frac", type=float, default=0.01)
    p.add_argument("--orientation-sigma", type=float, default=0.35)
    p.add_argument("--num-processes", type=int, default=0)
    p.add_argument("--case-prefix", default="gate2_decoder")
    p.add_argument("--cfml-helper", default=str(CFML_HELPER))
    return p.parse_args()


def encode_mask_vector(mask_vec, n_states: int = 3) -> int:
    ints = [_STATE_TO_INT[m] if isinstance(m, str) else int(m) for m in mask_vec]
    sym = 0
    for v in ints:
        sym = sym * n_states + v
    return sym


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


def load_target_counts(path: str) -> dict[int, int]:
    if not path:
        return dict(DEFAULT_RRUFF473_COUNTS)
    payload = json.loads(Path(path).read_text())
    return {int(k): int(v) for k, v in payload.items()}


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
    if int(space_group) == 1:
        atoms = Atoms(
            symbols=["Si", "O"],
            scaled_positions=[[0.137, 0.271, 0.389], [0.641, 0.517, 0.913]],
            cell=[[5.3, 0.0, 0.0], [0.2, 5.9, 0.0], [0.1, 0.3, 6.4]],
            pbc=True,
        )
        return SimpleGeneratedStructure(atoms, 1)
    raise RuntimeError(f"failed to generate SG {space_group}: {last_error}")


def create_pattern_json(structure, wavelength: float, tt_min: float, tt_max: float, tt_step: float, u_resolution: float):
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
                    "_pd_instr_resolution_u": float(u_resolution),
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


def cw_powder_pattern_backend(pattern_dict: dict, structure, helper_path: Path):
    atoms = structure.to_ase()
    npd = pattern_dict["experiments"][0]["NPD"]
    with tempfile.NamedTemporaryFile(suffix=".cif", delete=False) as handle:
        cif_path = handle.name
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as handle:
        out_path = handle.name
    try:
        from ase.io import write as ase_write

        ase_write(cif_path, atoms, format="cif")
        proc = subprocess.run(
            [
                sys.executable,
                str(helper_path),
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
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"CFML helper failed rc={proc.returncode} helper={helper_path} "
                f"stdout={proc.stdout[:500]!r} stderr={proc.stderr[:500]!r}"
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
            first = entry[0]
            if isinstance(first, (list, tuple, np.ndarray, dict)):
                return normalize_hkl(first)
    return None


def family_key(hkl: tuple[int, int, int] | None) -> str | None:
    if hkl is None:
        return None
    target = tuple(abs(int(v)) for v in hkl)
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


def local_window_stats(x: np.ndarray, y: np.ndarray, center: float, half_window: float) -> tuple[float, float]:
    mask = (x >= center - half_window) & (x <= center + half_window)
    if not np.any(mask):
        return 0.0, 0.0
    vals = np.asarray(y[mask], dtype=np.float64)
    return float(np.max(vals)), float(np.median(vals))


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
        unresolved = any(
            other != fam and other_center is not None and abs(other_center - center) <= overlap_window_deg
            for other, other_center in family_centers.items()
        )
        if unresolved:
            states.append("unresolved")
            continue
        peak, _baseline = local_window_stats(x, y, center, overlap_window_deg)
        states.append("present" if peak >= present_threshold else "unresolved")
    return states


def family_mask_states_soft(
    reference_peaks: dict[str, list[float]],
    x: np.ndarray,
    y: np.ndarray,
    *,
    overlap_window_deg: float,
    present_threshold_frac: float,
) -> list[str]:
    states: list[str] = []
    ymax = float(np.max(y)) if y.size else 0.0
    global_threshold = present_threshold_frac * ymax
    family_centers = {fam: (min(centers) if centers else None) for fam, centers in reference_peaks.items()}

    diffs = np.diff(np.asarray(y, dtype=np.float64)) if y.size else np.zeros(0, dtype=np.float64)
    global_noise = float(np.std(diffs) / np.sqrt(2.0)) if diffs.size else 0.0

    for fam, _proto in FAMILY_SPECS:
        center = family_centers[fam]
        if center is None:
            states.append("absent")
            continue
        overlap = any(
            other != fam and other_center is not None and abs(other_center - center) <= overlap_window_deg
            for other, other_center in family_centers.items()
        )
        peak, baseline = local_window_stats(x, y, center, overlap_window_deg)
        prominence = max(peak - baseline, 0.0)
        local_noise = max(global_noise, 0.05 * max(peak, baseline, 1e-8))
        snr = prominence / max(local_noise, 1e-8)
        decisive = (prominence >= global_threshold) and (snr >= 3.0)
        states.append("present" if decisive and not overlap else "unresolved")
    return states


def apply_gaussian_broadening(x: np.ndarray, y: np.ndarray, sigma_deg: float) -> np.ndarray:
    if sigma_deg <= 0 or y.size == 0 or x.size < 2:
        return y.copy()
    step = float(np.median(np.diff(x)))
    if not np.isfinite(step) or step <= 0:
        return y.copy()
    sigma_pts = float(sigma_deg / step)
    if sigma_pts <= 0.25:
        return y.copy()
    radius = max(1, int(np.ceil(4.0 * sigma_pts)))
    grid = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (grid / sigma_pts) ** 2)
    kernel /= np.sum(kernel)
    return np.convolve(y, kernel, mode="same")


def apply_family_attenuation(
    x: np.ndarray,
    y: np.ndarray,
    reference_peaks: dict[str, list[float]],
    rng: np.random.Generator,
    overlap_window_deg: float,
    orientation_sigma: float,
) -> np.ndarray:
    out = y.copy()
    for fam, centers in reference_peaks.items():
        if not centers:
            continue
        center = min(centers)
        window = (x >= center - overlap_window_deg) & (x <= center + overlap_window_deg)
        if not np.any(window):
            continue
        weight = float(np.clip(np.exp(-abs(rng.normal(0.0, orientation_sigma))), 0.2, 1.0))
        out[window] *= weight
    return np.clip(out, 0.0, None)


def apply_background_noise(y: np.ndarray, rng: np.random.Generator, background_frac: float, noise_frac: float) -> np.ndarray:
    ymax = float(np.max(y)) if y.size else 0.0
    if ymax <= 0:
        return y.copy()
    grid = np.linspace(0.0, 1.0, len(y), dtype=np.float64)
    baseline = ymax * background_frac * (0.25 + 0.55 * grid + 0.20 * grid * grid)
    noise = rng.normal(0.0, noise_frac * ymax, size=len(y))
    return np.clip(y + baseline + noise, 0.0, None)


def normalize_pattern(y: np.ndarray) -> np.ndarray:
    ymax = float(np.max(y)) if y.size else 0.0
    if ymax <= 0:
        return np.zeros_like(y, dtype=np.float32)
    return (y / ymax).astype(np.float32, copy=False)


def build_tasks(eg_to_sgs: dict[int, list[int]], counts: dict[int, int], seed: int):
    task_id = 0
    for eg in sorted(counts):
        sgs = eg_to_sgs[eg]
        for rep in range(int(counts[eg])):
            seq = np.random.SeedSequence([seed, int(eg), int(rep)])
            rng = np.random.default_rng(seq)
            sg = int(rng.choice(np.asarray(sgs, dtype=np.int64)))
            wseed = int(rng.integers(0, 2**63 - 1))
            yield (task_id, int(eg), sg, int(rep), wseed)
            task_id += 1


def init_worker(cfg):
    global WORKER_CFG
    WORKER_CFG = dict(cfg)
    random.seed(int(cfg["seed_base"]))
    np.random.seed(int(cfg["seed_base"]))


def _family_geom_two_theta(cell_mat, wavelength):
    """Geometric 2theta of each family prototype reflection from the cell metric
    (present OR absent -- the position you'd look at to decide present/absent)."""
    recip = 2.0 * np.pi * np.linalg.inv(np.asarray(cell_mat, dtype=np.float64)).T
    out = {}
    for name, hkl in FAMILY_SPECS:
        q = float(np.linalg.norm(np.asarray(hkl, dtype=np.float64) @ recip))
        if q <= 0:
            out[name] = None
            continue
        s = wavelength / (2.0 * (2.0 * np.pi / q))
        out[name] = float(np.degrees(2.0 * np.arcsin(s))) if abs(s) <= 1.0 else None
    return out


def _oracle_window(x, y, center, half_pts):
    n = 2 * half_pts + 1
    if center is None:
        return np.zeros(n, dtype=np.float32)
    step = float(x[1] - x[0])
    idx = int(round((center - float(x[0])) / step))
    out = np.zeros(n, dtype=np.float32)
    lo, hi = idx - half_pts, idx + half_pts + 1
    a, b = max(0, lo), min(len(y), hi)
    if a < b:
        out[a - lo:b - lo] = np.asarray(y[a:b], dtype=np.float32)
    return out


def worker(task):
    task_id, eg, sg, rep, seed = task
    rng = np.random.default_rng(seed)
    try:
        structure = get_valid_structure(sg, max_attempts=int(WORKER_CFG["max_attempts"]))
        ref = reference_family_peaks(
            structure,
            wavelength=float(WORKER_CFG["wavelength"]),
            tt_min=float(WORKER_CFG["tt_min"]),
            tt_max=float(WORKER_CFG["tt_max"]),
        )
        pattern_json = create_pattern_json(
            structure,
            wavelength=float(WORKER_CFG["wavelength"]),
            tt_min=float(WORKER_CFG["tt_min"]),
            tt_max=float(WORKER_CFG["tt_max"]),
            tt_step=float(WORKER_CFG["tt_step"]),
            u_resolution=float(WORKER_CFG["u_resolution_clean"]),
        )
        x_clean, y_clean = cw_powder_pattern_backend(pattern_json, structure, Path(WORKER_CFG["cfml_helper"]))
        y_clean = np.nan_to_num(np.asarray(y_clean, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)

        rung_states = {}
        rung_states["ideal_mask"] = [("present" if ref.get(fam) else "absent") for fam, _ in FAMILY_SPECS]
        rung_states["structure_factor_like"] = family_mask_states(
            ref, x_clean, y_clean,
            overlap_window_deg=float(WORKER_CFG["overlap_window_deg_clean"]),
            present_threshold_frac=float(WORKER_CFG["present_threshold_frac"]),
        )
        broadening_sigma = max(0.0, float(WORKER_CFG["overlap_window_deg_degraded"]) - float(WORKER_CFG["overlap_window_deg_clean"]))
        y_broad = apply_gaussian_broadening(x_clean, y_clean, sigma_deg=broadening_sigma)
        rung_states["broadening_overlap_like"] = family_mask_states(
            ref, x_clean, y_broad,
            overlap_window_deg=float(WORKER_CFG["overlap_window_deg_degraded"]),
            present_threshold_frac=float(WORKER_CFG["present_threshold_frac"]),
        )
        y_rruff = apply_family_attenuation(
            x_clean, y_broad, reference_peaks=ref, rng=rng,
            overlap_window_deg=float(WORKER_CFG["overlap_window_deg_degraded"]),
            orientation_sigma=float(WORKER_CFG["orientation_sigma"]),
        )
        rung_states["rruff_like"] = family_mask_states_soft(
            ref, x_clean, y_rruff,
            overlap_window_deg=float(WORKER_CFG["overlap_window_deg_degraded"]),
            present_threshold_frac=float(WORKER_CFG["present_threshold_frac"]),
        )
        y_bg = apply_background_noise(
            y_rruff, rng=rng,
            background_frac=float(WORKER_CFG["background_frac"]),
            noise_frac=float(WORKER_CFG["noise_frac"]),
        )
        rung_states["background_noise_like"] = family_mask_states_soft(
            ref, x_clean, y_bg,
            overlap_window_deg=float(WORKER_CFG["overlap_window_deg_degraded"]),
            present_threshold_frac=float(WORKER_CFG["present_threshold_frac"]),
        )

        rung_patterns = {
            "structure_factor_like": y_clean,
            "broadening_overlap_like": y_broad,
            "rruff_like": y_rruff,
            "background_noise_like": y_bg,
        }
        rung = str(WORKER_CFG["rung"])
        if rung == "ideal_mask":
            selected_y = y_clean
        else:
            selected_y = rung_patterns[rung]

        y_norm = normalize_pattern(np.asarray(selected_y, dtype=np.float64))
        geom = _family_geom_two_theta(np.asarray(structure.to_ase().get_cell(), dtype=np.float64),
                                      float(WORKER_CFG["wavelength"]))
        oracle_feats = np.stack([_oracle_window(x_clean, y_norm, geom[name], 30)
                                 for name, _ in FAMILY_SPECS], axis=0)  # (6, 61)
        return {
            "ok": True,
            "task_id": int(task_id),
            "eg": int(eg),
            "sg": int(sg),
            "rep": int(rep),
            "case_id": f"{WORKER_CFG['case_prefix']}_eg{eg}_sg{sg}_rep{rep}",
            "x": np.asarray(x_clean, dtype=np.float32),
            "y": np.asarray(y_norm, dtype=np.float32),
            "oracle_features": oracle_feats,
            "rung_states": rung_states,
            "rung_symbols": {name: int(encode_mask_vector(states)) for name, states in rung_states.items()},
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "task_id": int(task_id),
            "eg": int(eg),
            "sg": int(sg),
            "rep": int(rep),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def main():
    args = parse_args()
    helper_path = Path(args.cfml_helper).resolve()
    if not helper_path.exists():
        raise FileNotFoundError(f"CFML helper not found: {helper_path}")
    counts = load_target_counts(args.counts_json)
    eg_rows = load_eg_to_sgs(args.mapping_csv)
    eg_to_sgs = {int(eg): list(map(int, sgs)) for eg, sgs in eg_rows}
    missing = sorted(set(counts) - set(eg_to_sgs))
    if missing:
        raise ValueError(f"missing EG mappings for labels: {missing}")
    tasks = list(build_tasks(eg_to_sgs, counts, seed=args.seed))
    worker_cfg = {
        "seed_base": int(args.seed),
        "max_attempts": int(args.max_attempts),
        "wavelength": float(args.wavelength),
        "tt_min": float(args.two_theta_min),
        "tt_max": float(args.two_theta_max),
        "tt_step": float(args.two_theta_step),
        "u_resolution_clean": 0.008,
        "present_threshold_frac": float(args.present_threshold_frac),
        "overlap_window_deg_clean": float(args.overlap_window_deg_clean),
        "overlap_window_deg_degraded": float(args.overlap_window_deg_degraded),
        "background_frac": float(args.background_frac),
        "noise_frac": float(args.noise_frac),
        "orientation_sigma": float(args.orientation_sigma),
        "cfml_helper": str(helper_path),
        "rung": str(args.rung),
        "case_prefix": str(args.case_prefix),
    }

    num_processes = get_num_processes(args.num_processes)
    if num_processes <= 1:
        init_worker(worker_cfg)
        iterator = (worker(task) for task in tasks)
    else:
        pool = Pool(processes=num_processes, initializer=init_worker, initargs=(worker_cfg,))
        iterator = pool.imap_unordered(worker, tasks, chunksize=1)

    accepted = []
    dropped = []
    try:
        for res in iterator:
            if res["ok"]:
                accepted.append(res)
            else:
                dropped.append(res)
    finally:
        if num_processes > 1:
            pool.close()
            pool.join()

    if dropped:
        frac = len(dropped) / max(1, len(dropped) + len(accepted))
        print(f"WARNING: {len(dropped)} generation failures (frac {frac:.3f}); proceeding with "
              f"{len(accepted)} accepted. first={dropped[0]}", file=sys.stderr)
        if frac > 0.1:
            raise RuntimeError(f"too many generation failures ({frac:.2f}); first={dropped[0]}")
    accepted.sort(key=lambda row: row["task_id"])

    x0 = np.asarray(accepted[0]["x"], dtype=np.float32)
    y_stack = np.stack([np.asarray(row["y"], dtype=np.float32) for row in accepted], axis=0)
    oracle_stack = np.stack([np.asarray(row["oracle_features"], dtype=np.float32) for row in accepted], axis=0)
    labels = np.asarray([int(row["eg"]) for row in accepted], dtype=np.int16)
    case_ids = np.asarray([row["case_id"] for row in accepted], dtype=h5py.string_dtype("utf-8"))
    rung_symbols = {name: np.asarray([int(row["rung_symbols"][name]) for row in accepted], dtype=np.int64) for name in RUNG_NAMES}

    out_h5 = Path(args.output_h5)
    out_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_h5, "w") as h5:
        h5.create_dataset("X_test", data=y_stack, compression="gzip")
        h5.create_dataset("oracle_features", data=oracle_stack, compression="gzip")
        h5.create_dataset("y_test", data=labels, compression="gzip")
        h5.create_dataset("two_theta", data=x0, compression="gzip")
        h5.create_dataset("case_ids", data=case_ids, compression="gzip")
        h5.attrs["source"] = "gate3_decoder_benchmark"
        h5.attrs["rung"] = args.rung
        h5.attrs["n_rows"] = int(len(labels))

    np.savez_compressed(
        args.output_npz,
        labels=labels.astype(np.int64),
        x=x0.astype(np.float32),
        **{f"symbols_{name}": arr for name, arr in rung_symbols.items()},
    )

    summary = {
        "mapping_csv": args.mapping_csv,
        "output_h5": str(out_h5),
        "output_npz": str(args.output_npz),
        "rung": args.rung,
        "n_rows": int(len(labels)),
        "support": int(len(np.unique(labels))),
        "counts": {int(k): int(v) for k, v in counts.items()},
        "two_theta_min": float(args.two_theta_min),
        "two_theta_max": float(args.two_theta_max),
        "two_theta_step": float(args.two_theta_step),
    }
    Path(args.output_json).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
