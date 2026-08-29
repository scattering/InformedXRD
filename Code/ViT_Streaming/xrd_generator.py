"""
xrd_generator.py
----------------
Self-contained XRD pattern generation library for streaming_dataset.py.
Ported from gen_rruff_conditioned.py (Multilabel branch) with empirical
RRUFF/ICSD priors, Wyckoff-aware composition sampling, and Chebyshev
background replacing GP/RFF as the fast path.

Backend: pycrysfml.cfml_py_utilities (fast, direct call)
Fallback: CFML_api via subprocess (tacc/cfml_api_single_pattern.py)

Key exports for streaming_dataset.py:
  - NUM_POINTS          : int
  - TWO_THETA           : np.ndarray (float32, shape [NUM_POINTS])
  - EGS / SAMPLE_EGS    : np.ndarray (int16)
  - EG_WEIGHTS / SAMPLE_EG_WEIGHTS : np.ndarray (float64)
  - ALL_EGS             : np.ndarray (int16, all 99 groups)
  - ext_to_sg           : dict[int, list[int]]
  - ALL_SPACE_GROUPS    : np.ndarray (int16)
  - worker(task)        : (task_id, eg, sg_main, sg_imp, seed) -> (task_id, sg, eg, intensity) | (task_id, None)
  - build_task(task_id, seed) : -> task tuple
  - sample_main_sg(rng) : RRUFF-weighted (eg, sg)
  - sample_main_sg_balanced(task_id, seed) : uniform block schedule (eg, sg)
  - sample_impurity_sg_uniform(rng) : -> sg

Environment variables (all optional):
  EG_SAMPLING_MODE        : "rruff" (default) or "uniform"
  PRIOR_MODE              : "robust" (default) or "clean" — RRUFF row filter
  FULLPROF                : path to FullProf installation
  EXT_MAP_CSV             : path to FINAL_SPG_ExtG_CrysS_Table.csv
  RRUFF_PRIOR_CSV         : path to option1_metadata_qc_full.csv
  ICSD_CHEMISTRY_CSV      : path to ICSD_With_Bravais.csv
  USE_EXACT_GP            : "0" (default RFF) or "1" (exact GP, slower)
  RFF_FEATURES            : int, default 1024
  SAMPLE_RETRY_LIMIT      : int, default 3
  FAILURE_LOG_LIMIT       : int, default 200 (0 = disabled)
  FAILURE_LOG_PATH        : path for JSONL failure log
  CHEMISTRY_MARGINAL_TEMPER : float, default 0.5
  CHEMISTRY_PAIR_TEMPER   : float, default 0.5
  CHEMISTRY_SMOOTH        : float, default 1.0
"""

from __future__ import annotations

import csv
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
from pathlib import Path

import numpy as np
import pandas as pd
import spglib
from ase.io import write as ase_write
from pyxtal import pyxtal
from pyxtal.symmetry import Group, get_wyckoffs
from scipy.linalg import cholesky

try:
    from pymatgen.symmetry.groups import SpaceGroup as PmgSpaceGroup
except ImportError:
    PmgSpaceGroup = None

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------
# Environment / config
# -----------------------------------------------------------------------

_WORKROOT = os.environ.get("WORKROOT", "/work/ALLOC3/UserE/shared/pycrysfml_spr")
os.environ["FULLPROF"] = os.environ.get(
    "FULLPROF", os.path.join(_WORKROOT, "fullprof")
)

EXT_MAP_CSV = os.environ.get(
    "EXT_MAP_CSV",
    os.path.join(_WORKROOT, "FINAL_SPG_ExtG_CrysS_Table.csv"),
)
RRUFF_PRIOR_CSV = os.environ.get(
    "RRUFF_PRIOR_CSV",
    "/work2/ALLOC3/UserE/rruff-benchmark/option1_metadata_qc_full.csv",
)
ICSD_CHEMISTRY_CSV = os.environ.get(
    "ICSD_CHEMISTRY_CSV",
    "/work2/ALLOC3/UserE/reference_data/ICSD_With_Bravais.csv",
)
FAILURE_LOG_PATH = os.environ.get("FAILURE_LOG_PATH", "xrd_generator_failures.jsonl")

EG_SAMPLING_MODE   = os.environ.get("EG_SAMPLING_MODE", "rruff").strip().lower()
PRIOR_MODE         = os.environ.get("PRIOR_MODE", "robust").strip().lower()
USE_EXACT_GP       = bool(int(os.environ.get("USE_EXACT_GP", "0")))
RFF_FEATURES       = int(os.environ.get("RFF_FEATURES", "1024"))
SAMPLE_RETRY_LIMIT = int(os.environ.get("SAMPLE_RETRY_LIMIT", "3"))
FAILURE_LOG_LIMIT  = int(os.environ.get("FAILURE_LOG_LIMIT", "200"))

CHEMISTRY_MARGINAL_TEMPER = float(os.environ.get("CHEMISTRY_MARGINAL_TEMPER", "0.5"))
CHEMISTRY_PAIR_TEMPER     = float(os.environ.get("CHEMISTRY_PAIR_TEMPER", "0.5"))
CHEMISTRY_SMOOTH          = float(os.environ.get("CHEMISTRY_SMOOTH", "1.0"))

if EG_SAMPLING_MODE not in {"rruff", "uniform"}:
    raise RuntimeError(
        f"Unsupported EG_SAMPLING_MODE='{EG_SAMPLING_MODE}'. Use 'rruff' or 'uniform'."
    )
if PRIOR_MODE not in {"clean", "robust"}:
    raise RuntimeError(
        f"Unsupported PRIOR_MODE='{PRIOR_MODE}'. Use 'clean' or 'robust'."
    )

MAX_INTENSITY = 1000.0

# Cu Kα doublet
KALPHA1   = 1.54056
KALPHA2   = 1.54439
W_KALPHA1 = 2.0 / 3.0
W_KALPHA2 = 1.0 / 3.0

INSTRUMENT_PARAMS = {
    "twotheta_min": 5.0,
    "twotheta_max": 90.0,
    "step_size":    0.01,
}

NUM_POINTS = int(
    (INSTRUMENT_PARAMS["twotheta_max"] - INSTRUMENT_PARAMS["twotheta_min"])
    / INSTRUMENT_PARAMS["step_size"] + 1
)
TWO_THETA = np.linspace(
    INSTRUMENT_PARAMS["twotheta_min"],
    INSTRUMENT_PARAMS["twotheta_max"],
    NUM_POINTS,
    dtype=np.float32,
)
# Chebyshev domain for empirical background evaluation
X_SCALED = np.linspace(-1.0, 1.0, NUM_POINTS, dtype=np.float64)

# Path to the CFML_api subprocess helper — must sit next to xrd_generator.py
_HERE        = os.path.dirname(os.path.abspath(__file__))
CFML_HELPER  = os.path.join(_HERE, "tacc", "cfml_api_single_pattern.py")

# -----------------------------------------------------------------------
# Backend availability
# -----------------------------------------------------------------------

try:
    from pycrysfml import cfml_py_utilities as _cfml_py_utilities
except ImportError:
    _cfml_py_utilities = None

try:
    import CFML_api as _CFML_api
except ImportError:
    _CFML_api = None

if _cfml_py_utilities is None and _CFML_api is None:
    raise RuntimeError(
        "Neither pycrysfml.cfml_py_utilities nor CFML_api is available. "
        "Install at least one backend before using xrd_generator."
    )

_backend = "cfml_py_utilities" if _cfml_py_utilities is not None else "CFML_api"
print(f"[xrd_generator] backend: {_backend}")

# -----------------------------------------------------------------------
# SG <-> EG mapping
# -----------------------------------------------------------------------

def _build_maps(csv_path: str):
    df        = pd.read_csv(csv_path)
    sg_to_ext = {int(sg): int(eg)
                 for sg, eg in zip(df["Space Group"], df["Extinction Group"])}
    ext_to_sg: dict[int, list[int]] = {}
    for sg, eg in sg_to_ext.items():
        ext_to_sg.setdefault(int(eg), []).append(int(sg))
    all_sgs = np.array(sorted(sg_to_ext.keys()), dtype=np.int16)
    if all_sgs.size == 0:
        raise RuntimeError(
            "EXT_MAP_CSV produced empty SG list — check file path and columns."
        )
    return sg_to_ext, ext_to_sg, all_sgs


sg_to_ext, ext_to_sg, ALL_SPACE_GROUPS = _build_maps(EXT_MAP_CSV)

ALL_EGS = np.array(sorted(ext_to_sg.keys()), dtype=np.int16)
if ALL_EGS.size != 99:
    raise RuntimeError(
        f"Expected 99 extinction groups from EXT_MAP_CSV, found {ALL_EGS.size}"
    )

# -----------------------------------------------------------------------
# RRUFF EG distribution
# -----------------------------------------------------------------------

RRUFF_EG_COUNTS: dict[int, int] = {
    4: 153, 17: 121,  5: 112, 99: 107,  6:  98, 75:  75,  2:  55, 77:  41, 73:  40, 71:  34,
   29:  32, 22:  23, 10:  22, 76:  21, 21:  21, 93:  20, 63:  20,  3:  19, 97:  17, 18:  14,
   12:  13, 80:  12, 54:  12, 16:   9, 85:   9, 68:   8, 33:   8, 14:   7, 52:   7, 74:   6,
   89:   6, 45:   6, 62:   5, 30:   5, 94:   5, 55:   4, 86:   4, 72:   4, 24:   4, 34:   4,
    1:   4, 31:   4, 69:   4, 60:   4, 11:   4, 65:   3, 35:   3, 32:   3, 70:   3, 40:   3,
   38:   3, 26:   3, 81:   2, 39:   2, 53:   2, 19:   2, 95:   2, 51:   2, 49:   1, 42:   1,
   91:   1, 58:   1, 41:   1, 82:   1, 47:   1, 44:   1, 87:   1,  9:   1, 15:   1, 23:   1,
}

_valid_egs = [
    eg for eg in RRUFF_EG_COUNTS
    if eg in ext_to_sg and len(ext_to_sg[eg]) > 0
]
if not _valid_egs:
    raise RuntimeError(
        "No overlap between RRUFF_EG_COUNTS and EXT_MAP_CSV extinction groups."
    )

EGS        = np.array(_valid_egs, dtype=np.int16)
EG_WEIGHTS = np.array([RRUFF_EG_COUNTS[int(eg)] for eg in EGS], dtype=np.float64)
EG_WEIGHTS /= EG_WEIGHTS.sum()

_missing = sorted(set(RRUFF_EG_COUNTS.keys()) - set(_valid_egs))
if _missing:
    print(f"[xrd_generator] WARN: {len(_missing)} EGs not in EXT_MAP_CSV: {_missing}")

if EG_SAMPLING_MODE == "uniform":
    SAMPLE_EGS        = ALL_EGS
    SAMPLE_EG_WEIGHTS = np.full(ALL_EGS.shape, 1.0 / ALL_EGS.size, dtype=np.float64)
    print(f"[xrd_generator] UNIFORM sampling — {len(SAMPLE_EGS)} EGs")
else:
    SAMPLE_EGS        = EGS
    SAMPLE_EG_WEIGHTS = EG_WEIGHTS
    print(f"[xrd_generator] RRUFF-weighted sampling — {len(SAMPLE_EGS)} EGs")

print(f"[xrd_generator] NUM_POINTS={NUM_POINTS}  EG_SAMPLING_MODE={EG_SAMPLING_MODE}  PRIOR_MODE={PRIOR_MODE}")

# -----------------------------------------------------------------------
# Crystal system helpers
# -----------------------------------------------------------------------

def get_crystal_system_code(space_group: int) -> int:
    if   1 <= space_group <=   2: return 0   # triclinic
    if   3 <= space_group <=  15: return 1   # monoclinic
    if  16 <= space_group <=  74: return 2   # orthorhombic
    if  75 <= space_group <= 142: return 3   # tetragonal
    if 143 <= space_group <= 167: return 4   # trigonal
    if 168 <= space_group <= 194: return 5   # hexagonal
    if 195 <= space_group <= 230: return 6   # cubic
    return -1


def get_family_key(space_group: int) -> str:
    code = get_crystal_system_code(space_group)
    if code == 6:                return "cubic"
    if code in {2, 3}:           return "ortho_tetra"
    if code == 1:                return "monoclinic"
    if code in {4, 5}:           return "trig_hex"
    return "triclinic_other"

# -----------------------------------------------------------------------
# Element set
# -----------------------------------------------------------------------

_VALID_ELEMENT_SYMBOLS = {
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
_EXCLUDED_UNSTABLE = {
    "Tc", "Pm", "Po", "At", "Rn", "Fr", "Ra", "Ac",
    "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs",
    "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
}
_PYXTAL_UNSUPPORTED = {"Bk", "Cf"}
ALLOWED_CHEMISTRY_SYMBOLS = _VALID_ELEMENT_SYMBOLS - _EXCLUDED_UNSTABLE - _PYXTAL_UNSUPPORTED

ELEMENTS = [
    "H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Cl","Ar","K","Ca",
    "Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr","Rb","Sr",
    "Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I","Xe","Cs","Ba",
    "La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","Hf","Ta","W",
    "Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn","Fr","Ra","Ac","Th","Pa","U",
]
ELEMENTS = [el for el in ELEMENTS if el in ALLOWED_CHEMISTRY_SYMBOLS]

_FORMULA_TOKEN_RE = re.compile(r"([A-Z][a-z]?)([0-9]*\.?[0-9]*)")

# -----------------------------------------------------------------------
# Empirical prior loading
# -----------------------------------------------------------------------

@lru_cache(maxsize=None)
def _resolve_sg_number(raw_sg: str) -> int:
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
    lower = compact.lower()
    if any(t in lower for t in ["-3m", "m-3", "23", "432"]):
        return 221
    if compact.startswith(("P6", "P3", "R3", "R-3", "P-3", "P63", "R32")):
        return 168
    if compact.startswith(("P4", "I4", "P-4", "I-4")):
        return 75
    if compact.startswith(("P2", "C2", "A2", "B2", "I2")):
        return 3
    if compact.startswith(("P1", "P-1")):
        return 1
    return -1


def _classify_rwp(rwp: float) -> str:
    if rwp > 0.5:  return "catastrophic"
    if rwp > 0.22: return "poor"
    if rwp > 0.12: return "recoverable"
    return "usable_or_better"


def _as_float(row: dict, key: str, default: float = 0.0) -> float:
    raw = row.get(key, "")
    if raw in ("", "nan", "NaN"):
        return default
    try:
        return float(raw)
    except Exception:
        return default


def _load_rruff_priors(path: str, prior_mode: str) -> dict:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"RRUFF prior CSV not found: {csv_path}")

    allowed = {"usable_or_better"} if prior_mode == "clean" else {"usable_or_better", "recoverable"}
    by_family: dict[str, list[dict]] = {k: [] for k in
        ["cubic", "ortho_tetra", "monoclinic", "trig_hex", "triclinic_other"]}
    global_rows: list[dict] = []

    with csv_path.open() as fh:
        for row in csv.DictReader(fh):
            sg = _resolve_sg_number(row.get("space_group", ""))
            if sg <= 0:
                continue
            if _classify_rwp(_as_float(row, "rwp", 1.0)) not in allowed:
                continue
            prior_row = {
                "space_group": sg,
                "family":      get_family_key(sg),
                "U":    max(0.0,  _as_float(row, "U")),
                "V":          _as_float(row, "V"),
                "W":    max(0.0,  _as_float(row, "W")),
                "X":    max(0.0,  _as_float(row, "X")),
                "Y":    max(0.0,  _as_float(row, "Y")),
                "eta":  float(np.clip(_as_float(row, "eta"), 0.0, 1.0)),
                "zero": _as_float(row, "zero"),
                "bkg":  np.array([_as_float(row, f"bkg_c{i}") for i in range(4)], dtype=np.float64),
            }
            by_family[prior_row["family"]].append(prior_row)
            global_rows.append(prior_row)

    if not global_rows:
        raise RuntimeError(
            f"No usable prior rows loaded from {csv_path} for PRIOR_MODE={prior_mode}"
        )
    print("[xrd_generator] RRUFF priors loaded:")
    for fam, rows in by_family.items():
        print(f"  {fam}: {len(rows)}")
    return {"by_family": by_family, "global": global_rows}


def _reduce_integer_ratios(values: list[int]) -> list[int]:
    out = [max(1, int(v)) for v in values]
    g = 0
    for v in out:
        g = math.gcd(g, v)
    return [max(1, v // g) for v in out] if g > 1 else out


def _parse_formula(formula: str) -> tuple[list[str], list[int]] | None:
    if not formula or not isinstance(formula, str):
        return None
    tokens = _FORMULA_TOKEN_RE.findall(formula.strip())
    if not tokens:
        return None
    elements: list[str] = []
    counts: list[float] = []
    for symbol, raw in tokens:
        if symbol not in ALLOWED_CHEMISTRY_SYMBOLS:
            continue
        try:
            c = float(raw) if raw else 1.0
        except ValueError:
            c = 1.0
        if c <= 0:
            continue
        elements.append(symbol)
        counts.append(c)
    if not elements:
        return None
    min_c = min(counts)
    ratios = _reduce_integer_ratios([max(1, int(round(c / min_c))) for c in counts])
    if sum(ratios) > 16:
        scale = max(1, int(round(sum(ratios) / 12)))
        ratios = _reduce_integer_ratios([max(1, int(round(v / scale))) for v in ratios])
    return elements, ratios


def _load_chemistry_priors(path: str) -> dict:
    families = ["cubic", "ortho_tetra", "monoclinic", "trig_hex", "triclinic_other"]
    by_family = {
        k: {"element_counts": Counter(), "pair_counts": Counter(),
            "num_elements_counts": Counter(), "ratio_bank": []}
        for k in families
    }
    global_stats = {"element_counts": Counter(), "pair_counts": Counter(),
                    "num_elements_counts": Counter(), "ratio_bank": []}

    csv_path = Path(path)
    if not csv_path.exists():
        print(f"[xrd_generator] WARN: ICSD chemistry CSV not found: {csv_path}. "
              "Falling back to broad element sampling.")
        return {"by_family": by_family, "global": global_stats}

    with csv_path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            sg = (_resolve_sg_number(row.get("sym_group", ""))
                  or _resolve_sg_number(row.get("Space Group", "")))
            if sg <= 0:
                continue
            parsed = _parse_formula(row.get("name", ""))
            if parsed is None or len(parsed[0]) > 4:
                continue
            elements, ratios = parsed
            family = get_family_key(sg)
            stats = by_family[family]
            unique = tuple(sorted(set(elements)))
            stats["num_elements_counts"][len(unique)] += 1
            global_stats["num_elements_counts"][len(unique)] += 1
            stats["ratio_bank"].append(tuple(ratios))
            global_stats["ratio_bank"].append(tuple(ratios))
            for el in unique:
                stats["element_counts"][el] += 1
                global_stats["element_counts"][el] += 1
            for pair in combinations(unique, 2):
                stats["pair_counts"][tuple(sorted(pair))] += 1
                global_stats["pair_counts"][tuple(sorted(pair))] += 1

    print("[xrd_generator] ICSD chemistry priors loaded:")
    for fam, stats in by_family.items():
        print(f"  {fam}: {sum(stats['num_elements_counts'].values())}")
    return {"by_family": by_family, "global": global_stats}


PRIORS           = _load_rruff_priors(RRUFF_PRIOR_CSV, PRIOR_MODE)
CHEMISTRY_PRIORS = _load_chemistry_priors(ICSD_CHEMISTRY_CSV)

# -----------------------------------------------------------------------
# Wyckoff-aware composition sampling (lru_cache'd per SG)
# -----------------------------------------------------------------------

@lru_cache(maxsize=None)
def get_wyckoffs_cached(space_group: int):
    return tuple(get_wyckoffs(space_group, organized=True))


@lru_cache(maxsize=None)
def _get_group_base_bounds(space_group: int):
    group = Group(space_group)
    if hasattr(group, "_get_base_and_upper_bounds"):
        base, upper_bounds = group._get_base_and_upper_bounds()
        return tuple(int(x) for x in base), tuple(upper_bounds)
    wyckoff_data = get_wyckoffs_cached(space_group)
    counts: Counter = Counter()
    for positions in wyckoff_data:
        mult = int(sum(len(pos) for pos in positions))
        if mult > 0:
            counts[mult] += 1
    base = tuple(sorted(counts))
    return base, tuple(counts[m] for m in base)


@lru_cache(maxsize=None)
def _get_group_flexible_multiplicities(space_group: int):
    group = Group(space_group)
    flexible: set[int] = set()
    for wp in group:
        try:
            if wp.get_dof() > 0:
                flexible.add(int(wp.multiplicity))
        except Exception:
            continue
    return tuple(sorted(flexible))


@lru_cache(maxsize=None)
def _get_group_general_multiplicity(space_group: int) -> int | None:
    group = Group(space_group)
    mults = []
    for wp in group:
        try:
            if wp.get_dof() > 0:
                mults.append(int(wp.multiplicity))
        except Exception:
            continue
    return max(mults) if mults else None


def _sample_wyckoff_compatible_counts(
    space_group: int,
    n_species: int,
    max_extra_orbits: int = 2,
    max_attempts: int = 100,
    require_general_position: bool = False,
) -> list[int] | None:
    if n_species <= 0:
        return None
    group          = Group(space_group)
    base, upper_bounds = _get_group_base_bounds(space_group)
    flexible_mults = set(_get_group_flexible_multiplicities(space_group))
    general_mult   = _get_group_general_multiplicity(space_group)
    if not base:
        return None

    weights = np.array([1.0 / max(float(m), 1.0) for m in base], dtype=np.float64)
    weights /= weights.sum()

    for _ in range(max_attempts):
        remaining = [None if ub is None else int(ub) for ub in upper_bounds]
        counts = [0] * n_species

        def draw_mult(prefer_flexible: bool = False, require_general: bool = False) -> int | None:
            avail = [i for i, ub in enumerate(remaining) if ub is None or ub > 0]
            if not avail:
                return None
            if require_general and general_mult is not None:
                g_idx = [i for i in avail if int(base[i]) == int(general_mult)]
                if g_idx:
                    avail = g_idx
            if prefer_flexible:
                f_idx = [i for i in avail if int(base[i]) in flexible_mults]
                if f_idx:
                    avail = f_idx
            lw = np.array([weights[i] for i in avail], dtype=np.float64)
            lw /= lw.sum()
            choice = int(np.random.choice(avail, p=lw))
            if remaining[choice] is not None:
                remaining[choice] -= 1
            return int(base[choice])

        ok = True
        for i in range(n_species):
            m = draw_mult(
                prefer_flexible=(i == 0 and bool(flexible_mults)),
                require_general=(i == 0 and require_general_position and general_mult is not None),
            )
            if m is None:
                ok = False
                break
            counts[i] += m
        if not ok:
            continue

        for _ in range(int(np.random.randint(0, max_extra_orbits + 1))):
            m = draw_mult()
            if m is None:
                break
            counts[int(np.random.randint(0, n_species))] += m

        if group.check_compatible(counts)[0]:
            return counts

    return None


def _normalized_element_probs(stats: dict) -> dict[str, float]:
    counts = stats["element_counts"]
    if not counts:
        return {}
    total = float(sum(counts.values()))
    denom = total + CHEMISTRY_SMOOTH * len(counts)
    return {el: (c + CHEMISTRY_SMOOTH) / denom for el, c in counts.items()}


def _pair_affinity(stats: dict, a: str, b: str, el_probs: dict[str, float]) -> float:
    pair       = tuple(sorted((a, b)))
    pair_total = float(sum(stats["pair_counts"].values()))
    num_pairs  = max(1, len(stats["pair_counts"]))
    p_pair     = (stats["pair_counts"].get(pair, 0.0) + CHEMISTRY_SMOOTH) / (
        pair_total + CHEMISTRY_SMOOTH * num_pairs
    )
    expected   = max(el_probs.get(a, 0.0) * el_probs.get(b, 0.0), 1e-12)
    return max(p_pair / expected, 1e-6)


def _sample_empirical_element_set(space_group: int) -> list[str] | None:
    family = get_family_key(space_group)
    stats  = CHEMISTRY_PRIORS["by_family"].get(family, {})
    if not stats or not stats["element_counts"]:
        stats = CHEMISTRY_PRIORS["global"]
    if not stats["element_counts"]:
        return None

    num_choices  = sorted(stats["num_elements_counts"])
    num_weights  = np.array([stats["num_elements_counts"][k] for k in num_choices], dtype=np.float64)
    num_weights /= num_weights.sum()
    target_n     = int(np.random.choice(num_choices, p=num_weights))

    el_items   = sorted(stats["element_counts"].items())
    all_els    = [k for k, _ in el_items]
    el_probs   = _normalized_element_probs(stats)
    base_w     = np.array([el_probs[k] ** CHEMISTRY_MARGINAL_TEMPER for k in all_els], dtype=np.float64)
    base_w    /= base_w.sum()

    chosen = [str(np.random.choice(all_els, p=base_w))]
    while len(chosen) < target_n:
        scores = []
        cands  = []
        for el in all_els:
            if el in chosen:
                continue
            s = max(el_probs[el] ** CHEMISTRY_MARGINAL_TEMPER, 1e-12)
            for prior in chosen:
                s *= _pair_affinity(stats, el, prior, el_probs) ** CHEMISTRY_PAIR_TEMPER
            cands.append(el)
            scores.append(s)
        if not cands:
            break
        sc = np.array(scores, dtype=np.float64)
        if not np.isfinite(sc).all() or sc.sum() <= 0:
            sc = np.ones(len(cands), dtype=np.float64)
        sc /= sc.sum()
        chosen.append(str(np.random.choice(cands, p=sc)))

    return chosen


def sample_valid_composition(space_group: int, require_general_position: bool = False):
    sampled = _sample_empirical_element_set(space_group)
    if sampled is not None:
        counts = _sample_wyckoff_compatible_counts(
            space_group, len(sampled),
            require_general_position=require_general_position,
        )
        if counts is not None:
            return sampled, counts

    wyckoff_data = get_wyckoffs_cached(space_group)
    valid_counts = [
        sum(len(pos) for pos in positions)
        for positions in wyckoff_data
        if sum(len(pos) for pos in positions) > 1
    ]
    if not valid_counts:
        return ["C"], [2]

    n_elements = int(np.random.randint(2, 4))
    elements   = random.sample(ELEMENTS, n_elements)
    counts     = _sample_wyckoff_compatible_counts(
        space_group, n_elements,
        require_general_position=require_general_position,
    )
    if counts is not None:
        return elements, counts

    fallback = max(2, min(valid_counts))
    return elements[:1], [fallback]

# -----------------------------------------------------------------------
# Structure generation
# -----------------------------------------------------------------------

def _generate_p1_structure():
    st           = pyxtal()
    n_elements   = random.randint(1, 3)
    selected     = random.sample(ELEMENTS, n_elements)
    n_atoms      = random.randint(2, 10)
    dist         = [n_atoms // n_elements] * n_elements
    for i in range(n_atoms % n_elements):
        dist[i] += 1
    st.from_random(dim=3, group=1, species=selected, numIons=dist)
    return st


def _detect_sg(structure) -> tuple[int | None, str | None]:
    atoms    = structure.to_ase()
    cell     = (atoms.get_cell(), atoms.get_scaled_positions(), atoms.get_atomic_numbers())
    detected = spglib.get_spacegroup(cell, symprec=1e-3)
    if not detected:
        return None, None
    m = re.search(r"\((\d+)\)", detected)
    return (int(m.group(1)), detected) if m else (None, detected)

# -----------------------------------------------------------------------
# Failure logging
# -----------------------------------------------------------------------

_failure_log_count = 0


def _log_failure(event: dict) -> None:
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


def get_valid_structure(space_group: int, max_attempts: int = 50,
                        debug_task_id=None, role: str = "main"):
    target_eg = sg_to_ext.get(int(space_group))
    for attempt in range(max_attempts):
        elements = None
        ratios   = None
        try:
            if space_group == 1:
                structure = _generate_p1_structure()
            else:
                elements, ratios = sample_valid_composition(
                    space_group,
                    require_general_position=(role == "main"),
                )
                structure = pyxtal()
                structure.from_random(dim=3, group=space_group,
                                      species=elements, numIons=ratios)

            detected_sg, detected_label = _detect_sg(structure)
            if role == "impurity":
                return structure
            detected_eg = sg_to_ext.get(int(detected_sg)) if detected_sg is not None else None
            if detected_eg == target_eg and detected_eg is not None:
                return structure

            _log_failure({
                "stage": "validate_structure_false",
                "task_id": debug_task_id, "role": role,
                "space_group": int(space_group),
                "family": get_family_key(space_group),
                "attempt": attempt + 1,
                "elements": elements, "ratios": ratios,
                "target_eg": int(target_eg) if target_eg is not None else None,
                "detected_sg": detected_sg, "detected_eg": int(detected_eg) if detected_eg is not None else None,
            })

        except Exception as exc:
            _log_failure({
                "stage": "pyxtal_from_random" if space_group != 1 else "p1_generation",
                "task_id": debug_task_id, "role": role,
                "space_group": int(space_group),
                "family": get_family_key(space_group),
                "attempt": attempt + 1,
                "elements": elements, "ratios": ratios,
                "exc_type": type(exc).__name__, "exc_message": str(exc),
            })

    print(f"[xrd_generator] WARN: SG {space_group} failed after {max_attempts} attempts")
    return None

# -----------------------------------------------------------------------
# Pattern generation
# -----------------------------------------------------------------------

def _extract_structural_info(structure):
    atoms = structure.to_ase()
    a, b, c, alpha, beta, gamma = atoms.get_cell_lengths_and_angles()
    sg        = structure.group.number
    positions = atoms.get_scaled_positions()
    symbols   = atoms.get_chemical_symbols()
    atom_list = [{"symbol": s, "position": p} for s, p in zip(symbols, positions)]
    return sg, a, b, c, alpha, beta, gamma, atom_list


def sample_empirical_nuisance(space_group: int, rng: np.random.Generator) -> dict:
    """
    Sample instrument nuisance parameters (U, V, W, X, Y, eta, zero, bkg)
    from the empirical RRUFF prior for the given space group's crystal family.
    """
    family = get_family_key(space_group)
    pool   = PRIORS["by_family"].get(family) or PRIORS["global"]
    row    = pool[int(rng.integers(0, len(pool)))]

    u   = max(5e-4, row["U"] * np.exp(rng.normal(0.0, 0.12)))
    w   = max(5e-4, row["W"] * np.exp(rng.normal(0.0, 0.15)))
    x   = max(0.0,  row["X"] * np.exp(rng.normal(0.0, 0.20)))
    y   = max(0.0,  row["Y"] * np.exp(rng.normal(0.0, 0.30)))
    eta = float(np.clip(row["eta"] + rng.normal(0.0, 0.06), 0.0, 1.0))

    v_scale = {"cubic": 4e-4, "ortho_tetra": 9e-4,
               "monoclinic": 1.4e-3}.get(family, 2.0e-3)
    v = float(np.clip(-abs(rng.normal(0.0, v_scale)), -0.05, 0.0))

    bkg = row["bkg"].copy()
    bkg *= np.exp(rng.normal(0.0, np.array([0.10, 0.15, 0.20, 0.20], dtype=np.float64)))

    return {"u": u, "v": v, "w": w, "x": x, "y": y, "eta": eta, "zero": 0.0, "bkg": bkg}


def _create_pattern_dict(sg, a, b, c, alpha, beta, gamma,
                          atom_list, nuisance, wavelength) -> dict:
    sg_symbol = spglib.get_spacegroup_type(sg)["international_short"]
    return {
        "phases": [{
            "placeholder": {
                "_space_group_name_H-M_alt": sg_symbol,
                "_cell_length_a":    float(a),
                "_cell_length_b":    float(b),
                "_cell_length_c":    float(c),
                "_cell_angle_alpha": float(alpha),
                "_cell_angle_beta":  float(beta),
                "_cell_angle_gamma": float(gamma),
                "_atom_site": [
                    {
                        "_label":          atom["symbol"],
                        "_type_symbol":    atom["symbol"],
                        "_fract_x":        float(atom["position"][0]),
                        "_fract_y":        float(atom["position"][1]),
                        "_fract_z":        float(atom["position"][2]),
                        "_occupancy":      1.0,
                        "_adp_type":       "iso",
                        "_B_iso_or_equiv": 0.0,
                    }
                    for atom in atom_list
                ],
            }
        }],
        "experiments": [{
            "NPD": {
                "_diffrn_radiation_probe":           "xray",
                "_diffrn_radiation_wavelength":      float(wavelength),
                "_pd_instr_resolution_u":            float(nuisance["u"]),
                "_pd_instr_resolution_v":            float(nuisance["v"]),
                "_pd_instr_resolution_w":            float(nuisance["w"]),
                "_pd_instr_resolution_x":            float(nuisance["x"]),
                "_pd_instr_resolution_y":            float(nuisance["y"]),
                "_pd_instr_reflex_asymmetry_p1":     0.0,
                "_pd_instr_reflex_asymmetry_p2":     0.0,
                "_pd_instr_reflex_asymmetry_p3":     0.0,
                "_pd_instr_reflex_asymmetry_p4":     0.0,
                "_pd_meas_2theta_offset":            float(nuisance["zero"]),
                "_pd_meas_2theta_range_min":         float(INSTRUMENT_PARAMS["twotheta_min"]),
                "_pd_meas_2theta_range_max":         float(INSTRUMENT_PARAMS["twotheta_max"]),
                "_pd_meas_2theta_range_inc":         float(INSTRUMENT_PARAMS["step_size"]),
                "_phase":        [{"_label": "placeholder", "_scale": 1.0}],
                "_pd_background": [
                    {"_2theta": float(INSTRUMENT_PARAMS["twotheta_min"]), "_intensity": 0.0},
                    {"_2theta": float(INSTRUMENT_PARAMS["twotheta_max"]), "_intensity": 0.0},
                ],
            }
        }],
    }


def cw_powder_pattern_backend(pattern_dict: dict,
                               structure) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute CW powder pattern.
    Fast path: cfml_py_utilities (direct call).
    Slow path: CFML_api via tacc/cfml_api_single_pattern.py subprocess.
    """
    if _cfml_py_utilities is not None:
        return _cfml_py_utilities.cw_powder_pattern_from_dict(pattern_dict)

    if _CFML_api is None:
        raise RuntimeError(
            "Neither pycrysfml.cfml_py_utilities nor CFML_api is available."
        )
    if not os.path.isfile(CFML_HELPER):
        raise RuntimeError(
            f"CFML_api helper not found at {CFML_HELPER}. "
            "Ensure tacc/cfml_api_single_pattern.py exists."
        )

    atoms = structure.to_ase()
    with tempfile.NamedTemporaryFile(suffix=".cif", delete=False) as fh:
        cif_path = fh.name
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as fh:
        out_path = fh.name

    try:
        ase_write(cif_path, atoms, format="cif")
        npd = pattern_dict["experiments"][0]["NPD"]
        subprocess.run(
            [
                sys.executable, CFML_HELPER,
                "--cif",          cif_path,
                "--out",          out_path,
                "--theta-min",    str(float(npd["_pd_meas_2theta_range_min"])),
                "--theta-max",    str(float(npd["_pd_meas_2theta_range_max"])),
                "--step",         str(float(npd["_pd_meas_2theta_range_inc"])),
                "--u-resolution", str(float(npd["_pd_instr_resolution_u"])),
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
        for p in (cif_path, out_path):
            try:
                os.remove(p)
            except OSError:
                pass

# -----------------------------------------------------------------------
# Augmentation
# -----------------------------------------------------------------------

def _rbf_gp_exact(twotheta: np.ndarray, l: float,
                  rng: np.random.Generator,
                  jitter: float = 1e-8) -> np.ndarray:
    x  = twotheta.astype(np.float64)
    d2 = (x[:, None] - x[None, :]) ** 2
    K  = np.exp(-d2 / (2.0 * l * l))
    K[np.diag_indices_from(K)] += jitter
    L  = cholesky(K, lower=True, check_finite=False)
    return (L @ rng.standard_normal(x.shape[0])).astype(np.float64)


def _rbf_gp_rff(twotheta: np.ndarray, l: float,
                rng: np.random.Generator, m_features: int) -> np.ndarray:
    x     = twotheta.astype(np.float64)
    m     = int(m_features)
    omega = rng.normal(0.0, 1.0 / l, size=m)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=m)
    w     = rng.normal(0.0, 1.0, size=m)
    return (np.sqrt(2.0 / m)
            * (np.cos(x[:, None] * omega[None, :] + phase[None, :]) @ w)
            ).astype(np.float64)


def _smooth_field(twotheta: np.ndarray, length_scale: float,
                  rng: np.random.Generator) -> np.ndarray:
    field = (_rbf_gp_exact(twotheta, length_scale, rng)
             if USE_EXACT_GP
             else _rbf_gp_rff(twotheta, length_scale, rng, RFF_FEATURES))
    field -= field.mean()
    std = float(field.std())
    if std > 0:
        field /= std
    return field


def _evaluate_empirical_background(nuisance: dict) -> np.ndarray:
    """Chebyshev polynomial background from empirical RRUFF coefficients.
    Much faster than GP/RFF for the baseline component."""
    coeffs = nuisance["bkg"] / MAX_INTENSITY
    return np.polynomial.chebyshev.chebval(X_SCALED, coeffs).astype(np.float64)


def _sample_amplitude_distortion_config(space_group: int,
                                         rng: np.random.Generator) -> tuple[bool, float]:
    family    = get_family_key(space_group)
    base_prob = {"cubic": 0.05, "ortho_tetra": 0.10,
                 "monoclinic": 0.15, "trig_hex": 0.15,
                 "triclinic_other": 0.20}[family]
    if PRIOR_MODE == "robust":
        base_prob += 0.05
    if EG_SAMPLING_MODE == "rruff":
        base_prob += 0.03
    enabled  = bool(rng.random() < min(base_prob, 0.35))
    strength = {"cubic": 0.02, "ortho_tetra": 0.04,
                "monoclinic": 0.06, "trig_hex": 0.06,
                "triclinic_other": 0.08}[family]
    return enabled, strength


def _apply_amplitude_distortion(intensity_01: np.ndarray, twotheta: np.ndarray,
                                 rng: np.random.Generator, strength: float) -> np.ndarray:
    if strength <= 0.0:
        return intensity_01.astype(np.float32, copy=True)
    field    = _smooth_field(twotheta, float(rng.uniform(2.5, 9.0)), rng)
    envelope = np.exp(strength * field)
    envelope /= np.median(envelope)
    out      = intensity_01.astype(np.float64, copy=True) * envelope
    peak     = float(out.max())
    if peak > 0:
        out /= peak
    return out.astype(np.float32)


def _add_background_noise(intensity_counts: np.ndarray,
                           twotheta: np.ndarray,
                           rng: np.random.Generator,
                           nuisance: dict,
                           space_group: int) -> np.ndarray:
    """
    Add empirical Chebyshev background + stochastic GP component + noise.
    """
    counts  = intensity_counts.astype(np.float64, copy=True)
    counts += _evaluate_empirical_background(nuisance) * MAX_INTENSITY

    family    = get_family_key(space_group)
    gp_scale  = {"cubic": (0.0, 10.0), "ortho_tetra": (5.0, 20.0),
                 "monoclinic": (8.0, 30.0), "trig_hex": (8.0, 30.0),
                 "triclinic_other": (10.0, 40.0)}[family]
    field     = _smooth_field(twotheta, float(rng.uniform(7.0, 35.0)), rng)
    gp        = field - field.min()
    total     = float(gp.sum())
    if total > 0:
        gp /= total
        gp *= float(rng.uniform(*gp_scale))
        counts += gp

    sigma_add  = float(rng.uniform(0.5, 8.0))
    counts    += rng.normal(0.0, sigma_add, size=counts.shape)
    sigma_mult = float(rng.uniform(0.0, 0.10 if PRIOR_MODE == "clean" else 0.14))
    counts    *= rng.normal(1.0, sigma_mult, size=counts.shape)

    return np.clip(counts, 0.0, None).astype(np.float32)


def _generate_pure_pattern_01(structure, rng: np.random.Generator,
                               nuisance: dict) -> tuple[int, np.ndarray]:
    """Returns (spacegroup, intensity_0_to_1 float32)."""
    sg, a, b, c, alpha, beta, gamma, atom_list = _extract_structural_info(structure)

    p_k1 = _create_pattern_dict(sg, a, b, c, alpha, beta, gamma, atom_list, nuisance, KALPHA1)
    p_k2 = _create_pattern_dict(sg, a, b, c, alpha, beta, gamma, atom_list, nuisance, KALPHA2)

    _, y1 = cw_powder_pattern_backend(p_k1, structure)
    _, y2 = cw_powder_pattern_backend(p_k2, structure)

    intensity = (W_KALPHA1 * np.asarray(y1, dtype=np.float32)
                 + W_KALPHA2 * np.asarray(y2, dtype=np.float32))
    max_val   = float(intensity.max())
    if max_val > 0:
        intensity = (intensity / max_val).astype(np.float32)

    distort, strength = _sample_amplitude_distortion_config(sg, rng)
    if distort:
        intensity = _apply_amplitude_distortion(intensity, TWO_THETA, rng, strength)

    return sg, intensity


def _mix_impurity(main_01: np.ndarray, imp_01: np.ndarray,
                  rng: np.random.Generator) -> np.ndarray:
    """Sparse impurity mixing — 70% chance of zero impurity."""
    p = float(rng.random())
    if p < 0.70:
        return main_01.astype(np.float32, copy=True)
    frac = (float(rng.uniform(0.005, 0.03)) if p < 0.90
            else float(rng.uniform(0.03, 0.10)))
    return ((1.0 - frac) * main_01 + frac * imp_01).astype(np.float32)

# -----------------------------------------------------------------------
# EG / SG sampling
# -----------------------------------------------------------------------

def sample_main_sg(rng: np.random.Generator) -> tuple[int, int]:
    """RRUFF-weighted (or uniform-weighted) sample of (extinction_group, space_group)."""
    eg  = int(rng.choice(SAMPLE_EGS, p=SAMPLE_EG_WEIGHTS))
    sgs = ext_to_sg[eg]
    sg  = int(sgs[int(rng.integers(0, len(sgs)))])
    return eg, sg


def sample_main_sg_balanced(task_id: int, seed: int) -> tuple[int, int]:
    """
    Deterministic balanced block schedule — every 99 consecutive tasks
    cover all extinction groups exactly once (shuffled per block).
    Used when EG_SAMPLING_MODE='uniform'.
    """
    block_size   = ALL_EGS.size                        # 99
    block_id     = task_id // block_size
    block_offset = task_id  % block_size
    block_rng    = np.random.default_rng(np.random.SeedSequence([seed, block_id, 17]))
    eg_perm      = block_rng.permutation(ALL_EGS)
    eg           = int(eg_perm[block_offset])
    sg_rng       = np.random.default_rng(np.random.SeedSequence([seed, task_id, 23]))
    sgs          = ext_to_sg[eg]
    sg           = int(sgs[int(sg_rng.integers(0, len(sgs)))])
    return eg, sg


def sample_impurity_sg_uniform(rng: np.random.Generator) -> int:
    return int(ALL_SPACE_GROUPS[int(rng.integers(0, ALL_SPACE_GROUPS.size))])

# -----------------------------------------------------------------------
# Task builder
# -----------------------------------------------------------------------

def build_task(task_id: int, seed: int) -> tuple:
    """
    Build a worker task tuple: (task_id, eg, sg_main, sg_imp, wseed).
    Uses EG_SAMPLING_MODE to choose balanced vs RRUFF sampling.
    """
    rng = np.random.default_rng(np.random.SeedSequence([seed, task_id]))

    if EG_SAMPLING_MODE == "uniform":
        eg, sg_main = sample_main_sg_balanced(task_id, seed)
    else:
        eg, sg_main = sample_main_sg(rng)

    sg_imp = sample_impurity_sg_uniform(rng)
    wseed  = int(rng.integers(0, 2**63 - 1))
    return (task_id, eg, sg_main, sg_imp, wseed)

# -----------------------------------------------------------------------
# Worker  (called by multiprocessing.Pool in streaming_dataset.py)
# -----------------------------------------------------------------------

def worker(task: tuple) -> tuple:
    """
    Args:
        task: (task_id, eg, sg_main, sg_imp, seed)

    Returns:
        (task_id, sg_out, eg, intensity_float32)  on success
        (task_id, None)                           on failure after retries
    """
    task_id, eg, sg_main, sg_imp, seed = task
    rng = np.random.default_rng(seed)

    last_error = None
    for attempt in range(SAMPLE_RETRY_LIMIT):
        try:
            st_main = get_valid_structure(sg_main, debug_task_id=task_id, role="main")
            if st_main is None:
                continue

            st_imp = get_valid_structure(sg_imp, debug_task_id=task_id, role="impurity")
            if st_imp is None:
                continue

            nuisance_main = sample_empirical_nuisance(sg_main, rng)
            nuisance_imp  = sample_empirical_nuisance(sg_imp,  rng)

            sg_out, main_01 = _generate_pure_pattern_01(st_main, rng, nuisance_main)
            _,      imp_01  = _generate_pure_pattern_01(st_imp,  rng, nuisance_imp)

            mixed_01   = _mix_impurity(main_01, imp_01, rng)
            augmented  = _add_background_noise(
                mixed_01 * MAX_INTENSITY, TWO_THETA, rng, nuisance_main, sg_out
            )

            if not np.all(np.isfinite(augmented)):
                raise RuntimeError("intensity contains non-finite values")

            return (task_id, sg_out, eg, augmented.astype(np.float32))

        except Exception as e:
            last_error = e
            continue

    print(
        f"[xrd_generator] WARN: dropped task_id={task_id} after "
        f"{SAMPLE_RETRY_LIMIT} attempts "
        f"(sg_main={sg_main}, sg_imp={sg_imp}): {last_error}"
    )
    return (task_id, None)
