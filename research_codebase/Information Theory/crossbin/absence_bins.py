"""
absence_bins.py -- canonical, gemmi-free extinction-group absence-bin map.

Two extinction groups (EGs) share an *absence bin* iff they produce the same
set of systematically absent reflections in the standard setting -- i.e. they
have the same lattice centering and the same glide/screw operators in each
symbol position, regardless of crystal system / Laue class. EGs in the same
bin differ only by point-group (Laue) symmetry, which carries no systematic
absences; confusing two such EGs is a *within-bin* (Laue) error. Confusing EGs
in different bins is a *cross-bin* error: the absence pattern itself was misread.

The map is built directly from the canonical extinction symbols
(canonical_extinction_to_space_group.csv), so it is setting-consistent and
reproducible with numpy alone -- no gemmi / space-group-setting dependence.
"""
from __future__ import annotations
import csv
from pathlib import Path

# 37-bit extinction template layout (crystal system | lattice | per-position ops).
# The absence pattern is determined by lattice + operators only (system ignored).
FEATURES = [
    "Sys_Triclinic", "Sys_Monoclinic", "Sys_Orthorhombic", "Sys_Tetragonal",
    "Sys_Trigonal", "Sys_Hexagonal", "Sys_Cubic",
    "Lattice_P", "Lattice_I", "Lattice_F", "Lattice_C", "Lattice_R",
    "Pos1_21", "Pos1_31", "Pos1_41", "Pos1_42", "Pos1_61", "Pos1_62", "Pos1_63",
    "Pos1_a", "Pos1_b", "Pos1_c", "Pos1_d", "Pos1_n",
    "Pos2_21", "Pos2_a", "Pos2_b", "Pos2_c", "Pos2_d", "Pos2_n",
    "Pos3_21", "Pos3_a", "Pos3_b", "Pos3_c", "Pos3_d", "Pos3_n", "Pos3_ab",
]
IDX = {n: i for i, n in enumerate(FEATURES)}
ABSENCE_START = IDX["Lattice_P"]          # bits >= here are absence-relevant
LATTICE_MAP = {
    "P": "Lattice_P", "I": "Lattice_I", "F": "Lattice_F", "C": "Lattice_C",
    "A": "Lattice_C", "B": "Lattice_C", "R_obv": "Lattice_R", "R_rev": "Lattice_R", "R": "Lattice_R",
}
DEFAULT_CSV = Path(__file__).resolve().parent / "canonical_extinction_to_space_group.csv"


def _normalize_symbol(symbol: str) -> str:
    s = symbol.split("(equiv:")[0].strip()
    for a, b in [("R (obv)-", "R_obv "), ("R (obv) -", "R_obv "), ("R (obv)", "R_obv"),
                 ("R (rev)-", "R_rev "), ("R (rev) -", "R_rev "), ("R (rev)", "R_rev"),
                 ("(ab)", "ab")]:
        s = s.replace(a, b)
    return " ".join(s.split())


def absence_signature(symbol: str) -> tuple:
    """Lattice + per-position glide/screw bits (the absence pattern)."""
    parts = _normalize_symbol(symbol).split()
    bits = [0] * len(FEATURES)
    bits[IDX[LATTICE_MAP[parts[0]]]] = 1
    for pos_idx, token in enumerate(parts[1:], start=1):
        if pos_idx > 3:
            break
        for op in token.split("/"):
            if op in {"-", "1"}:
                continue
            feat = f"Pos{pos_idx}_{op}"
            if feat in IDX:
                bits[IDX[feat]] = 1
    return tuple(bits[ABSENCE_START:])          # drop the 7 crystal-system bits


def build_bin_map(csv_path: Path | str = DEFAULT_CSV) -> dict:
    """Return {eg_id (1..99) -> integer absence-bin id}."""
    eg_symbol = {}
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            eg_symbol[int(row["Index"]) + 1] = row["Canonical Extinction Group"]
    sig = {e: absence_signature(s) for e, s in eg_symbol.items()}
    uniq = {s: i for i, s in enumerate(sorted(set(sig.values()), key=str))}
    return {e: uniq[sig[e]] for e in sig}


def decompose(true_egs, pred_egs, bin_of=None):
    """Within-bin (Laue) vs cross-bin (absence-misread) split of wrong calls."""
    if bin_of is None:
        bin_of = build_bin_map()
    wrong = [(t, p) for t, p in zip(true_egs, pred_egs) if t != p]
    within = [(t, p) for t, p in wrong if bin_of.get(t) == bin_of.get(p)]
    cross = [(t, p) for t, p in wrong if bin_of.get(t) != bin_of.get(p)]
    n = len(wrong)
    return {
        "n_total": len(true_egs),
        "n_correct": len(true_egs) - n,
        "n_wrong": n,
        "within_bin_Laue": len(within),
        "cross_bin_absence": len(cross),
        "within_bin_frac_of_wrong": (len(within) / n) if n else 0.0,
        "cross_bin_frac_of_wrong": (len(cross) / n) if n else 0.0,
    }
