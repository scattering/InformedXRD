#!/usr/bin/env python3
"""
Construct the canonical no-window binary ideal channel over the six diagnostic
families used in Gate 2 / Gate 3.

This removes sampled-structure, lattice, and 2theta-window effects entirely:

1. Load the canonical EG -> SG mapping CSV.
2. For each SG, determine whether each prototype reflection family
   (100, 010, 001, 110, 101, 011) is systematically absent from the space-group
   symmetry operations alone.
3. Collapse SGs back to the 99 canonical EGs.
4. Run exact deterministic-channel analysis on the resulting binary masks.

The output is the clean mask-design ceiling for this specific six-family binary
mask, together with any SG-internal inconsistencies within an EG.
"""

from __future__ import annotations

import argparse
import ast
import csv
import itertools
import json
import re
import urllib.request
from pathlib import Path

import numpy as np

from ideal_channel_analysis import ideal_channel_analysis


FAMILY_SPECS = [
    ("100", (1, 0, 0)),
    ("010", (0, 1, 0)),
    ("001", (0, 0, 1)),
    ("110", (1, 1, 0)),
    ("101", (1, 0, 1)),
    ("011", (0, 1, 1)),
]

EXPO_URL = "https://www.ba.ic.cnr.it/softwareic/expo/extinction_symbols/"


def parse_args():
    parser = argparse.ArgumentParser(description="Exact canonical no-window ideal-channel construction for the six-family binary mask.")
    parser.add_argument("--mapping-csv", required=True)
    parser.add_argument("--out-json", default="")
    return parser.parse_args()


def _load_spglib():
    try:
        import spglib  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("spglib is required for canonical_no_window_ideal_channel.py") from exc
    return spglib


def load_eg_to_sgs(mapping_csv: str | Path) -> list[tuple[int, str, list[int]]]:
    rows: list[tuple[int, str, list[int]]] = []
    with open(mapping_csv, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            eg_idx = int(row["Index"])
            canon = row["Canonical Extinction Group"]
            sgs = [int(v) for v in ast.literal_eval(row["Space Group Numbers"])]
            rows.append((eg_idx, canon, sgs))
    return rows


def _normalize_symbol(symbol: str) -> str:
    symbol = symbol.replace("(ab)", "ab").replace("(ac)", "ac").replace("(bc)", "bc")
    symbol = symbol.replace("R (obv)-", "R_obv ").replace("R (obv) -", "R_obv ")
    symbol = symbol.replace("R (obv)", "R_obv")
    symbol = symbol.replace("R (rev)-", "R_rev ").replace("R (rev) -", "R_rev ")
    symbol = symbol.replace("R (rev)", "R_rev")
    return " ".join(symbol.split())


def fetch_sg_to_expo_symbol() -> dict[int, str]:
    html = urllib.request.urlopen(EXPO_URL, timeout=30).read().decode("utf-8", errors="ignore")
    rows = re.findall(r"<tr.*?>(.*?)</tr>", html, flags=re.S | re.I)
    sg_to_symbol: dict[int, str] = {}
    for row in rows:
        cells = re.findall(r"<td.*?>(.*?)</td>", row, flags=re.S | re.I)
        if len(cells) != 3:
            continue
        idx_txt = re.sub(r"<.*?>", "", cells[0]).strip()
        if not idx_txt.isdigit():
            continue
        symbol = re.sub(r"<.*?>", "", cells[1]).strip().replace("\xa0", " ").replace("–", "-")
        sg_raw = re.sub(r"<.*?>", "", cells[2]).strip().replace("\xa0", " ").replace("–", "-")
        for sg_num in re.findall(r"\((\d+)\)", sg_raw):
            sg_to_symbol[int(sg_num)] = symbol
    if len(sg_to_symbol) < 200:
        raise RuntimeError(f"Unexpectedly small SG->EXPO symbol map: {len(sg_to_symbol)} entries")
    return sg_to_symbol


def _map_lattice(lat: str, perm: tuple[int, int, int]) -> str:
    if lat in {"A", "B", "C"}:
        axis = {"A": 0, "B": 1, "C": 2}[lat]
        new_axis = perm[axis]
        return {0: "A", 1: "B", 2: "C"}[new_axis]
    return lat


def _map_token(token: str, perm: tuple[int, int, int]) -> str:
    if token in {"-", "1", "21", "31", "41", "42", "61", "62", "63"}:
        return token
    letter_map = {"a": "abc"[perm[0]], "b": "abc"[perm[1]], "c": "abc"[perm[2]]}
    mapped = "".join(letter_map.get(ch, ch) for ch in token)
    if set(mapped) <= set("abc"):
        mapped = "".join(sorted(mapped))
    return mapped


def transform_symbol(canonical_symbol: str, perm: tuple[int, int, int]) -> str:
    parts = _normalize_symbol(canonical_symbol).split()
    lat = _map_lattice(parts[0], perm)
    toks = (parts[1:4] + ["1", "1", "1"])[:3]
    mapped = [_map_token(tok, perm) for tok in toks]
    raw = ["", "", ""]
    for i, tok in enumerate(mapped):
        raw[perm[i]] = tok
    return _normalize_symbol(" ".join([lat] + raw))


def infer_permutation(canonical_symbol: str, raw_symbol: str) -> tuple[int, int, int]:
    canon = _normalize_symbol(canonical_symbol)
    raw = _normalize_symbol(raw_symbol)
    matches = [perm for perm in itertools.permutations([0, 1, 2]) if transform_symbol(canon, perm) == raw]
    if not matches:
        raise RuntimeError(f"Could not align canonical symbol '{canonical_symbol}' to raw symbol '{raw_symbol}'")
    return matches[0]


def permute_hkl(hkl: tuple[int, int, int], perm: tuple[int, int, int]) -> tuple[int, int, int]:
    raw = [0, 0, 0]
    for i, value in enumerate(hkl):
        raw[perm[i]] = int(value)
    return tuple(raw)


def is_systematically_absent(spglib, sg_number: int, hkl: tuple[int, int, int], tol: float = 1e-10) -> bool:
    """
    Determine whether reflection hkl is systematically absent for a general
    position in the given space group.

    For operation x' = R x + t, the contribution to F(h) is
        exp(2πi h·t) * exp(2πi (R^T h)·x)
    Group operations by the transformed reciprocal index q = R^T h. A general
    reflection is systematically absent iff the phase sum vanishes in every q-bin.
    """
    sym = spglib.get_symmetry_from_database(int(sg_number))
    rotations = np.asarray(sym["rotations"], dtype=np.int64)
    translations = np.asarray(sym["translations"], dtype=np.float64)
    h = np.asarray(hkl, dtype=np.int64)

    bins: dict[tuple[int, int, int], complex] = {}
    for R, t in zip(rotations, translations):
        q = tuple(int(v) for v in (R.T @ h).tolist())
        phase = np.exp(2j * np.pi * float(np.dot(h, t)))
        bins[q] = bins.get(q, 0.0 + 0.0j) + phase

    return all(abs(val) < tol for val in bins.values())


def sg_family_mask(spglib, sg_number: int) -> list[str]:
    raise RuntimeError("sg_family_mask requires canonical alignment context")


def sg_family_mask_canonical(
    spglib,
    sg_number: int,
    canonical_symbol: str,
    sg_to_raw_symbol: dict[int, str],
) -> list[str]:
    raw_symbol = sg_to_raw_symbol[int(sg_number)]
    perm = infer_permutation(canonical_symbol, raw_symbol)
    states = []
    for _name, hkl in FAMILY_SPECS:
        raw_hkl = permute_hkl(hkl, perm)
        states.append("absent" if is_systematically_absent(spglib, sg_number, raw_hkl) else "present")
    return states


def build_canonical_masks(mapping_csv: str | Path) -> dict:
    spglib = _load_spglib()
    rows = load_eg_to_sgs(mapping_csv)
    sg_to_raw_symbol = fetch_sg_to_expo_symbol()

    eg_masks: list[list[str]] = []
    eg_details = []
    inconsistent = []

    for eg_idx, canon, sgs in rows:
        canonical_main = canon.split("(equiv:")[0].strip()
        sg_masks = {
            int(sg): sg_family_mask_canonical(spglib, int(sg), canonical_main, sg_to_raw_symbol)
            for sg in sgs
        }
        unique_masks = {tuple(mask) for mask in sg_masks.values()}
        if len(unique_masks) != 1:
            inconsistent.append(
                {
                    "eg_index": int(eg_idx),
                    "canonical_symbol": canon,
                    "sg_masks": {str(k): v for k, v in sg_masks.items()},
                }
            )
        # Keep deterministic canonical surface explicit. If an EG has multiple
        # masks across its member SGs, retain the lexicographically first so the
        # issue is visible in the inconsistency report rather than silently mixed.
        canonical_mask = list(sorted(unique_masks)[0])
        eg_masks.append(canonical_mask)
        eg_details.append(
            {
                "eg_index": int(eg_idx),
                "canonical_symbol": canon,
                "space_groups": [int(v) for v in sgs],
                "mask": canonical_mask,
            }
        )

    analysis = ideal_channel_analysis(eg_masks)
    analysis["families"] = [name for name, _ in FAMILY_SPECS]
    analysis["eg_details"] = eg_details
    analysis["inconsistent_egs"] = inconsistent
    analysis["inconsistent_eg_count"] = int(len(inconsistent))
    return analysis


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def main():
    args = parse_args()
    out = build_canonical_masks(args.mapping_csv)
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(out, indent=2, default=_json_default))
    print(json.dumps(out, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
