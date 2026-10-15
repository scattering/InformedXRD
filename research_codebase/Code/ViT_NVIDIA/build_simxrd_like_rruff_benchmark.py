#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import gemmi
import h5py
import numpy as np


RRUFF_ID_RE = re.compile(r"__(R[^_]+)(?:-[^_]+)?__")
XY_LINE_RE = re.compile(r"^\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a SimXRD-like RRUFF benchmark from matched xy_raw and DIF files."
    )
    parser.add_argument("--xy-dir", required=True)
    parser.add_argument("--dif-dir", required=True)
    parser.add_argument("--final-table-path", required=True)
    parser.add_argument("--output-h5", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--require-cu", action="store_true", help="Require Cu-like wavelength in DIF file")
    parser.add_argument("--two-theta-min", type=float, default=5.0)
    parser.add_argument("--two-theta-max", type=float, default=90.0)
    parser.add_argument("--step-size", type=float, default=0.01)
    return parser.parse_args()


def extract_rruff_id(name: str) -> str | None:
    match = RRUFF_ID_RE.search(name)
    return match.group(1) if match else None


def load_sg_to_eg_map(final_table_path: str) -> dict[int, int]:
    sg_to_eg: dict[int, int] = {}
    with open(final_table_path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sg_to_eg[int(row["Space Group"])] = int(row["Extinction Group"])
    if len(sg_to_eg) != 230:
        raise ValueError(f"Expected 230 SG->EG mappings, found {len(sg_to_eg)}")
    return sg_to_eg


def parse_dif_metadata(path: Path) -> tuple[str | None, str | None]:
    text = path.read_text(errors="ignore")
    sg_match = re.search(r"SPACE GROUP\s*[:=]\s*([^\n\r]+)", text)
    wl_match = re.search(r"X-RAY WAVELENGTH\s*[:=]\s*([^\n\r]+)", text)
    sg_symbol = sg_match.group(1).strip() if sg_match else None
    wavelength = wl_match.group(1).strip() if wl_match else None
    return sg_symbol, wavelength


def is_cu_like(wavelength: str | None) -> bool:
    if not wavelength:
        return False
    nums = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+", wavelength)]
    return any(1.53 <= x <= 1.55 for x in nums)


def normalize_hm_symbol(symbol: str) -> list[str]:
    raw = symbol.strip().replace("\\par", "").strip()
    candidates = [raw]

    s = raw.replace("_", "")
    s = re.sub(r"^([A-ZR])(?=[0-9-])", r"\1 ", s)
    candidates.append(s)

    s2 = raw.replace("_", "")
    s2 = re.sub(r"^([A-ZR])(?=[0-9-])", r"\1 ", s2)
    s2 = s2.replace("/", " /")
    candidates.append(s2)

    seen = []
    for cand in candidates:
        cand = " ".join(cand.split())
        if cand and cand not in seen:
            seen.append(cand)
    return seen


def space_group_symbol_to_number(symbol: str) -> int | None:
    for candidate in normalize_hm_symbol(symbol):
        sg = gemmi.find_spacegroup_by_name(candidate)
        if sg is not None:
            return int(sg.number)
    return None


def parse_xy_raw(path: Path) -> tuple[np.ndarray, np.ndarray]:
    two_theta = []
    intensity = []
    with path.open() as handle:
        for line in handle:
            m = XY_LINE_RE.match(line)
            if m:
                two_theta.append(float(m.group(1)))
                intensity.append(float(m.group(2)))
    if not two_theta:
        raise ValueError(f"No XY rows parsed from {path}")
    return np.asarray(two_theta, dtype=np.float32), np.asarray(intensity, dtype=np.float32)


def main() -> None:
    args = parse_args()
    xy_dir = Path(args.xy_dir)
    dif_dir = Path(args.dif_dir)
    sg_to_eg = load_sg_to_eg_map(args.final_table_path)

    xy_by_id = {extract_rruff_id(p.name): p for p in xy_dir.glob("*.txt") if extract_rruff_id(p.name)}
    dif_by_id = {extract_rruff_id(p.name): p for p in dif_dir.glob("*.txt") if extract_rruff_id(p.name)}
    common_ids = sorted(set(xy_by_id) & set(dif_by_id))

    spectra = []
    ext_labels = []
    rruffids = []
    sg_numbers = []
    sg_symbols = []
    skipped = {"missing_sg": 0, "non_cu": 0, "unparsed_sg": 0, "sg_not_in_table": 0, "bad_xy": 0}
    n_points = int(round((args.two_theta_max - args.two_theta_min) / args.step_size)) + 1
    reference_grid = np.linspace(args.two_theta_min, args.two_theta_max, n_points, dtype=np.float32)

    for rid in common_ids:
        sg_symbol, wavelength = parse_dif_metadata(dif_by_id[rid])
        if not sg_symbol:
            skipped["missing_sg"] += 1
            continue
        if args.require_cu and not is_cu_like(wavelength):
            skipped["non_cu"] += 1
            continue

        sg_num = space_group_symbol_to_number(sg_symbol)
        if sg_num is None:
            skipped["unparsed_sg"] += 1
            continue
        eg = sg_to_eg.get(sg_num)
        if eg is None:
            skipped["sg_not_in_table"] += 1
            continue

        try:
            two_theta, intensity = parse_xy_raw(xy_by_id[rid])
        except Exception:
            skipped["bad_xy"] += 1
            continue

        intensity = np.interp(reference_grid, two_theta, intensity, left=0.0, right=0.0).astype(np.float32)

        max_val = float(np.max(intensity))
        if max_val > 0:
            intensity = intensity / max_val

        spectra.append(intensity.astype(np.float32))
        ext_labels.append(int(eg))
        rruffids.append(rid)
        sg_numbers.append(int(sg_num))
        sg_symbols.append(sg_symbol)

    X_test = np.stack(spectra, axis=0)
    y_test = np.asarray(ext_labels, dtype=np.int64)
    sg_numbers_arr = np.asarray(sg_numbers, dtype=np.int64)
    rruffids_arr = np.asarray(rruffids, dtype=h5py.string_dtype("utf-8"))
    sg_symbols_arr = np.asarray(sg_symbols, dtype=h5py.string_dtype("utf-8"))

    output_h5 = Path(args.output_h5)
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_h5, "w") as handle:
        handle.create_dataset("X_test", data=X_test, compression="gzip")
        handle.create_dataset("y_test", data=y_test)
        handle.create_dataset("sg_number_test", data=sg_numbers_arr)
        handle.create_dataset("rruffid_test", data=rruffids_arr)
        handle.create_dataset("sg_symbol_test", data=sg_symbols_arr)
        handle.create_dataset("two_theta", data=reference_grid.astype(np.float32))

    summary = {
        "xy_dir": str(xy_dir),
        "dif_dir": str(dif_dir),
        "common_rruffid": len(common_ids),
        "included_samples": int(y_test.shape[0]),
        "skipped": skipped,
        "require_cu": bool(args.require_cu),
        "spec_length": int(X_test.shape[1]),
        "two_theta_min": float(reference_grid.min()),
        "two_theta_max": float(reference_grid.max()),
        "unique_ext_groups": int(len(set(ext_labels))),
        "sample_rruffids": rruffids[:10],
    }
    Path(args.output_json).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
