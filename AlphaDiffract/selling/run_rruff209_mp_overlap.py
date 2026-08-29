#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.core import Structure
from pymatgen.symmetry.groups import SpaceGroup
from scipy.signal import find_peaks

from mp_overlap_search import Case, Candidate, match_strength, two_theta_to_Q


WAVELENGTH = 1.5406


def parse_args():
    p = argparse.ArgumentParser(description="Run the validated RRUFF-209 -> MP overlap search.")
    p.add_argument("--cases-json", required=True)
    p.add_argument("--mp-jsonl", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--q-min", type=float, default=1.5)
    p.add_argument("--peak-prominence-frac", type=float, default=0.02)
    p.add_argument("--two-theta-min", type=float, default=10.0)
    p.add_argument("--two-theta-max", type=float, default=50.0)
    p.add_argument("--cell-tol", type=float, default=0.6)
    p.add_argument("--n-decoy", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-cases", type=int, default=0)
    p.add_argument("--processed-dir", default="")
    return p.parse_args()


def read_processed_xy(path: Path, tt_min: float, tt_max: float):
    xs, ys = [], []
    for line in path.read_text(errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "," not in s:
            continue
        a, b = s.split(",", 1)
        xs.append(float(a))
        ys.append(float(b))
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    mask = (x >= tt_min) & (x <= tt_max)
    return x[mask], y[mask]


def extract_q_peaks(path: Path, tt_min: float, tt_max: float, prominence_frac: float, q_min: float):
    tt, intensity = read_processed_xy(path, tt_min, tt_max)
    if tt.size == 0:
        return np.zeros(0, dtype=float)
    prom = max(float(np.max(intensity)) * prominence_frac, 1e-6)
    peaks, _ = find_peaks(intensity, prominence=prom)
    q = two_theta_to_Q(tt[peaks], wavelength=WAVELENGTH)
    return np.sort(q[q >= q_min])


def resolve_processed_xy(raw_path: str, processed_dir: str) -> Path:
    path = Path(raw_path)
    if path.exists():
        return path
    if processed_dir:
        candidate = Path(processed_dir) / path.name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"processed XY not found: {raw_path}")


def spg_number_from_symbol(symbol: str) -> int:
    return int(SpaceGroup(symbol).int_number)


def load_mp_pool(mp_jsonl: Path, sg_numbers: set[int], tt_min: float, tt_max: float):
    calc = XRDCalculator(wavelength=WAVELENGTH)
    by_sg = defaultdict(list)
    with mp_jsonl.open() as handle:
        for line in handle:
            row = json.loads(line)
            try:
                sg_num = spg_number_from_symbol(row["spg"])
            except Exception:
                continue
            if sg_num not in sg_numbers:
                continue
            structure = Structure.from_dict(row["structure"])
            pattern = calc.get_pattern(structure, two_theta_range=(tt_min, tt_max))
            q_peaks = np.asarray(two_theta_to_Q(pattern.x, wavelength=WAVELENGTH), dtype=float)
            lat = next((ch.upper() for ch in row["spg"] if ch.isalpha()), "P")
            lattice = structure.lattice
            cell = (
                float(lattice.a),
                float(lattice.b),
                float(lattice.c),
                float(lattice.alpha),
                float(lattice.beta),
                float(lattice.gamma),
            )
            by_sg[sg_num].append(
                Candidate(
                    mp_id=row["material_id"],
                    sg=sg_num,
                    lat=lat,
                    cell=cell,
                    q_peaks=q_peaks,
                )
            )
    return by_sg


def main():
    args = parse_args()
    payload = json.loads(Path(args.cases_json).read_text())
    rows = payload["rows"]
    if args.max_cases and args.max_cases > 0:
        rows = rows[: args.max_cases]
    sg_numbers = {int(row["standard_sg_number"]) for row in rows}
    mp_by_sg = load_mp_pool(Path(args.mp_jsonl), sg_numbers, args.two_theta_min, args.two_theta_max)
    rng = np.random.default_rng(args.seed)

    results = []
    for row in rows:
        q_peaks = extract_q_peaks(
            resolve_processed_xy(row["processed_xy"], args.processed_dir),
            tt_min=args.two_theta_min,
            tt_max=args.two_theta_max,
            prominence_frac=args.peak_prominence_frac,
            q_min=args.q_min,
        )
        case = Case(
            rid=row["rid"],
            sg=int(row["standard_sg_number"]),
            lat=row["centering"],
            cell=tuple(float(v) for v in row["cell"]),
            q_peaks=q_peaks,
        )
        pool = mp_by_sg.get(case.sg, [])
        match = match_strength(case, pool, cell_tol=args.cell_tol, n_decoy=args.n_decoy, rng=rng)
        results.append(
            {
                "rid": row["rid"],
                "name": row["name"],
                "standard_sg_number": int(row["standard_sg_number"]),
                "standard_sg_symbol": row["standard_sg_symbol"],
                "processed_xy": row["processed_xy"],
                "n_case_q_peaks": int(q_peaks.size),
                **match,
            }
        )

    out = {
        "n_cases": len(results),
        "mp_jsonl": args.mp_jsonl,
        "parameters": {
            "q_min": args.q_min,
            "peak_prominence_frac": args.peak_prominence_frac,
            "two_theta_min": args.two_theta_min,
            "two_theta_max": args.two_theta_max,
            "cell_tol": args.cell_tol,
            "n_decoy": args.n_decoy,
            "seed": args.seed,
        },
        "results": results,
    }
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"n_cases": len(results), "output_json": str(out_path)}, indent=2))


if __name__ == "__main__":
    main()
