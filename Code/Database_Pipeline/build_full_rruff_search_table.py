#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


EXPECTED_LENGTH = 8501
EXPECTED_TWOTHETA_MIN = 5.0
EXPECTED_TWOTHETA_MAX = 90.0
EXPECTED_STEP = 0.01
REFERENCE_CASE_ALIASES = {
    "Magnesiotaaffeite2N2S__X090001.9__11416": "Magnesiotaaffeite2N2S__R090035-9__10055",
    "Yttriumaluminumgarnet__X090003.9__10005": "Yttriumaluminumgarnet__R110105-9__11190",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a searchable table over the full preserved RRUFF Cu-Kalpha source surface.")
    p.add_argument("--manifest-json", required=True)
    p.add_argument("--reference-manifest-json", required=True)
    p.add_argument("--rruff-icsd-json", required=True)
    p.add_argument("--xy-dir", required=True)
    p.add_argument("--dif-dir", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-csv", required=True)
    return p.parse_args()


def parse_xy(path: Path) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("##"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 2:
                continue
            try:
                x = float(parts[0])
                y = float(parts[1])
            except ValueError:
                continue
            xs.append(x)
            ys.append(y)
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def normalize(y: np.ndarray) -> np.ndarray:
    ymax = float(np.max(np.abs(y))) if y.size else 0.0
    if ymax <= 0:
        return np.zeros_like(y)
    return y / ymax


def validate_grid(xs: np.ndarray) -> bool:
    return (
        len(xs) == EXPECTED_LENGTH
        and abs(float(xs[0]) - EXPECTED_TWOTHETA_MIN) <= 5e-3
        and abs(float(xs[-1]) - EXPECTED_TWOTHETA_MAX) <= 5e-3
        and bool(np.allclose(np.diff(xs), EXPECTED_STEP, atol=1e-4))
    )


def top_peaks(xs: np.ndarray, y_norm: np.ndarray, k: int = 8) -> tuple[list[float], list[float]]:
    if len(y_norm) < 3:
        return [], []
    maxima = np.where((y_norm[1:-1] >= y_norm[:-2]) & (y_norm[1:-1] >= y_norm[2:]))[0] + 1
    if maxima.size == 0:
        return [], []
    order = np.argsort(y_norm[maxima])[::-1][:k]
    peak_idx = maxima[order]
    return xs[peak_idx].round(4).tolist(), y_norm[peak_idx].round(6).tolist()


def build_source_case_id(row: dict[str, object]) -> str:
    suffix = Path(str(row["dif_file"])).stem.split("__")[-1]
    return f"{str(row['name']).strip()}__{str(row['rid']).strip()}__{suffix}"


def canonical_case_id(case_id: str) -> str:
    return REFERENCE_CASE_ALIASES.get(case_id, case_id)


def load_json(path: str):
    with open(path) as fh:
        return json.load(fh)


def main() -> None:
    args = parse_args()

    manifest = load_json(args.manifest_json)
    reference = load_json(args.reference_manifest_json)
    rruff_icsd = load_json(args.rruff_icsd_json)

    xy_dir = Path(args.xy_dir)
    dif_dir = Path(args.dif_dir)
    reference_ids = {canonical_case_id(str(row["rid"])) for row in reference}
    icsd_by_rid = {row["rid"]: row for row in rruff_icsd["results"]}

    rows = []
    for src in manifest:
        rid = str(src["rid"]).strip()
        source_case_id = build_source_case_id(src)
        xy_path = xy_dir / str(src["xy_file"])
        dif_path = dif_dir / str(src["dif_file"])
        xs = ys = None
        grid_ok = False
        n_points = 0
        theta_min = theta_max = step_mean = max_intensity = None
        peak_pos = []
        peak_val = []
        if xy_path.exists():
            try:
                xs, ys = parse_xy(xy_path)
                n_points = int(len(xs))
                if n_points:
                    theta_min = float(xs[0])
                    theta_max = float(xs[-1])
                    step_mean = float(np.mean(np.diff(xs))) if n_points > 1 else None
                    max_intensity = float(np.max(ys))
                    grid_ok = validate_grid(xs)
                    y_norm = normalize(ys)
                    peak_pos, peak_val = top_peaks(xs, y_norm)
            except Exception:
                pass

        anchor_row = icsd_by_rid.get(rid)
        top = anchor_row["top_matches"][0] if anchor_row and anchor_row["top_matches"] else None
        loose = bool(top and top["length_rel_mean"] <= 0.05 and top["volume_rel"] <= 0.15)
        row = {
            "rid": rid,
            "name": str(src["name"]).strip(),
            "source_case_id": source_case_id,
            "canonical_case_id": canonical_case_id(source_case_id),
            "in_rruff473": canonical_case_id(source_case_id) in reference_ids,
            "xy_file": str(src["xy_file"]),
            "dif_file": str(src["dif_file"]),
            "xy_path": str(xy_path),
            "dif_path": str(dif_path),
            "xy_exists": xy_path.exists(),
            "dif_exists": dif_path.exists(),
            "space_group_symbol": str(src["space_group_symbol"]).strip(),
            "cell": [float(v) for v in src["cell"]],
            "crystal_system_guess": src.get("crystal_system_guess"),
            "true_bravais_guess": src.get("true_bravais_guess"),
            "n_points": n_points,
            "grid_ok": grid_ok,
            "two_theta_min": theta_min,
            "two_theta_max": theta_max,
            "two_theta_step_mean": step_mean,
            "max_intensity": max_intensity,
            "top_peak_positions": peak_pos,
            "top_peak_values": peak_val,
            "has_anchor_candidates": top is not None,
            "candidate_count": anchor_row["candidate_count"] if anchor_row else 0,
            "top1_icsd_idnum": top["icsd_idnum"] if top else None,
            "top1_icsd_min_name": top["icsd_min_name"] if top else None,
            "top1_icsd_chem_name": top["icsd_chem_name"] if top else None,
            "top1_icsd_sum_form": top["icsd_sum_form"] if top else None,
            "top1_icsd_sgr": top["icsd_sgr"] if top else None,
            "top1_score": top["score"] if top else None,
            "top1_sg_exact": bool(top and top["sg_exact"]),
            "top1_sg_relaxed": bool(top and top["sg_relaxed"]),
            "top1_loose_cell_consistent": loose,
            "top1_length_rel_mean": top["length_rel_mean"] if top else None,
            "top1_angle_abs_mean": top["angle_abs_mean"] if top else None,
            "top1_volume_rel": top["volume_rel"] if top else None,
        }
        rows.append(row)

    summary = {
        "n_rows": len(rows),
        "n_in_rruff473": sum(1 for row in rows if row["in_rruff473"]),
        "n_xy_exists": sum(1 for row in rows if row["xy_exists"]),
        "n_dif_exists": sum(1 for row in rows if row["dif_exists"]),
        "n_grid_ok": sum(1 for row in rows if row["grid_ok"]),
        "n_anchor_candidates": sum(1 for row in rows if row["has_anchor_candidates"]),
        "n_top1_sg_exact": sum(1 for row in rows if row["top1_sg_exact"]),
        "n_top1_sg_relaxed": sum(1 for row in rows if row["top1_sg_relaxed"]),
        "n_top1_loose_cell_consistent": sum(1 for row in rows if row["top1_loose_cell_consistent"]),
        "n_top1_sg_exact_and_loose_cell": sum(1 for row in rows if row["top1_sg_exact"] and row["top1_loose_cell_consistent"]),
    }

    Path(args.out_json).write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    fieldnames = [
        "rid", "name", "source_case_id", "canonical_case_id", "in_rruff473",
        "xy_file", "dif_file", "xy_path", "dif_path", "xy_exists", "dif_exists",
        "space_group_symbol", "crystal_system_guess", "true_bravais_guess",
        "n_points", "grid_ok", "two_theta_min", "two_theta_max", "two_theta_step_mean",
        "max_intensity", "top_peak_positions", "top_peak_values",
        "has_anchor_candidates", "candidate_count", "top1_icsd_idnum", "top1_icsd_min_name",
        "top1_icsd_chem_name", "top1_icsd_sum_form", "top1_icsd_sgr", "top1_score",
        "top1_sg_exact", "top1_sg_relaxed", "top1_loose_cell_consistent",
        "top1_length_rel_mean", "top1_angle_abs_mean", "top1_volume_rel",
    ]
    with open(args.out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["top_peak_positions"] = json.dumps(out["top_peak_positions"])
            out["top_peak_values"] = json.dumps(out["top_peak_values"])
            writer.writerow({k: out.get(k) for k in fieldnames})

    print(json.dumps(summary, indent=2))
    print(args.out_json)
    print(args.out_csv)


if __name__ == "__main__":
    main()
