#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


DEFAULTS = {
    "low_angle_max": 35.0,
    "max_intensity": 1000.0,
    "lattice_threshold": 0.0005,
    "anchor_full_threshold": 0.88,
    "anchor_low_threshold": 0.75,
    "anchor_cell_threshold": 0.0015,
    "partial_tiebreak_margin": 0.08,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run a small one-at-a-time composition-only sensitivity sweep around the "
            "released RRUFF-473 family-coherence thresholds."
        )
    )
    p.add_argument("--reconstruct-script", required=True)
    p.add_argument("--manifest-json", required=True)
    p.add_argument("--xy-dir", required=True)
    p.add_argument("--reference-manifest-json", required=True)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def build_variants() -> list[dict[str, object]]:
    variants: list[dict[str, object]] = []
    variants.append({"name": "baseline", **DEFAULTS})
    axes = {
        "anchor_full_threshold": [0.86, 0.88, 0.90],
        "anchor_low_threshold": [0.70, 0.75, 0.80],
        "anchor_cell_threshold": [0.0010, 0.0015, 0.0020],
        "partial_tiebreak_margin": [0.04, 0.08, 0.12],
        "lattice_threshold": [0.0004, 0.0005, 0.0006],
    }
    for axis, values in axes.items():
        for value in values:
            if value == DEFAULTS[axis]:
                continue
            variant = {"name": f"{axis}_{value:g}", **DEFAULTS}
            variant[axis] = value
            variants.append(variant)
    return variants


def run_variant(
    python_exe: str,
    reconstruct_script: str,
    manifest_json: str,
    xy_dir: str,
    reference_manifest_json: str,
    output_dir: Path,
    variant: dict[str, object],
) -> dict[str, object]:
    out_json = output_dir / f"{variant['name']}.json"
    cmd = [
        python_exe,
        reconstruct_script,
        "--manifest-json",
        manifest_json,
        "--xy-dir",
        xy_dir,
        "--reference-manifest-json",
        reference_manifest_json,
        "--output-json",
        str(out_json),
        "--low-angle-max",
        str(variant["low_angle_max"]),
        "--max-intensity",
        str(variant["max_intensity"]),
        "--lattice-threshold",
        str(variant["lattice_threshold"]),
        "--anchor-full-threshold",
        str(variant["anchor_full_threshold"]),
        "--anchor-low-threshold",
        str(variant["anchor_low_threshold"]),
        "--anchor-cell-threshold",
        str(variant["anchor_cell_threshold"]),
        "--partial-tiebreak-margin",
        str(variant["partial_tiebreak_margin"]),
    ]
    subprocess.run(cmd, check=True)
    data = json.loads(out_json.read_text())
    data["variant"] = variant["name"]
    data.update(
        {
            "lattice_threshold": variant["lattice_threshold"],
            "anchor_full_threshold": variant["anchor_full_threshold"],
            "anchor_low_threshold": variant["anchor_low_threshold"],
            "anchor_cell_threshold": variant["anchor_cell_threshold"],
            "partial_tiebreak_margin": variant["partial_tiebreak_margin"],
        }
    )
    return data


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    variants = build_variants()
    results = []
    for variant in variants:
        results.append(
            run_variant(
                python_exe=sys.executable,
                reconstruct_script=args.reconstruct_script,
                manifest_json=args.manifest_json,
                xy_dir=args.xy_dir,
                reference_manifest_json=args.reference_manifest_json,
                output_dir=output_dir,
                variant=variant,
            )
        )

    combined_json = output_dir / "family_threshold_composition_sweep.json"
    combined_json.write_text(json.dumps(results, indent=2))

    csv_path = output_dir / "family_threshold_composition_sweep.csv"
    fieldnames = [
        "variant",
        "selected_cases",
        "reference_cases",
        "intersection",
        "precision",
        "recall",
        "restricted_upstream_cases",
        "restricted_minerals",
        "singleton_admitted_count",
        "partial_resolution_count",
        "selected_only_count",
        "reference_only_count",
        "lattice_threshold",
        "anchor_full_threshold",
        "anchor_low_threshold",
        "anchor_cell_threshold",
        "partial_tiebreak_margin",
    ]
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(
                {
                    "variant": row["variant"],
                    "selected_cases": row["selected_cases"],
                    "reference_cases": row["reference_cases"],
                    "intersection": row["intersection"],
                    "precision": row["precision"],
                    "recall": row["recall"],
                    "restricted_upstream_cases": row["restricted_upstream_cases"],
                    "restricted_minerals": row["restricted_minerals"],
                    "singleton_admitted_count": len(row.get("singleton_admitted", [])),
                    "partial_resolution_count": len(row.get("partial_resolutions", [])),
                    "selected_only_count": len(row.get("selected_only", [])),
                    "reference_only_count": len(row.get("reference_only", [])),
                    "lattice_threshold": row["lattice_threshold"],
                    "anchor_full_threshold": row["anchor_full_threshold"],
                    "anchor_low_threshold": row["anchor_low_threshold"],
                    "anchor_cell_threshold": row["anchor_cell_threshold"],
                    "partial_tiebreak_margin": row["partial_tiebreak_margin"],
                }
            )

    print(json.dumps({"output_dir": str(output_dir), "n_variants": len(results)}, indent=2))


if __name__ == "__main__":
    main()
