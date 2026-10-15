#!/usr/bin/env python3
"""Project RRUFF-473 cases onto the existing RRUFF->ICSD anchor surface.

This does not reconstruct AlphaDiffract's unpublished 734/240 manifest.
It answers a narrower question: how many of our frozen RRUFF-473 benchmark
cases are structurally anchorable in the same ICSD-linked sense that their
curated subset requires?
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_json(path: Path):
    with path.open() as fh:
        return json.load(fh)


def derive_rid(case_id: str) -> str | None:
    parts = case_id.split("__")
    if len(parts) < 3:
        return None
    return parts[1]


def summarize(rows: list[dict]) -> dict:
    matched_by_rid = sum(1 for row in rows if row["rid_match"])
    matched_with_candidates = sum(1 for row in rows if row["has_anchor_candidates"])
    sg_exact = sum(1 for row in rows if row["top1_sg_exact"])
    sg_relaxed = sum(1 for row in rows if row["top1_sg_relaxed"])
    loose_cell = sum(1 for row in rows if row["top1_loose_cell_consistent"])
    sg_exact_loose = sum(1 for row in rows if row["top1_sg_exact"] and row["top1_loose_cell_consistent"])
    sg_relaxed_loose = sum(1 for row in rows if row["top1_sg_relaxed"] and row["top1_loose_cell_consistent"])
    return {
        "n_rruff473_cases": len(rows),
        "n_case_ids_with_parseable_rid": sum(1 for row in rows if row["rid"] is not None),
        "n_rid_match_in_rruff_manifest": matched_by_rid,
        "n_cases_with_anchor_candidates": matched_with_candidates,
        "n_top1_sg_exact": sg_exact,
        "n_top1_sg_relaxed": sg_relaxed,
        "n_top1_loose_cell_consistent": loose_cell,
        "n_top1_sg_exact_and_loose_cell": sg_exact_loose,
        "n_top1_sg_relaxed_and_loose_cell": sg_relaxed_loose,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rruff473-json", required=True, type=Path)
    parser.add_argument("--rruff-icsd-json", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    args = parser.parse_args()

    rruff473 = load_json(args.rruff473_json)
    rruff_icsd = load_json(args.rruff_icsd_json)

    by_rid = {row["rid"]: row for row in rruff_icsd["results"]}
    out_rows = []
    for row in rruff473["rows"]:
        case_id = row["case_id"]
        rid = derive_rid(case_id)
        manifest_row = by_rid.get(rid)
        top = manifest_row["top_matches"][0] if manifest_row and manifest_row["top_matches"] else None
        loose = bool(top and top["length_rel_mean"] <= 0.05 and top["volume_rel"] <= 0.15)
        out_rows.append(
            {
                "case_id": case_id,
                "rid": rid,
                "mineral": row["mineral"],
                "eg_label": row["y"],
                "rid_match": manifest_row is not None,
                "has_anchor_candidates": top is not None,
                "candidate_count": manifest_row["candidate_count"] if manifest_row else 0,
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
        )

    summary = summarize(out_rows)
    args.out_json.write_text(json.dumps({"summary": summary, "rows": out_rows}, indent=2))

    fieldnames = list(out_rows[0].keys()) if out_rows else []
    with args.out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print(json.dumps(summary, indent=2))
    print(args.out_json)
    print(args.out_csv)


if __name__ == "__main__":
    main()
