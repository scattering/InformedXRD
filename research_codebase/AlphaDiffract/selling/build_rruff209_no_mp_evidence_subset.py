#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_TARGET_CATEGORIES = (
    "no_good_mp_family",
    "no_icsd_anchor_no_mp_evidence",
)


def parse_args():
    p = argparse.ArgumentParser(description="Build the 92-case RRUFF209 subset with no current MP evidence.")
    p.add_argument("--leakage-json", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument(
        "--include-category",
        action="append",
        default=[],
        help="Evidence category to include. Repeatable. Defaults to the full 92-case no-MP set.",
    )
    p.add_argument("--subset-name", default="")
    return p.parse_args()


def main():
    args = parse_args()
    payload = json.loads(Path(args.leakage_json).read_text())
    categories = tuple(args.include_category) if args.include_category else DEFAULT_TARGET_CATEGORIES
    rows = [
        {
            "rid": row["rid"],
            "case_id": row["case_id"],
            "name": row["name"],
            "evidence_category": row["evidence_category"],
        }
        for row in payload["rows"]
        if row["evidence_category"] in categories
    ]
    out = {
        "subset_name": args.subset_name or ("rruff209_" + "_".join(categories)),
        "n_cases": len(rows),
        "rows": rows,
        "included_categories": list(categories),
    }
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"n_cases": len(rows), "output_json": str(out_path)}, indent=2))


if __name__ == "__main__":
    main()
