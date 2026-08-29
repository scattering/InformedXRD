#!/usr/bin/env python3
"""Convert XRDBench RRUFF eval JSON into the topology analyzer failure format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-json", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    eval_blob = json.loads(Path(args.eval_json).read_text())

    example_label_base = int(
        eval_blob.get("label_conventions", {}).get("examples_eg_label_base", 0)
    )
    if example_label_base not in (0, 1):
        raise ValueError(f"Unsupported example EG label base: {example_label_base}")
    to_one_based = 1 - example_label_base

    # Topology tools use crystallographic EG numbers 1..99.
    examples = [
        {
            "case_id": row["case_id"],
            "true_eg": int(row["true_eg"]) + to_one_based,
            "pred_eg": int(row["pred_eg"]) + to_one_based,
            "top5_eg": [int(v) + to_one_based for v in row.get("top5_eg", [])],
            "correct": bool(row["correct"]),
        }
        for row in eval_blob["examples"]
    ]
    for row in examples:
        labels = [row["true_eg"], row["pred_eg"], *row["top5_eg"]]
        if any(label < 1 or label > 99 for label in labels):
            raise ValueError(f"EG number outside 1..99 after conversion: {row}")

    failure_blob = {
        "models": {
            args.model_name: {
                "accuracy_top1_pct": 100.0 * float(eval_blob["top1_eg"]),
                "accuracy_top5_pct": 100.0 * float(eval_blob["top5_eg"]),
                "examples": examples,
            }
        }
    }
    Path(args.output_json).write_text(json.dumps(failure_blob, indent=2) + "\n")


if __name__ == "__main__":
    main()
