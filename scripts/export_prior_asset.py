#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Export compact extinction-group prior assets from a trainready HDF5.")
    parser.add_argument("--prior-h5", required=True, help="HDF5 path containing y_train.")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--smoothing", type=float, default=1.0)
    parser.add_argument("--num-classes", type=int, default=99)
    parser.add_argument("--probability-decimals", type=int, default=8)
    parser.add_argument("--log-prior-decimals", type=int, default=6)
    return parser.parse_args()


def main():
    args = parse_args()
    with h5py.File(args.prior_h5, "r") as handle:
        if "y_train" not in handle:
            raise KeyError(f"{args.prior_h5} does not contain y_train")
        y_train = handle["y_train"][:]

    counts = {ext: float(args.smoothing) for ext in range(1, args.num_classes + 1)}
    for raw in y_train.tolist():
        counts[int(raw)] = counts.get(int(raw), float(args.smoothing)) + 1.0

    total = float(sum(counts.values()))
    rows = []
    for ext in range(1, args.num_classes + 1):
        prob = counts[ext] / total
        rows.append(
            {
                "ext_group": ext,
                "probability": round(prob, args.probability_decimals),
                "log_prior": round(float(np.log(prob)), args.log_prior_decimals),
                "seen_in_source_corpus": counts[ext] > float(args.smoothing),
            }
        )

    df = pd.DataFrame(rows)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    Path(args.output_json).write_text(
        json.dumps(
            {
                "source_artifact": "aggregate_extinction_group_prior",
                "provenance_note": "Aggregate extinction-group prior used for calibration; released as rounded probabilities/log-priors with no raw source records.",
                "smoothing": args.smoothing,
                "num_classes": args.num_classes,
                "probability_decimals": args.probability_decimals,
                "log_prior_decimals": args.log_prior_decimals,
                "rows": rows,
            },
            indent=2,
        )
    )
    print(f"Wrote {args.output_csv} and {args.output_json}")


if __name__ == "__main__":
    main()
