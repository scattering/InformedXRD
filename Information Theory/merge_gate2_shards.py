#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out-npz", required=True)
    parser.add_argument("--out-json", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    clean_labels = []
    clean_symbols = []
    degraded_labels = []
    degraded_symbols = []
    shard_rows = []

    for path_str in args.inputs:
        path = Path(path_str)
        with np.load(path) as data:
            clean_labels.append(np.asarray(data["clean_labels"], dtype=np.int64))
            clean_symbols.append(np.asarray(data["clean_symbols"], dtype=np.int64))
            degraded_labels.append(np.asarray(data["degraded_labels"], dtype=np.int64))
            degraded_symbols.append(np.asarray(data["degraded_symbols"], dtype=np.int64))
            shard_rows.append(
                {
                    "path": str(path),
                    "clean_rows": int(data["clean_labels"].shape[0]),
                    "degraded_rows": int(data["degraded_labels"].shape[0]),
                }
            )

    merged = {
        "clean_labels": np.concatenate(clean_labels, axis=0),
        "clean_symbols": np.concatenate(clean_symbols, axis=0),
        "degraded_labels": np.concatenate(degraded_labels, axis=0),
        "degraded_symbols": np.concatenate(degraded_symbols, axis=0),
    }
    np.savez_compressed(args.out_npz, **merged)
    summary = {
        "inputs": shard_rows,
        "clean_rows": int(merged["clean_labels"].shape[0]),
        "degraded_rows": int(merged["degraded_labels"].shape[0]),
        "out_npz": str(args.out_npz),
    }
    Path(args.out_json).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
