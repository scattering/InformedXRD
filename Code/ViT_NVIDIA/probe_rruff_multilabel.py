#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter

import h5py
import numpy as np

from extinction_multilabel import FEATURES, build_extinction_templates, ext_group_to_multilabel_target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect an RRUFF-style HDF5 and verify extinction-group to multilabel conversion."
    )
    parser.add_argument("--h5", required=True, help="Path to HDF5 file")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"], help="Dataset split")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10000,
        help="Number of labels to inspect from the requested split",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    templates = build_extinction_templates()

    with h5py.File(args.h5, "r") as handle:
        x_key = f"X_{args.split}"
        y_key = f"y_{args.split}"
        x_ds = handle[x_key]
        y_ds = handle[y_key]

        sample_count = min(args.max_samples, y_ds.shape[0])
        y = np.asarray(y_ds[:sample_count], dtype=np.int64)
        x = np.asarray(x_ds[: min(3, x_ds.shape[0])], dtype=np.float32)

    uniques = np.unique(y)
    missing = [int(label) for label in uniques if int(label) not in templates]
    if missing:
        raise SystemExit(f"Found labels with no multilabel template: {missing}")

    counts = Counter(int(label) for label in y)
    first_label = int(y[0])
    first_target = ext_group_to_multilabel_target(first_label, templates)
    active_features = [FEATURES[idx] for idx, value in enumerate(first_target.tolist()) if value == 1.0]

    print(f"{x_key}: shape={x.shape if x.ndim else 'unknown'} dtype={x.dtype}")
    print(f"{y_key}: sampled={sample_count} unique_labels={len(uniques)} min={int(y.min())} max={int(y.max())}")
    print(f"top_counts={counts.most_common(10)}")
    print(f"first_label={first_label} active_feature_count={int(first_target.sum().item())}")
    print(f"first_active_features={active_features}")
    print(f"first_spectrum_minmax=({float(x[0].min()):.4f}, {float(x[0].max()):.4f})")


if __name__ == "__main__":
    main()
