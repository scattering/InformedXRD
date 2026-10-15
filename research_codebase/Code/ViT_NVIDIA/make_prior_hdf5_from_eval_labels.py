#!/usr/bin/env python3
import argparse

import h5py
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Create a prior-only HDF5 by copying y_test to y_train.")
    parser.add_argument("--eval-data-path", required=True)
    parser.add_argument("--output-path", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    with h5py.File(args.eval_data_path, "r") as src:
        y_test = src["y_test"][:]
    y_train = np.asarray(y_test, dtype=np.int64) + 1
    with h5py.File(args.output_path, "w") as dst:
        dst.create_dataset("y_train", data=y_train, compression="gzip")
    print(f"Wrote {args.output_path} with {y_train.shape[0]} labels")


if __name__ == "__main__":
    main()
