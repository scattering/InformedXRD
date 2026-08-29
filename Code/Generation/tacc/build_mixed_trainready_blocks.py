#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import h5py
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Build a mixed trainready HDF5 by taking contiguous blocks from standard and PO inputs.")
    p.add_argument("--standard", required=True)
    p.add_argument("--po", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--std-train", type=int, required=True)
    p.add_argument("--po-train", type=int, required=True)
    p.add_argument("--std-val", type=int, required=True)
    p.add_argument("--po-val", type=int, required=True)
    p.add_argument("--std-test", type=int, required=True)
    p.add_argument("--po-test", type=int, required=True)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--compression", default="gzip")
    p.add_argument("--chunk-rows", type=int, default=256)
    return p.parse_args()


def choose_block(n_total: int, n_take: int, rng: np.random.Generator) -> tuple[int, int]:
    if n_take > n_total:
        raise ValueError(f"Requested {n_take} rows from split of size {n_total}")
    if n_take == n_total:
        return 0, n_total
    start = int(rng.integers(0, n_total - n_take + 1))
    return start, start + n_take


def copy_range(src_x, src_y, start: int, end: int, dst_x, dst_y, dst_start: int, chunk_rows: int):
    count = end - start
    for off in range(0, count, chunk_rows):
        src_lo = start + off
        src_hi = min(end, src_lo + chunk_rows)
        dst_lo = dst_start + off
        dst_hi = dst_lo + (src_hi - src_lo)
        dst_x[dst_lo:dst_hi] = src_x[src_lo:src_hi].astype(np.float32, copy=False)
        dst_y[dst_lo:dst_hi] = src_y[src_lo:src_hi].astype(np.int16, copy=False)


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    standard_path = Path(args.standard).expanduser().resolve()
    po_path = Path(args.po).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_output_path = output_path.with_name(f"{output_path.name}.partial.{os.getpid()}")
    if temp_output_path.exists():
        temp_output_path.unlink()

    split_specs = {
        "train": (args.std_train, args.po_train),
        "val": (args.std_val, args.po_val),
        "test": (args.std_test, args.po_test),
    }

    try:
        with h5py.File(standard_path, "r") as std_h5, h5py.File(po_path, "r") as po_h5, h5py.File(temp_output_path, "w") as out_h5:
            feature_width = int(std_h5["X_train"].shape[1])
            if int(po_h5["X_train"].shape[1]) != feature_width:
                raise ValueError("Feature width mismatch between standard and PO datasets")

            for split, (std_n, po_n) in split_specs.items():
                std_x = std_h5[f"X_{split}"]
                std_y = std_h5[f"y_{split}"]
                po_x = po_h5[f"X_{split}"]
                po_y = po_h5[f"y_{split}"]

                std_start, std_end = choose_block(int(std_x.shape[0]), std_n, rng)
                po_start, po_end = choose_block(int(po_x.shape[0]), po_n, rng)
                total_n = std_n + po_n
                x_out = out_h5.create_dataset(
                    f"X_{split}",
                    shape=(total_n, feature_width),
                    dtype=np.float32,
                    compression=args.compression,
                )
                y_out = out_h5.create_dataset(
                    f"y_{split}",
                    shape=(total_n,),
                    dtype=np.int16,
                    compression=args.compression,
                )
                copy_range(std_x, std_y, std_start, std_end, x_out, y_out, 0, args.chunk_rows)
                copy_range(po_x, po_y, po_start, po_end, x_out, y_out, std_n, args.chunk_rows)
                out_h5.flush()

            out_h5.attrs["source_format"] = "mixed_trainready_blocks"
            out_h5.attrs["standard_path"] = str(standard_path)
            out_h5.attrs["po_path"] = str(po_path)
            out_h5.attrs["mix_std_train"] = int(args.std_train)
            out_h5.attrs["mix_po_train"] = int(args.po_train)
            out_h5.attrs["mix_std_val"] = int(args.std_val)
            out_h5.attrs["mix_po_val"] = int(args.po_val)
            out_h5.attrs["mix_std_test"] = int(args.std_test)
            out_h5.attrs["mix_po_test"] = int(args.po_test)
            out_h5.attrs["seed"] = int(args.seed)
            print(
                f"[INFO] mixed train={out_h5['X_train'].shape[0]:,} "
                f"val={out_h5['X_val'].shape[0]:,} test={out_h5['X_test'].shape[0]:,}"
            )

        os.replace(temp_output_path, output_path)
        print(f"[INFO] wrote {output_path}")
    except Exception:
        if temp_output_path.exists():
            print(f"[WARN] leaving partial output for debugging: {temp_output_path}")
        raise


if __name__ == "__main__":
    main()
