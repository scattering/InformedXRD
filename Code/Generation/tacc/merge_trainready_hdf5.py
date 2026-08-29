#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import h5py
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Merge one or more train-ready HDF5 files with X_train/y_train/... layout.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input train-ready HDF5 files.")
    parser.add_argument("--output", required=True, help="Output merged HDF5 path.")
    parser.add_argument("--compression", default="gzip", help="Compression codec for output datasets.")
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=32768,
        help="Number of rows to copy per chunk from each input split to avoid large in-memory reads.",
    )
    return parser.parse_args()


def infer_feature_width(inputs):
    for path in inputs:
        with h5py.File(path, "r") as handle:
            for key in ("X_train", "X_val", "X_test"):
                if key in handle and handle[key].ndim == 2 and handle[key].shape[1] > 0:
                    return int(handle[key].shape[1])
    raise ValueError("Could not infer feature width from inputs")


def ensure_output(handle, feature_width, compression):
    for split in ("train", "val", "test"):
        x_name = f"X_{split}"
        y_name = f"y_{split}"
        if x_name not in handle:
            handle.create_dataset(
                x_name,
                shape=(0, feature_width),
                maxshape=(None, feature_width),
                dtype="float32",
                compression=compression,
            )
        if y_name not in handle:
            handle.create_dataset(
                y_name,
                shape=(0,),
                maxshape=(None,),
                dtype="int16",
                compression=compression,
            )
    str_dtype = h5py.string_dtype(encoding="utf-8")
    if "source_files" not in handle:
        handle.create_dataset("source_files", shape=(0,), maxshape=(None,), dtype=str_dtype)


def append_split(dst_x, dst_y, x, y):
    count = int(x.shape[0])
    if count == 0:
        return
    old = int(dst_x.shape[0])
    dst_x.resize((old + count, dst_x.shape[1]))
    dst_x[old:old + count] = x
    dst_y.resize((old + count,))
    dst_y[old:old + count] = y


def append_split_chunked(dst_x, dst_y, src_x, src_y, chunk_rows: int):
    count = int(src_x.shape[0])
    if count == 0:
        return
    if src_x.ndim != 2 or src_x.shape[1] != dst_x.shape[1]:
        raise ValueError(f"Feature width mismatch while appending chunked data: expected width {dst_x.shape[1]}, got {src_x.shape}")

    old = int(dst_x.shape[0])
    dst_x.resize((old + count, dst_x.shape[1]))
    dst_y.resize((old + count,))

    for start in range(0, count, chunk_rows):
        end = min(start + chunk_rows, count)
        dst_x[old + start:old + end] = src_x[start:end].astype(np.float32, copy=False)
        dst_y[old + start:old + end] = src_y[start:end].astype(np.int16, copy=False)


def append_source_file(handle, source_path):
    ds = handle["source_files"]
    old = int(ds.shape[0])
    ds.resize((old + 1,))
    ds[old] = str(source_path)


def main():
    args = parse_args()
    input_paths = [Path(p).expanduser().resolve() for p in args.inputs]
    for path in input_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_width = infer_feature_width(input_paths)
    temp_output_path = output_path.with_name(f"{output_path.name}.partial.{os.getpid()}")
    if temp_output_path.exists():
        temp_output_path.unlink()

    try:
        with h5py.File(temp_output_path, "w") as out_h5:
            ensure_output(out_h5, feature_width, args.compression)
            out_h5.attrs["feature_width"] = feature_width
            out_h5.attrs["source_format"] = "merged_trainready_hdf5"

            for source_path in input_paths:
                with h5py.File(source_path, "r") as in_h5:
                    for split in ("train", "val", "test"):
                        x_name = f"X_{split}"
                        y_name = f"y_{split}"
                        if x_name not in in_h5 or y_name not in in_h5:
                            continue
                        x = in_h5[x_name]
                        y = in_h5[y_name]
                        if x.ndim != 2 or x.shape[1] != feature_width:
                            raise ValueError(
                                f"Feature width mismatch in {source_path}: expected {feature_width}, got {x.shape}"
                            )
                        append_split_chunked(out_h5[x_name], out_h5[y_name], x, y, args.chunk_rows)
                append_source_file(out_h5, source_path)
                out_h5.flush()
                print(f"[INFO] Merged {source_path}")

            print(
                "[INFO] Merge complete. "
                f"train={out_h5['X_train'].shape[0]:,}, "
                f"val={out_h5['X_val'].shape[0]:,}, "
                f"test={out_h5['X_test'].shape[0]:,}"
            )
        os.replace(temp_output_path, output_path)
        print(f"[INFO] Finalized output atomically at {output_path}")
    except Exception:
        if temp_output_path.exists():
            print(f"[WARN] Leaving partial output for debugging: {temp_output_path}")
        raise


if __name__ == "__main__":
    main()
