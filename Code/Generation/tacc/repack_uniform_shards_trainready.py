#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import h5py
import numpy as np


SHARD_ID_RE = re.compile(r"_shard(\d+)_")


def parse_args():
    parser = argparse.ArgumentParser(description="Repack completed synthetic shard HDF5 files into a train-ready HDF5.")
    parser.add_argument("--input-dir", required=True, help="Directory containing generated shard HDF5 files.")
    parser.add_argument("--output-h5", required=True, help="Output HDF5 path with X_train/y_train/... layout.")
    parser.add_argument("--pattern", default="*.hdf5", help="Glob pattern for shard files.")
    parser.add_argument("--train-percent", type=int, default=90, help="Percent of shard ids assigned to train.")
    parser.add_argument("--val-percent", type=int, default=5, help="Percent of shard ids assigned to val.")
    parser.add_argument("--split-modulus", type=int, default=100, help="Modulo used for stable shard-to-split assignment.")
    parser.add_argument("--compression", default="gzip", help="Compression codec for output datasets.")
    parser.add_argument("--max-shards", type=int, default=None, help="Optional cap on number of complete shards to ingest.")
    parser.add_argument("--allow-partial", action="store_true", help="Allow ingestion of partial shards using successes_written rows.")
    parser.add_argument("--min-written", type=int, default=1, help="Minimum written rows required to ingest a shard.")
    parser.add_argument(
        "--split-mode",
        choices=("modulo", "order"),
        default="modulo",
        help="How to assign shards to train/val/test. 'modulo' uses shard_id %% split_modulus, 'order' uses the shard's order among eligible inputs.",
    )
    return parser.parse_args()


def extract_shard_id(path: Path) -> int:
    match = SHARD_ID_RE.search(path.name)
    if match is None:
        raise ValueError(f"Could not parse shard id from filename: {path.name}")
    return int(match.group(1))


def stable_split_name(shard_id: int, train_percent: int, val_percent: int, split_modulus: int) -> str:
    if train_percent <= 0 or val_percent < 0 or (train_percent + val_percent) >= split_modulus:
        raise ValueError("Split percentages must satisfy 0 < train, 0 <= val, and train + val < split_modulus")
    bucket = shard_id % split_modulus
    if bucket < train_percent:
        return "train"
    if bucket < train_percent + val_percent:
        return "val"
    return "test"


def ordered_split_name(order_index: int, total_count: int, train_percent: int, val_percent: int) -> str:
    if total_count <= 0:
        raise ValueError("total_count must be positive for ordered split assignment")
    if train_percent <= 0 or val_percent < 0 or (train_percent + val_percent) >= 100:
        raise ValueError("Ordered split percentages must satisfy 0 < train, 0 <= val, and train + val < 100")

    bucket = int((order_index * 100) / total_count)
    if bucket < train_percent:
        return "train"
    if bucket < train_percent + val_percent:
        return "val"
    return "test"


def is_complete_shard(handle: h5py.File) -> bool:
    root = handle["/"]
    expected = int(root.attrs.get("local_samples_target", 0))
    written = int(root.attrs.get("successes_written", 0))
    if expected <= 0 or "intensity" not in handle or "extinction_group" not in handle:
        return False
    return written >= expected and handle["intensity"].shape[0] >= expected


def get_rows_to_ingest(handle: h5py.File, allow_partial: bool, min_written: int) -> int:
    if "intensity" not in handle or "extinction_group" not in handle:
        return 0
    root = handle["/"]
    written = int(root.attrs.get("successes_written", 0))
    available = int(handle["intensity"].shape[0])
    usable = min(written, available)
    if usable < min_written:
        return 0
    if allow_partial:
        return usable
    expected = int(root.attrs.get("local_samples_target", 0))
    if expected > 0 and usable >= expected:
        return expected
    return 0


def open_or_init_output(output_h5: Path, compression: str):
    handle = h5py.File(output_h5, "a")
    if "processed_paths" in handle:
        return handle

    feature_width = None
    if "2theta" in handle:
        feature_width = handle["2theta"].shape[0]

    str_dtype = h5py.string_dtype(encoding="utf-8")
    handle.create_dataset("processed_paths", shape=(0,), maxshape=(None,), dtype=str_dtype)
    handle.create_dataset("processed_shard_ids", shape=(0,), maxshape=(None,), dtype="int32")
    handle.create_dataset("processed_splits", shape=(0,), maxshape=(None,), dtype=str_dtype)

    if feature_width is None:
        handle.attrs["feature_width"] = -1
    else:
        handle.attrs["feature_width"] = int(feature_width)

    for split in ("train", "val", "test"):
        x_name = f"X_{split}"
        y_name = f"y_{split}"
        if x_name not in handle:
            shape = (0, 0) if feature_width is None else (0, feature_width)
            maxshape = (None, 0) if feature_width is None else (None, feature_width)
            handle.create_dataset(x_name, shape=shape, maxshape=maxshape, dtype="float32", compression=compression)
        if y_name not in handle:
            handle.create_dataset(y_name, shape=(0,), maxshape=(None,), dtype="int16", compression=compression)
    handle.attrs["source_format"] = "uniform_synthetic_shards"
    handle.flush()
    return handle


def ensure_feature_width(handle: h5py.File, width: int, compression: str):
    current = int(handle.attrs.get("feature_width", -1))
    if current in (-1, 0):
        for split in ("train", "val", "test"):
            x_name = f"X_{split}"
            if x_name in handle:
                del handle[x_name]
            handle.create_dataset(
                x_name,
                shape=(0, width),
                maxshape=(None, width),
                dtype="float32",
                compression=compression,
            )
        handle.attrs["feature_width"] = int(width)
        return
    if current != width:
        raise ValueError(f"Feature width mismatch: existing output width {current}, incoming shard width {width}")


def append_split(handle: h5py.File, split: str, x: np.ndarray, y: np.ndarray):
    x_name = f"X_{split}"
    y_name = f"y_{split}"
    ds_x = handle[x_name]
    ds_y = handle[y_name]
    n_old = ds_x.shape[0]
    count = x.shape[0]
    ds_x.resize((n_old + count, ds_x.shape[1]))
    ds_x[n_old:n_old + count] = x
    ds_y.resize((n_old + count,))
    ds_y[n_old:n_old + count] = y


def append_processed_metadata(handle: h5py.File, shard_path: Path, shard_id: int, split: str):
    for name, value in (
        ("processed_paths", str(shard_path)),
        ("processed_shard_ids", int(shard_id)),
        ("processed_splits", split),
    ):
        ds = handle[name]
        n_old = ds.shape[0]
        ds.resize((n_old + 1,))
        ds[n_old] = value


def main():
    args = parse_args()
    if args.train_percent + args.val_percent >= args.split_modulus:
        raise ValueError("train-percent + val-percent must be smaller than split-modulus")

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_h5 = Path(args.output_h5).expanduser().resolve()
    output_h5.parent.mkdir(parents=True, exist_ok=True)

    shard_paths = sorted(input_dir.glob(args.pattern), key=extract_shard_id)
    if not shard_paths:
        raise FileNotFoundError(f"No shard files matched {args.pattern} in {input_dir}")

    with open_or_init_output(output_h5, args.compression) as out_h5:
        processed_paths = set(str(value) for value in out_h5["processed_paths"].asstr()[:].tolist())
        eligible_paths = []
        for shard_path in shard_paths:
            if str(shard_path) in processed_paths:
                continue
            try:
                with h5py.File(shard_path, "r") as shard_h5:
                    rows_to_ingest = get_rows_to_ingest(shard_h5, args.allow_partial, args.min_written)
                    if rows_to_ingest > 0:
                        eligible_paths.append((shard_path, rows_to_ingest))
            except BlockingIOError:
                print(f"[WARN] Skipping locked shard during eligibility scan: {shard_path.name}")
                continue

        if args.max_shards is not None:
            eligible_paths = eligible_paths[: args.max_shards]

        total_eligible = len(eligible_paths)
        eligible_count = 0
        ingested_count = 0

        for shard_path, rows_to_ingest in eligible_paths:
            try:
                with h5py.File(shard_path, "r") as shard_h5:
                    eligible_count += 1

                    x = shard_h5["intensity"][:rows_to_ingest].astype(np.float32, copy=False)
                    y = shard_h5["extinction_group"][:rows_to_ingest].astype(np.int16, copy=False)
                    width = int(x.shape[1])
                    ensure_feature_width(out_h5, width, args.compression)

                    shard_id = extract_shard_id(shard_path)
                    if args.split_mode == "order":
                        split = ordered_split_name(eligible_count - 1, total_eligible, args.train_percent, args.val_percent)
                    else:
                        split = stable_split_name(shard_id, args.train_percent, args.val_percent, args.split_modulus)
                    append_split(out_h5, split, x, y)
                    append_processed_metadata(out_h5, shard_path, shard_id, split)
                    out_h5.flush()
                    ingested_count += 1
                    status = "partial" if args.allow_partial and not is_complete_shard(shard_h5) else "complete"
                    print(f"[INFO] Ingested {status} shard {shard_id:04d} -> {split}: {x.shape[0]:,} samples from {shard_path.name}")
            except BlockingIOError:
                print(f"[WARN] Skipping locked shard during ingest: {shard_path.name}")
                continue

        train_n = int(out_h5["X_train"].shape[0])
        val_n = int(out_h5["X_val"].shape[0])
        test_n = int(out_h5["X_test"].shape[0])
        print(
            f"[INFO] Repack complete. New shards ingested={ingested_count}, "
            f"totals train={train_n:,}, val={val_n:,}, test={test_n:,}"
        )


if __name__ == "__main__":
    main()
