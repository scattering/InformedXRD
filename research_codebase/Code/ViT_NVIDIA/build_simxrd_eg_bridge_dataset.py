#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import json
import sqlite3
from pathlib import Path

import h5py
import numpy as np
from ase.db import connect


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a SimXRD ASE SQLite split into an EG-labeled trainready HDF5.")
    parser.add_argument("--sqlite-db", required=True)
    parser.add_argument("--final-table-path", required=True)
    parser.add_argument("--output-h5", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--max-samples", type=int, default=40000)
    parser.add_argument("--max-record-id", type=int, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--two-theta-min", type=float, default=5.0)
    parser.add_argument("--two-theta-max", type=float, default=90.0)
    parser.add_argument("--step-size", type=float, default=0.01)
    parser.add_argument("--wavelength", type=float, default=1.5406)
    return parser.parse_args()


def load_sg_to_eg_map(final_table_path: str) -> dict[int, int]:
    sg_to_eg: dict[int, int] = {}
    with open(final_table_path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sg_to_eg[int(row["Space Group"])] = int(row["Extinction Group"])
    if len(sg_to_eg) != 230:
        raise ValueError(f"Expected 230 SG->EG mappings, found {len(sg_to_eg)}")
    return sg_to_eg


def main() -> None:
    args = parse_args()
    sg_to_eg = load_sg_to_eg_map(args.final_table_path)

    n_points = int(round((args.two_theta_max - args.two_theta_min) / args.step_size)) + 1
    target_grid = np.linspace(args.two_theta_min, args.two_theta_max, n_points, dtype=np.float32)
    sqlite_path = Path(args.sqlite_db)

    con = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    cur = con.cursor()
    db = connect(str(sqlite_path))
    total_records = int(cur.execute("SELECT COUNT(*) FROM systems").fetchone()[0])
    max_record_id = int(args.max_record_id) if args.max_record_id is not None else total_records

    valid = []
    skipped = {"missing_fields": 0, "bad_tager": 0, "bad_curve": 0, "bad_bragg": 0, "sg_not_in_table": 0}
    for rec_id in range(1, max_record_id + 1):
        if args.max_samples and len(valid) >= args.max_samples:
            break
        try:
            row = db.get(id=rec_id)
            tager = ast.literal_eval(getattr(row, "tager"))
            d_vals = np.asarray(ast.literal_eval(getattr(row, "latt_dis")), dtype=np.float32)
            i_vals = np.asarray(ast.literal_eval(getattr(row, "intensity")), dtype=np.float32)
            chem_form = getattr(row, "chem_form", "")
            sim_param = getattr(row, "simulation_param", "")
        except Exception:
            skipped["bad_curve"] += 1
            continue
        if not isinstance(tager, (list, tuple)) or len(tager) < 1:
            skipped["bad_tager"] += 1
            continue
        sg = int(tager[0])
        eg = sg_to_eg.get(sg)
        if eg is None:
            skipped["sg_not_in_table"] += 1
            continue
        if d_vals.ndim != 1 or i_vals.ndim != 1 or d_vals.size != i_vals.size or d_vals.size == 0:
            skipped["bad_curve"] += 1
            continue
        arg = args.wavelength / (2.0 * d_vals.astype(np.float64))
        if np.any(arg <= 0) or np.any(arg > 1):
            skipped["bad_bragg"] += 1
            continue
        two_theta = 2.0 * np.degrees(np.arcsin(arg))
        order = np.argsort(two_theta)
        two_theta = two_theta[order]
        i_vals = i_vals[order]
        interp = np.interp(target_grid, two_theta, i_vals, left=0.0, right=0.0).astype(np.float32)
        max_val = float(interp.max())
        if max_val > 0:
            interp /= max_val
        valid.append((rec_id, interp, eg, sg, chem_form, sim_param))

    rng = np.random.default_rng(args.seed)
    order = np.arange(len(valid))
    rng.shuffle(order)
    selected = [valid[i] for i in order]

    n_total = len(selected)
    n_train = int(round(n_total * args.train_frac))
    n_val = int(round(n_total * args.val_frac))
    n_test = n_total - n_train - n_val
    train = selected[:n_train]
    val = selected[n_train:n_train + n_val]
    test = selected[n_train + n_val:]

    def pack(rows_subset):
        X = np.stack([row[1] for row in rows_subset], axis=0).astype(np.float32)
        y = np.asarray([row[2] for row in rows_subset], dtype=np.int64)
        sg = np.asarray([row[3] for row in rows_subset], dtype=np.int64)
        rec_id = np.asarray([str(row[0]) for row in rows_subset], dtype=h5py.string_dtype("utf-8"))
        chem = np.asarray([row[4] for row in rows_subset], dtype=h5py.string_dtype("utf-8"))
        sim = np.asarray([row[5] for row in rows_subset], dtype=h5py.string_dtype("utf-8"))
        return X, y, sg, rec_id, chem, sim

    output_h5 = Path(args.output_h5)
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_h5, "w") as handle:
        for split_name, subset in [("train", train), ("val", val), ("test", test)]:
            X, y, sg, rec_id, chem, sim = pack(subset)
            handle.create_dataset(f"X_{split_name}", data=X, compression="gzip")
            handle.create_dataset(f"y_{split_name}", data=y)
            handle.create_dataset(f"sg_{split_name}", data=sg)
            handle.create_dataset(f"record_id_{split_name}", data=rec_id)
            handle.create_dataset(f"chem_form_{split_name}", data=chem)
            handle.create_dataset(f"simulation_param_{split_name}", data=sim)
        handle.create_dataset("two_theta", data=target_grid)

    summary = {
        "sqlite_db": str(sqlite_path),
        "output_h5": str(output_h5),
        "max_samples": args.max_samples,
        "max_record_id": args.max_record_id,
        "seed": args.seed,
        "n_records_total": total_records,
        "n_rows_fetched": len(valid),
        "n_valid_before_sampling": len(valid),
        "n_selected": n_total,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "skipped": skipped,
        "spec_length": int(target_grid.shape[0]),
        "two_theta_min": float(target_grid.min()),
        "two_theta_max": float(target_grid.max()),
        "unique_ext_groups": int(len({row[2] for row in selected})),
        "sample_record_ids": [str(row[0]) for row in selected[:10]],
    }
    Path(args.output_json).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
