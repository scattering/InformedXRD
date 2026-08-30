#!/usr/bin/env python3
"""Evaluate an XRDBench 230-way SG checkpoint on RRUFF EG labels."""

from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path

import h5py
import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval-h5", required=True)
    parser.add_argument("--mapping-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-npz", default="")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def load_sg_to_eg_map(mapping_csv: str) -> np.ndarray:
    sg_to_eg = np.full(231, -1, dtype=np.int64)
    with open(mapping_csv, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            eg_index = int(row["Index"])
            for sg in ast.literal_eval(row["Space Group Numbers"]):
                sg_to_eg[int(sg)] = eg_index
    if np.any(sg_to_eg[1:] < 0):
        missing = np.where(sg_to_eg[1:] < 0)[0] + 1
        raise ValueError(f"Missing SG->EG mapping for SGs: {missing.tolist()}")
    return sg_to_eg


def collapse_logits_to_eg_probs(logits: np.ndarray, sg_to_eg: np.ndarray) -> np.ndarray:
    sg_probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    eg_probs = np.zeros((sg_probs.shape[0], int(sg_to_eg.max()) + 1), dtype=np.float32)
    for sg in range(1, 231):
        eg_probs[:, sg_to_eg[sg]] += sg_probs[:, sg - 1]
    return eg_probs


def topk_accuracy(probs: np.ndarray, targets: np.ndarray, k: int) -> float:
    k = min(k, probs.shape[1])
    topk = np.argpartition(-probs, kth=k - 1, axis=1)[:, :k]
    return float((topk == targets[:, None]).any(axis=1).mean())


def main():
    args = parse_args()
    sg_to_eg = load_sg_to_eg_map(args.mapping_csv)

    with h5py.File(args.eval_h5, "r") as handle:
        x = np.asarray(handle["X_test"], dtype=np.float32)
        y = np.asarray(handle["y_test"], dtype=np.int64)
        case_ids = [
            item.decode() if isinstance(item, bytes) else str(item)
            for item in handle.get("case_ids", np.array([f"row_{i}" for i in range(len(y))]))[:]
        ]
    if y.min() >= 1:
        y = y - 1

    model = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.eval()
    logits = []
    with torch.no_grad():
        for start in range(0, len(x), args.batch_size):
            batch = torch.from_numpy(x[start : start + args.batch_size]).unsqueeze(1)
            logits.append(model(batch).cpu().numpy())
    logits_arr = np.concatenate(logits, axis=0)
    eg_probs = collapse_logits_to_eg_probs(logits_arr, sg_to_eg)
    top1 = np.argmax(eg_probs, axis=1)
    top5 = np.argsort(-eg_probs, axis=1)[:, :5]

    result = {
        "checkpoint": args.checkpoint,
        "eval_h5": args.eval_h5,
        "n": int(len(y)),
        "top1_eg": topk_accuracy(eg_probs, y, 1),
        "top5_eg": topk_accuracy(eg_probs, y, 5),
        "examples": [
            {
                "case_id": case_ids[i],
                "true_eg": int(y[i]),
                "pred_eg": int(top1[i]),
                "top5_eg": [int(v) for v in top5[i]],
                "correct": bool(top1[i] == y[i]),
            }
            for i in range(len(y))
        ],
    }
    Path(args.output_json).write_text(json.dumps(result, indent=2) + "\n")
    if args.output_npz:
        np.savez_compressed(
            args.output_npz,
            calibrated_probs=eg_probs.astype(np.float32),
            targets=y.astype(np.int64),
            case_ids=np.asarray(case_ids, dtype=object),
            top1_pred=top1.astype(np.int64),
            top5_pred=top5.astype(np.int64),
        )
    print(json.dumps({k: v for k, v in result.items() if k != "examples"}, indent=2))


if __name__ == "__main__":
    main()
