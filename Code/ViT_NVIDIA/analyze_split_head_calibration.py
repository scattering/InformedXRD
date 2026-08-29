#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np

from extinction_multilabel import FEATURES, OPERATOR_SLICE


def multiclass_probs(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def multiclass_metrics(logits: np.ndarray, targets: np.ndarray, n_bins: int = 15) -> dict:
    probs = multiclass_probs(logits)
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    correct = (pred == targets).astype(np.float64)
    n = len(targets)

    row_idx = np.arange(n)
    true_probs = probs[row_idx, targets]
    nll = float((-np.log(np.clip(true_probs, 1e-12, 1.0))).mean())
    onehot = np.zeros_like(probs)
    onehot[row_idx, targets] = 1.0
    brier = float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi == 1.0:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        if not mask.any():
            continue
        acc = float(correct[mask].mean())
        c = float(conf[mask].mean())
        w = float(mask.mean())
        ece += w * abs(acc - c)
        bins.append({"lo": float(lo), "hi": float(hi), "n": int(mask.sum()), "acc": acc, "conf": c})
    return {
        "acc_pct": 100.0 * float(correct.mean()),
        "mean_conf_pct": 100.0 * float(conf.mean()),
        "nll_nat": nll,
        "brier": brier,
        "ece": float(ece),
        "bins": bins,
    }


def binary_group_metrics(probs: np.ndarray, targets: np.ndarray, n_bins: int = 15) -> dict:
    pred = (probs >= 0.5).astype(np.float32)
    correct = (pred == targets).astype(np.float64)
    conf = np.where(pred > 0.5, probs, 1.0 - probs)

    nll = float(
        -np.mean(targets * np.log(np.clip(probs, 1e-12, 1.0)) + (1.0 - targets) * np.log(np.clip(1.0 - probs, 1e-12, 1.0)))
    )
    brier = float(np.mean((probs - targets) ** 2))

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi == 1.0:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        if not mask.any():
            continue
        acc = float(correct[mask].mean())
        c = float(conf[mask].mean())
        w = float(mask.mean())
        ece += w * abs(acc - c)
        bins.append({"lo": float(lo), "hi": float(hi), "n": int(mask.sum()), "acc": acc, "conf": c})
    return {
        "acc_pct": 100.0 * float(correct.mean()),
        "mean_conf_pct": 100.0 * float(conf.mean()),
        "positive_rate_pct": 100.0 * float(targets.mean()),
        "nll_nat": nll,
        "brier": brier,
        "ece": float(ece),
        "bins": bins,
    }


def summarize(npz_path: str) -> dict:
    d = np.load(npz_path, allow_pickle=True)
    system_logits = d["system_logits"]
    lattice_logits = d["lattice_logits"]
    operator_logits = d["operator_logits"]
    system_targets = d["system_targets"]
    lattice_targets = d["lattice_targets"]
    operator_targets = d["operator_targets"]

    op_names = FEATURES[OPERATOR_SLICE]
    serial_idx = [i for i, n in enumerate(op_names) if any(tag in n for tag in ("_21", "_31", "_41", "_42", "_61", "_62", "_63"))]
    glide_idx = [i for i, n in enumerate(op_names) if any(tag in n for tag in ("_a", "_b", "_c", "_d", "_n", "_ab"))]

    op_probs = sigmoid(operator_logits)
    per_bit = []
    for i, name in enumerate(op_names):
        m = binary_group_metrics(op_probs[:, i], operator_targets[:, i])
        m["feature"] = name
        per_bit.append(m)

    serial = binary_group_metrics(op_probs[:, serial_idx].reshape(-1), operator_targets[:, serial_idx].reshape(-1))
    glide = binary_group_metrics(op_probs[:, glide_idx].reshape(-1), operator_targets[:, glide_idx].reshape(-1))

    return {
        "npz_path": npz_path,
        "system_head": multiclass_metrics(system_logits, system_targets),
        "lattice_head": multiclass_metrics(lattice_logits, lattice_targets),
        "operator_groups": {
            "serial_screw": serial,
            "glide_like": glide,
        },
        "operator_bits": per_bit,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", nargs="+", required=True)
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()
    out = {Path(p).stem: summarize(p) for p in args.npz}
    text = json.dumps(out, indent=2)
    print(text)
    if args.out_json:
        Path(args.out_json).write_text(text)


if __name__ == "__main__":
    main()
