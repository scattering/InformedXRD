#!/usr/bin/env python3
"""
Simulator-conditioned Bayes-optimal Top-k ceilings for mask-channel decoding.

This computes the exact ceiling from an estimated channel matrix:

    Acc_Bayes,Top-k = sum_m sum_{g in top-k posterior classes at m} p(g, m)

It also provides an optional held-out estimate for finite-sample pessimism /
optimism checks. The exact-from-channel quantity is the main reported number for
the Gate 2 / Gate 3 simulation artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def fit_channel(labels: np.ndarray, symbols: np.ndarray, n_classes: int, n_symbols: int, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    counts = np.zeros((n_classes, n_symbols), dtype=np.float64)
    np.add.at(counts, (labels, symbols), 1.0)
    counts += alpha
    W = counts / counts.sum(axis=1, keepdims=True)
    prior = np.bincount(labels, minlength=n_classes).astype(np.float64)
    prior /= prior.sum()
    return W, prior


def bayes_topk_exact(W: np.ndarray, prior: np.ndarray, ks: tuple[int, ...] = (1, 3, 5)) -> dict:
    joint = prior[:, None] * W
    by_symbol = joint.T
    pm = by_symbol.sum(axis=1)
    order = np.argsort(-by_symbol, axis=1)

    out = {}
    for k in ks:
        acc = 0.0
        topk = order[:, :k]
        for m in range(by_symbol.shape[0]):
            acc += by_symbol[m, topk[m]].sum()
        out[f"top{k}_bayes_exact"] = float(acc)

    out["top1_exact_from_joint"] = float(np.sum(np.max(by_symbol, axis=1)))

    Hc = 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        post = by_symbol / pm[:, None]
    for m in range(post.shape[0]):
        p = post[m]
        nz = p > 0
        Hc += pm[m] * (-np.sum(p[nz] * np.log2(p[nz])))
    out["H_G_given_Mhat_bits"] = float(Hc)
    out["effective_candidates"] = float(2 ** Hc)
    out["top1_minentropy_lower_bound"] = float(2 ** (-Hc))
    return out


def _fit_posterior(labels_tr: np.ndarray, sym_tr: np.ndarray, n_classes: int, n_symbols: int, alpha: float) -> np.ndarray:
    counts = np.zeros((n_symbols, n_classes), dtype=np.float64)
    np.add.at(counts, (sym_tr, labels_tr), 1.0)
    counts += alpha
    row_sums = counts.sum(axis=1, keepdims=True)
    prior = np.bincount(labels_tr, minlength=n_classes).astype(np.float64)
    if prior.sum() == 0:
        prior = np.full(n_classes, 1.0 / n_classes, dtype=np.float64)
    else:
        prior /= prior.sum()
    post = np.empty_like(counts)
    seen = row_sums[:, 0] > 0
    post[seen] = counts[seen] / row_sums[seen]
    post[~seen] = prior[None, :]
    return post


def bayes_topk_heldout(
    labels: np.ndarray,
    symbols: np.ndarray,
    n_classes: int,
    ks: tuple[int, ...] = (1, 3, 5),
    *,
    alpha: float,
    test_frac: float = 0.5,
    n_boot: int = 300,
    seed: int = 0,
) -> dict:
    labels = np.asarray(labels, dtype=np.int64)
    symbols = np.asarray(symbols, dtype=np.int64)
    rng = np.random.default_rng(seed)
    n = len(labels)
    kmax = max(ks)
    n_symbols = int(symbols.max()) + 1
    acc = {k: [] for k in ks}

    for _ in range(n_boot):
        perm = rng.permutation(n)
        cut = int(n * (1.0 - test_frac))
        tr, te = perm[:cut], perm[cut:]
        post = _fit_posterior(labels[tr], symbols[tr], n_classes, n_symbols, alpha)
        post_te = post[symbols[te]]
        topk_pred = np.argsort(-post_te, axis=1)[:, :kmax]
        truth = labels[te][:, None]
        for k in ks:
            hit = (topk_pred[:, :k] == truth).any(axis=1)
            acc[k].append(float(hit.mean()))

    out = {}
    for k in ks:
        arr = np.asarray(acc[k], dtype=np.float64)
        lo, hi = np.percentile(arr, [2.5, 97.5])
        out[f"top{k}_ceiling"] = float(arr.mean())
        out[f"top{k}_ceiling_95"] = [float(lo), float(hi)]
    return out


def bayes_topk_heldout_with_null(
    labels: np.ndarray,
    symbols: np.ndarray,
    n_classes: int,
    ks: tuple[int, ...] = (1, 3, 5),
    *,
    alpha: float,
    test_frac: float = 0.5,
    n_boot: int = 300,
    seed: int = 0,
) -> dict:
    labels = np.asarray(labels, dtype=np.int64)
    symbols = np.asarray(symbols, dtype=np.int64)
    rng = np.random.default_rng(seed)
    n = len(labels)
    kmax = max(ks)
    n_symbols = int(symbols.max()) + 1
    obs_acc = {k: [] for k in ks}
    null_acc = {k: [] for k in ks}

    for _ in range(n_boot):
        perm = rng.permutation(n)
        cut = int(n * (1.0 - test_frac))
        tr, te = perm[:cut], perm[cut:]

        post = _fit_posterior(labels[tr], symbols[tr], n_classes, n_symbols, alpha)
        post_te = post[symbols[te]]
        topk_pred = np.argsort(-post_te, axis=1)[:, :kmax]
        truth = labels[te][:, None]
        for k in ks:
            hit = (topk_pred[:, :k] == truth).any(axis=1)
            obs_acc[k].append(float(hit.mean()))

        y_shuf = labels.copy()
        rng.shuffle(y_shuf)
        post_null = _fit_posterior(y_shuf[tr], symbols[tr], n_classes, n_symbols, alpha)
        post_null_te = post_null[symbols[te]]
        topk_null = np.argsort(-post_null_te, axis=1)[:, :kmax]
        truth_null = y_shuf[te][:, None]
        for k in ks:
            hit = (topk_null[:, :k] == truth_null).any(axis=1)
            null_acc[k].append(float(hit.mean()))

    out = {}
    for k in ks:
        obs = np.asarray(obs_acc[k], dtype=np.float64)
        null = np.asarray(null_acc[k], dtype=np.float64)
        out[f"top{k}_heldout"] = float(obs.mean())
        out[f"top{k}_heldout_95"] = [float(x) for x in np.percentile(obs, [2.5, 97.5])]
        out[f"top{k}_null_mean"] = float(null.mean())
        out[f"top{k}_null_95"] = [float(x) for x in np.percentile(null, [2.5, 97.5])]
        out[f"top{k}_lift"] = float(obs.mean() - null.mean())
    return out


def parse_args():
    parser = argparse.ArgumentParser(description="Compute Bayes-optimal Top-k mask-channel ceilings from a saved NPZ artifact.")
    parser.add_argument("--npz", required=True)
    parser.add_argument("--labels-key", required=True)
    parser.add_argument("--symbols-key", required=True)
    parser.add_argument("--n-classes", type=int, required=True)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--heldout", action="store_true")
    parser.add_argument("--test-frac", type=float, default=0.5)
    parser.add_argument("--n-boot", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-json", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    data = np.load(args.npz, allow_pickle=True)
    labels = np.asarray(data[args.labels_key], dtype=np.int64)
    symbols = np.asarray(data[args.symbols_key], dtype=np.int64)
    n_symbols = int(symbols.max()) + 1
    alpha = float(args.alpha) if args.alpha > 0 else 1.0 / n_symbols

    W, prior = fit_channel(labels, symbols, args.n_classes, n_symbols, alpha)
    out = {
        "artifact": str(args.npz),
        "labels_key": args.labels_key,
        "symbols_key": args.symbols_key,
        "n_samples": int(len(labels)),
        "n_classes": int(args.n_classes),
        "n_symbols": int(n_symbols),
        "alpha": float(alpha),
    }
    out.update(bayes_topk_exact(W, prior))

    if args.heldout:
        out["heldout"] = bayes_topk_heldout(
            labels,
            symbols,
            args.n_classes,
            alpha=alpha,
            test_frac=args.test_frac,
            n_boot=args.n_boot,
            seed=args.seed,
        )

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
