"""
gate2_diagnose.py
=================

Gate-2 estimator diagnostics with the revised methodology:

- debiased MI vs permutation null
- N-sweep from one generated artifact
- collapsed-alphabet companion (unresolved -> present/absent)
- shuffled-capacity null
"""

from __future__ import annotations

import numpy as np

from discrete_channel import blahut_arimoto_capacity


def _entropy_from_counts(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=np.float64)
    counts = counts / np.sum(counts)
    nz = counts > 0
    return float(-np.sum(counts[nz] * np.log2(counts[nz])))


def mi_plugin(labels, symbols, n_classes, n_symbols, alpha=0.0):
    J = np.zeros((n_classes, n_symbols), dtype=np.float64)
    np.add.at(J, (labels, symbols), 1.0)
    J += alpha
    J /= J.sum()
    pg = J.sum(axis=1)
    pm = J.sum(axis=0)
    nz = J > 0
    return float(np.sum(J[nz] * np.log2(J[nz] / (pg[:, None] * pm[None, :])[nz])))


def estimate_channel(labels, symbols, n_classes, n_symbols, alpha):
    W = np.zeros((n_classes, n_symbols), dtype=np.float64)
    np.add.at(W, (labels, symbols), 1.0)
    W += alpha
    W /= W.sum(axis=1, keepdims=True)
    return W


def mi_shuffle_null(labels, symbols, n_classes, n_symbols, alpha, *, n_perm=500, seed=0):
    rng = np.random.default_rng(seed)
    y = labels.copy()
    null = np.empty(n_perm, dtype=np.float64)
    for i in range(n_perm):
        rng.shuffle(y)
        null[i] = mi_plugin(y, symbols, n_classes, n_symbols, alpha)
    return {
        "null_mean": float(null.mean()),
        "null_ci": np.percentile(null, [2.5, 97.5]).tolist(),
        "samples": null,
    }


def capacity_shuffle_null(labels, symbols, n_classes, n_symbols, alpha, *, n_perm=120, seed=0):
    rng = np.random.default_rng(seed)
    y = labels.copy()
    vals = np.empty(n_perm, dtype=np.float64)
    for i in range(n_perm):
        rng.shuffle(y)
        W = estimate_channel(y, symbols, n_classes, n_symbols, alpha)
        vals[i], _ = blahut_arimoto_capacity(W)
    return {
        "null_mean": float(vals.mean()),
        "null_ci": np.percentile(vals, [2.5, 97.5]).tolist(),
        "samples": vals,
    }


def collapse_unresolved_to_binary(symbols: np.ndarray, n_families: int) -> tuple[np.ndarray, int]:
    symbols = np.asarray(symbols, dtype=np.int64)
    out = np.empty_like(symbols)
    for i, sym in enumerate(symbols.tolist()):
        digits = []
        x = int(sym)
        for _ in range(n_families):
            digits.append(x % 3)
            x //= 3
        digits.reverse()
        bits = [0 if d == 1 else 1 for d in digits]  # absent->0, present/unresolved->1
        val = 0
        for b in bits:
            val = 2 * val + b
        out[i] = val
    return out, 2 ** n_families


def diagnose(
    labels,
    symbols,
    *,
    n_classes=99,
    n_symbols=None,
    n_families=6,
    alpha_current=0.5,
    alpha_sane=None,
    n_perm=500,
    capacity_perm=120,
    seed=0,
):
    labels = np.asarray(labels, dtype=np.int64)
    symbols = np.asarray(symbols, dtype=np.int64)
    if n_symbols is None:
        n_symbols = int(symbols.max()) + 1
    if alpha_sane is None:
        alpha_sane = 1.0 / n_symbols

    H_G = _entropy_from_counts(np.bincount(labels, minlength=n_classes))
    H_M = _entropy_from_counts(np.bincount(symbols, minlength=n_symbols))
    n_distinct = int(len(np.unique(symbols)))

    mi_cur = mi_plugin(labels, symbols, n_classes, n_symbols, alpha_current)
    mi_sane = mi_plugin(labels, symbols, n_classes, n_symbols, alpha_sane)
    mi_raw = mi_plugin(labels, symbols, n_classes, n_symbols, 0.0)
    null = mi_shuffle_null(labels, symbols, n_classes, n_symbols, alpha_sane, n_perm=n_perm, seed=seed)
    debiased = float(mi_sane - null["null_mean"])

    fracs = [0.25, 0.5, 1.0]
    rng = np.random.default_rng(seed + 1)
    n_sweep = []
    for frac in fracs:
        m = max(2, int(len(labels) * frac))
        idx = rng.choice(len(labels), m, replace=False)
        v = mi_plugin(labels[idx], symbols[idx], n_classes, n_symbols, alpha_sane)
        n_sweep.append((int(m), float(v)))

    W = estimate_channel(labels, symbols, n_classes, n_symbols, alpha_sane)
    capacity_bits, pi_star = blahut_arimoto_capacity(W)
    capacity_null = capacity_shuffle_null(
        labels, symbols, n_classes, n_symbols, alpha_sane, n_perm=capacity_perm, seed=seed + 2
    )

    collapsed_symbols, collapsed_n_symbols = collapse_unresolved_to_binary(symbols, n_families=n_families)
    collapsed_alpha = 1.0 / collapsed_n_symbols
    collapsed_null = mi_shuffle_null(
        labels, collapsed_symbols, n_classes, collapsed_n_symbols, collapsed_alpha, n_perm=n_perm, seed=seed + 3
    )
    collapsed_mi = mi_plugin(labels, collapsed_symbols, n_classes, collapsed_n_symbols, collapsed_alpha)
    collapsed_debiased = float(collapsed_mi - collapsed_null["null_mean"])

    ratio = float(null["null_mean"] / max(mi_sane, 1e-12))
    rising = n_sweep[-1][1] > 1.15 * max(n_sweep[0][1], 1e-4)
    clears_null = mi_sane > null["null_ci"][1]
    trustworthy = clears_null and ratio < 0.25
    if trustworthy and not rising:
        verdict = "CONVERGED / TRUSTWORTHY"
    elif clears_null:
        verdict = "A: ESTIMATION-STARVED"
    elif H_M < 0.15 * H_G:
        verdict = "B1: NON-DIAGNOSTIC MASK (near-constant)"
    else:
        verdict = "B2: MASK VARIES BUT NOT WITH G"

    return {
        "H_G": float(H_G),
        "H_M": float(H_M),
        "n_distinct": int(n_distinct),
        "mi_current": float(mi_cur),
        "mi_sane": float(mi_sane),
        "mi_raw": float(mi_raw),
        "mi_debiased": debiased,
        "null_mean": float(null["null_mean"]),
        "null_ci": list(null["null_ci"]),
        "null_over_signal": ratio,
        "n_sweep": n_sweep,
        "capacity_bits": float(capacity_bits),
        "capacity_achieving_prior": pi_star.tolist(),
        "capacity_null_mean": float(capacity_null["null_mean"]),
        "capacity_null_ci": list(capacity_null["null_ci"]),
        "collapsed": {
            "n_symbols": int(collapsed_n_symbols),
            "mi_sane": float(collapsed_mi),
            "mi_debiased": float(collapsed_debiased),
            "null_mean": float(collapsed_null["null_mean"]),
            "null_ci": list(collapsed_null["null_ci"]),
        },
        "verdict": verdict,
    }

