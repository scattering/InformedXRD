"""
paired_nuisance_diff.py
=======================

Paired clean-vs-degraded MI gap estimator. Shared structure indices remove the
dominant structure-sampling variance from tiny nuisance gaps.
"""

from __future__ import annotations

import numpy as np


def _mi(labels, symbols, n_classes, n_symbols, alpha):
    J = np.zeros((n_classes, n_symbols), dtype=np.float64)
    np.add.at(J, (labels, symbols), 1.0)
    J += alpha
    J /= J.sum()
    pg = J.sum(axis=1)
    pm = J.sum(axis=0)
    nz = J > 0
    return float(np.sum(J[nz] * np.log2(J[nz] / (pg[:, None] * pm[None, :])[nz])))


def _null_mean(labels, symbols, n_classes, n_symbols, alpha, *, n_perm, rng):
    y = labels.copy()
    vals = np.empty(n_perm, dtype=np.float64)
    for i in range(n_perm):
        rng.shuffle(y)
        vals[i] = _mi(y, symbols, n_classes, n_symbols, alpha)
    return float(vals.mean())


def paired_mi_diff(
    labels,
    sym_clean,
    sym_degraded,
    n_classes,
    *,
    n_sym_clean=None,
    n_sym_deg=None,
    alpha=None,
    n_perm=200,
    n_boot=1000,
    seed=0,
):
    labels = np.asarray(labels, dtype=np.int64)
    sym_clean = np.asarray(sym_clean, dtype=np.int64)
    sym_degraded = np.asarray(sym_degraded, dtype=np.int64)
    assert len(labels) == len(sym_clean) == len(sym_degraded), "inputs must be paired"

    if n_sym_clean is None:
        n_sym_clean = int(sym_clean.max()) + 1
    if n_sym_deg is None:
        n_sym_deg = int(sym_degraded.max()) + 1
    ac = (1.0 / n_sym_clean) if alpha is None else alpha
    ad = (1.0 / n_sym_deg) if alpha is None else alpha
    rng = np.random.default_rng(seed)

    mi_c = _mi(labels, sym_clean, n_classes, n_sym_clean, ac)
    mi_d = _mi(labels, sym_degraded, n_classes, n_sym_deg, ad)
    null_c = _null_mean(labels, sym_clean, n_classes, n_sym_clean, ac, n_perm=n_perm, rng=rng)
    null_d = _null_mean(labels, sym_degraded, n_classes, n_sym_deg, ad, n_perm=n_perm, rng=rng)

    gaps = np.empty(n_boot, dtype=np.float64)
    n = len(labels)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        gaps[i] = (
            _mi(labels[idx], sym_clean[idx], n_classes, n_sym_clean, ac)
            - _mi(labels[idx], sym_degraded[idx], n_classes, n_sym_deg, ad)
        )
    ci = np.percentile(gaps, [2.5, 97.5])
    return {
        "I_clean_debiased": float(mi_c - null_c),
        "I_degraded_debiased": float(mi_d - null_d),
        "gap_debiased": float((mi_c - null_c) - (mi_d - null_d)),
        "gap_paired_boot_mean": float(gaps.mean()),
        "gap_paired_boot_95": [float(ci[0]), float(ci[1])],
        "gap_excludes_zero": bool(ci[0] > 0 or ci[1] < 0),
    }

