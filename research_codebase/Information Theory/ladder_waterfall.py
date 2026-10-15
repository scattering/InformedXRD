"""
ladder_waterfall.py
===================

Gate-3 paired waterfall analysis with methodology upgrades:

- per-rung debiased MI
- paired per-rung losses
- optional exact ideal-mask anchor
- per-family marginal loss attribution from encoded mask symbols
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


def _null_mean(labels, symbols, n_classes, n_symbols, alpha, n_perm, rng):
    y = labels.copy()
    vals = np.empty(n_perm, dtype=np.float64)
    for i in range(n_perm):
        rng.shuffle(y)
        vals[i] = _mi(y, symbols, n_classes, n_symbols, alpha)
    return float(vals.mean())


def _bootstrap_debiased_loss(
    labels,
    sym_from,
    sym_to,
    *,
    n_classes,
    n_symbols,
    alpha,
    null_from,
    null_to,
    n_boot,
    rng,
):
    n = len(labels)
    diffs = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        mi_from = _mi(labels[idx], sym_from[idx], n_classes, n_symbols, alpha)
        mi_to = _mi(labels[idx], sym_to[idx], n_classes, n_symbols, alpha)
        diffs[i] = (mi_from - null_from) - (mi_to - null_to)
    return diffs


def _decode_base3(symbols: np.ndarray, n_families: int) -> np.ndarray:
    symbols = np.asarray(symbols, dtype=np.int64)
    out = np.zeros((len(symbols), n_families), dtype=np.int8)
    for i, sym in enumerate(symbols.tolist()):
        x = int(sym)
        for j in range(n_families - 1, -1, -1):
            out[i, j] = x % 3
            x //= 3
    return out


def _encode_base3(digits: np.ndarray) -> np.ndarray:
    digits = np.asarray(digits, dtype=np.int64)
    out = np.zeros(digits.shape[0], dtype=np.int64)
    for j in range(digits.shape[1]):
        out = 3 * out + digits[:, j]
    return out


def ladder_waterfall(
    labels,
    rung_symbols,
    rung_names,
    n_classes,
    *,
    n_families=6,
    alpha=None,
    n_perm=150,
    n_boot=600,
    seed=0,
    ideal_exact_bits=None,
):
    labels = np.asarray(labels, dtype=np.int64)
    syms = [np.asarray(s, dtype=np.int64) for s in rung_symbols]
    assert len(syms) == len(rung_names)
    rng = np.random.default_rng(seed)

    # All rungs live in the same 3^F alphabet even if a given artifact only
    # realizes a smaller subset. Use the fixed alphabet to keep null floors and
    # pairwise losses on comparable footing.
    nsym = [3 ** n_families for _ in syms]
    alphas = [(1.0 / m if alpha is None else alpha) for m in nsym]
    rows = []
    for name, s, m, a in zip(rung_names, syms, nsym, alphas):
        mi = _mi(labels, s, n_classes, m, a)
        null = _null_mean(labels, s, n_classes, m, a, n_perm, rng)
        rows.append({
            "rung": name,
            "n_symbols": int(m),
            "mi_debiased": float(mi - null),
            "null_mean": float(null),
        })

    if ideal_exact_bits is not None and rows:
        rows[0]["mi_exact_bits"] = float(ideal_exact_bits)
        rows[0]["mi_display_bits"] = float(ideal_exact_bits)
    for row in rows:
        row.setdefault("mi_display_bits", row["mi_debiased"])

    losses = []
    for k in range(1, len(syms)):
        diffs = _bootstrap_debiased_loss(
            labels,
            syms[k - 1],
            syms[k],
            n_classes=n_classes,
            n_symbols=nsym[k],
            alpha=alphas[k],
            null_from=rows[k - 1]["null_mean"],
            null_to=rows[k]["null_mean"],
            n_boot=n_boot,
            rng=rng,
        )
        ci = np.percentile(diffs, [2.5, 97.5])
        losses.append({
            "from": rung_names[k - 1],
            "to": rung_names[k],
            "loss_mean": float(diffs.mean()),
            "loss_95": [float(ci[0]), float(ci[1])],
            "significant": bool(ci[0] > 0),
            "dpi_violation_noise": bool(ci[1] < 0),
        })

    family_losses = []
    if len(syms) >= 2:
        clean_digits = _decode_base3(syms[0], n_families=n_families)
        for k in range(1, len(syms)):
            deg_digits = _decode_base3(syms[k], n_families=n_families)
            rung_family = []
            for fam_idx in range(n_families):
                hybrid = deg_digits.copy()
                hybrid[:, fam_idx] = clean_digits[:, fam_idx]
                hybrid_sym = _encode_base3(hybrid)
                m = int(max(hybrid_sym.max(), syms[k].max())) + 1
                a = 1.0 / m if alpha is None else alpha
                mi_hybrid = _mi(labels, hybrid_sym, n_classes, m, a)
                mi_deg = _mi(labels, syms[k], n_classes, nsym[k], alphas[k])
                rung_family.append({
                    "family_index": int(fam_idx),
                    "marginal_loss_bits": float(mi_hybrid - mi_deg),
                })
            family_losses.append({
                "rung": rung_names[k],
                "family_losses": rung_family,
            })

    total_loss = float(rows[0]["mi_display_bits"] - rows[-1]["mi_display_bits"]) if rows else 0.0
    return {
        "rungs": rows,
        "losses": losses,
        "family_losses": family_losses,
        "total_loss": total_loss,
    }
