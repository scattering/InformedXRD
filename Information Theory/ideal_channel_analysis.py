"""
ideal_channel_analysis.py
=========================

Exact, sampling-free analysis of the deterministic ideal-mask channel.

This is the free Gate-2/3 ceiling check from the revised methodology:

- map each EG to its ideal mask symbol
- compute H(M_ideal) exactly under a chosen prior
- identify forced collisions when multiple EGs share the same mask

For a deterministic channel G -> M, I(G; M) = H(M) exactly.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from discrete_channel import encode_mask_vector


def load_mapping_rows(mapping_csv: str | Path) -> list[dict[str, str]]:
    with open(mapping_csv, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    p = p / np.sum(p)
    nz = p > 0
    return float(-np.sum(p[nz] * np.log2(p[nz])))


def ideal_channel_analysis(
    mask_rows: list[list[str] | tuple[str, ...]],
    *,
    prior: np.ndarray | None = None,
) -> dict:
    n_classes = len(mask_rows)
    if prior is None:
        prior = np.full(n_classes, 1.0 / n_classes, dtype=np.float64)
    else:
        prior = np.asarray(prior, dtype=np.float64)
        prior = prior / np.sum(prior)

    symbols = np.asarray([encode_mask_vector(row, n_states=2) for row in mask_rows], dtype=np.int64)
    n_symbols_realized = int(len(np.unique(symbols)))

    pm = np.zeros(int(symbols.max()) + 1, dtype=np.float64)
    np.add.at(pm, symbols, prior)
    H_M = _entropy(pm[pm > 0])

    groups_by_symbol: dict[int, list[int]] = {}
    for eg_idx, sym in enumerate(symbols.tolist()):
        groups_by_symbol.setdefault(int(sym), []).append(int(eg_idx))
    collisions = {sym: eg_list for sym, eg_list in groups_by_symbol.items() if len(eg_list) > 1}

    return {
        "n_classes": int(n_classes),
        "n_distinct_ideal_patterns": int(n_symbols_realized),
        "ideal_mi_bits": float(H_M),
        "ideal_effective_candidates": float(2.0 ** (np.log2(n_classes) - H_M)),
        "forced_collision_count": int(sum(len(v) for v in collisions.values())),
        "collision_bins": int(len(collisions)),
        "collisions": collisions,
        "symbols": symbols,
    }


if __name__ == "__main__":
    # Minimal sanity check: 4 classes, 3 distinct ideal patterns.
    rows = [
        ["present", "absent"],
        ["present", "absent"],
        ["absent", "present"],
        ["absent", "absent"],
    ]
    res = ideal_channel_analysis(rows)
    print(res)
    assert res["n_distinct_ideal_patterns"] == 3
    assert res["collision_bins"] == 1
