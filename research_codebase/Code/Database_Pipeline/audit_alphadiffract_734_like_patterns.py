#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit reconstructed AlphaDiffract-like clusters against raw XY patterns.")
    p.add_argument("--search-json", required=True)
    p.add_argument("--cell-threshold", type=float, default=0.0061)
    p.add_argument("--out-json", required=True)
    return p.parse_args()


class DSU:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def cell_distance(a: list[float], b: list[float]) -> float:
    acc = 0.0
    n = 0
    for x, y in zip(a, b):
        scale = max(abs(x), abs(y), 1.0)
        acc += ((x - y) / scale) ** 2
        n += 1
    return math.sqrt(acc / n) if n else 0.0


def parse_xy(path: Path) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("##"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 2:
                continue
            try:
                x = float(parts[0])
                y = float(parts[1])
            except ValueError:
                continue
            xs.append(x)
            ys.append(y)
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def normalize(y: np.ndarray) -> np.ndarray:
    ymax = float(np.max(np.abs(y))) if y.size else 0.0
    if ymax <= 0:
        return np.zeros_like(y)
    return y / ymax


def corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def main() -> None:
    args = parse_args()
    payload = json.loads(Path(args.search_json).read_text())
    rows = [
        row for row in payload["rows"]
        if row["grid_ok"] and row["top1_sg_exact"] and row["top1_loose_cell_consistent"]
    ]

    by_group: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        by_group[(row["name"], row["space_group_symbol"])].append(row)

    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def load_norm_pattern(path_str: str) -> tuple[np.ndarray, np.ndarray]:
        if path_str not in cache:
            xs, ys = parse_xy(Path(path_str))
            cache[path_str] = xs, normalize(ys)
        return cache[path_str]

    cluster_rows = []
    all_full = []
    all_low = []
    low_mask_cache: dict[str, np.ndarray] = {}

    for (name, sg), group in by_group.items():
        dsu = DSU(len(group))
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                if cell_distance(group[i]["cell"], group[j]["cell"]) <= args.cell_threshold:
                    dsu.union(i, j)
        comps: dict[int, list[dict]] = defaultdict(list)
        for i, row in enumerate(group):
            comps[dsu.find(i)].append(row)

        for comp in comps.values():
            full_corrs = []
            low_corrs = []
            for i in range(len(comp)):
                xs_i, y_i = load_norm_pattern(comp[i]["xy_path"])
                key_i = comp[i]["xy_path"]
                if key_i not in low_mask_cache:
                    low_mask_cache[key_i] = xs_i <= 35.0
                low_i = y_i[low_mask_cache[key_i]]
                for j in range(i + 1, len(comp)):
                    xs_j, y_j = load_norm_pattern(comp[j]["xy_path"])
                    if xs_i.shape != xs_j.shape or not np.allclose(xs_i, xs_j, atol=1e-6):
                        continue
                    key_j = comp[j]["xy_path"]
                    if key_j not in low_mask_cache:
                        low_mask_cache[key_j] = xs_j <= 35.0
                    low_j = y_j[low_mask_cache[key_j]]
                    full_corrs.append(corr(y_i, y_j))
                    low_corrs.append(corr(low_i, low_j))
            cluster_rows.append(
                {
                    "name": name,
                    "space_group_symbol": sg,
                    "cluster_size": len(comp),
                    "members": [row["source_case_id"] for row in comp],
                    "full_corr_min": min(full_corrs) if full_corrs else None,
                    "full_corr_mean": float(np.mean(full_corrs)) if full_corrs else None,
                    "low_corr_min": min(low_corrs) if low_corrs else None,
                    "low_corr_mean": float(np.mean(low_corrs)) if low_corrs else None,
                }
            )
            all_full.extend(full_corrs)
            all_low.extend(low_corrs)

    summary = {
        "cell_threshold": args.cell_threshold,
        "n_eligible_rows": len(rows),
        "n_clusters": len(cluster_rows),
        "n_clusters_gt1": sum(1 for row in cluster_rows if row["cluster_size"] > 1),
        "full_corr_pair_count": len(all_full),
        "low_corr_pair_count": len(all_low),
        "full_corr_mean": float(np.mean(all_full)) if all_full else None,
        "full_corr_min": float(np.min(all_full)) if all_full else None,
        "full_corr_p10": float(np.percentile(all_full, 10)) if all_full else None,
        "low_corr_mean": float(np.mean(all_low)) if all_low else None,
        "low_corr_min": float(np.min(all_low)) if all_low else None,
        "low_corr_p10": float(np.percentile(all_low, 10)) if all_low else None,
    }

    Path(args.out_json).write_text(json.dumps({"summary": summary, "clusters": cluster_rows}, indent=2))
    print(json.dumps(summary, indent=2))
    print(args.out_json)


if __name__ == "__main__":
    main()
