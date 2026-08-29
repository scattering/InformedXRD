#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Construct an AlphaDiffract-like RRUFF subset from the full RRUFF search surface.")
    p.add_argument("--search-json", required=True)
    p.add_argument("--cell-threshold", type=float, default=0.006)
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


def main() -> None:
    args = parse_args()
    payload = json.loads(Path(args.search_json).read_text())
    rows = payload["rows"]

    eligible = [
        row
        for row in rows
        if row["grid_ok"] and row["top1_sg_exact"] and row["top1_loose_cell_consistent"]
    ]

    by_group: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in eligible:
        by_group[(row["name"], row["space_group_symbol"])].append(row)

    selected = []
    cluster_rows = []
    for (name, sg), group in by_group.items():
        dsu = DSU(len(group))
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                if cell_distance(group[i]["cell"], group[j]["cell"]) <= args.cell_threshold:
                    dsu.union(i, j)
        comps: dict[int, list[dict]] = defaultdict(list)
        for i, row in enumerate(group):
            comps[dsu.find(i)].append(row)
        for comp_id, comp in comps.items():
            rep = min(
                comp,
                key=lambda row: (
                    row["top1_score"] if row["top1_score"] is not None else 1e9,
                    -(row["max_intensity"] if row["max_intensity"] is not None else 0.0),
                    row["source_case_id"],
                ),
            )
            selected.append(rep)
            cluster_rows.append(
                {
                    "name": name,
                    "space_group_symbol": sg,
                    "cluster_size": len(comp),
                    "representative_case_id": rep["source_case_id"],
                    "representative_rid": rep["rid"],
                    "representative_top1_icsd_idnum": rep["top1_icsd_idnum"],
                    "members": [row["source_case_id"] for row in comp],
                }
            )

    summary = {
        "cell_threshold": args.cell_threshold,
        "n_search_rows": len(rows),
        "n_eligible_rows": len(eligible),
        "n_reconstructed_cases": len(selected),
        "n_unique_minerals": len({row["name"] for row in selected}),
        "n_rruff473_overlap": sum(1 for row in selected if row["in_rruff473"]),
    }
    out = {"summary": summary, "selected_rows": selected, "clusters": cluster_rows}
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    print(json.dumps(summary, indent=2))
    print(args.out_json)


if __name__ == "__main__":
    main()
