#!/usr/bin/env python3
import argparse
import json
from collections import Counter, deque

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Bootstrap confidence intervals for topology-aware error metrics.")
    parser.add_argument("--graph-json", required=True, help="Path to extinction_group_adjacency.json")
    parser.add_argument("--failure-json", required=True, help="Path to compare_325_failure_modes.json")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def build_graph(graph_json_path):
    with open(graph_json_path, "r") as handle:
        data = json.load(handle)

    nodes = {str(key): value for key, value in data["nodes"].items()}
    directed = {str(key): [str(child) for child in value] for key, value in data["adjacency"].items()}
    undirected = {node: set() for node in nodes}
    for parent, children in directed.items():
        for child in children:
            undirected.setdefault(parent, set()).add(child)
            undirected.setdefault(child, set()).add(parent)

    label_to_node = {}
    for node, attrs in nodes.items():
        for raw_eg in attrs.get("merged_from", []):
            label_to_node[int(raw_eg) + 1] = node

    return directed, undirected, label_to_node


def shortest_distance(undirected, src, dst):
    if src == dst:
        return 0
    queue = deque([(src, 0)])
    seen = {src}
    while queue:
        node, dist = queue.popleft()
        for nxt in undirected.get(node, ()):
            if nxt == dst:
                return dist + 1
            if nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, dist + 1))
    return None


def has_directed_path(directed, src, dst):
    if src == dst:
        return True
    queue = deque([src])
    seen = {src}
    while queue:
        node = queue.popleft()
        for nxt in directed.get(node, ()):
            if nxt == dst:
                return True
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return False


def classify_examples(examples, directed, undirected, label_to_node):
    rows = []
    for example in examples:
        true_node = label_to_node.get(int(example["true_eg"]))
        pred_node = label_to_node.get(int(example["pred_eg"]))
        if true_node is None or pred_node is None:
            rows.append({"correct": False, "direction": "unmapped", "distance": None})
            continue
        if true_node == pred_node:
            rows.append({"correct": True, "direction": "correct", "distance": 0})
            continue

        dist = shortest_distance(undirected, true_node, pred_node)
        if has_directed_path(directed, true_node, pred_node):
            direction = "descendant"
        elif has_directed_path(directed, pred_node, true_node):
            direction = "ancestor"
        else:
            direction = "branch_jump"
        rows.append({"correct": False, "direction": direction, "distance": dist})
    return rows


def summarize_rows(rows):
    n = len(rows)
    correct = np.array([1.0 if row["correct"] else 0.0 for row in rows], dtype=np.float64)
    wrong_rows = [row for row in rows if not row["correct"]]
    wrong_n = len(wrong_rows)
    finite_dists = [row["distance"] for row in wrong_rows if row["distance"] is not None]
    direction_counts = Counter(row["direction"] for row in rows)

    out = {
        "top1_pct": 100.0 * float(correct.mean()) if n else None,
        "correct": int(direction_counts.get("correct", 0)),
        "wrong_n": int(wrong_n),
        "distance_buckets": {
            "1": int(sum(1 for row in wrong_rows if row["distance"] == 1)),
            "2": int(sum(1 for row in wrong_rows if row["distance"] == 2)),
            "3": int(sum(1 for row in wrong_rows if row["distance"] == 3)),
            "4plus": int(sum(1 for row in wrong_rows if row["distance"] is not None and row["distance"] >= 4)),
            "inf": int(sum(1 for row in wrong_rows if row["distance"] is None)),
        },
        "pct_wrong_le2": (100.0 * sum(1 for row in wrong_rows if row["distance"] in (1, 2)) / wrong_n) if wrong_n else None,
        "mean_wrong_dist": (float(np.mean(finite_dists)) if finite_dists else None),
        "directionality": {
            "descendant_pred_lower_sym": int(direction_counts.get("descendant", 0)),
            "ancestor_pred_higher_sym": int(direction_counts.get("ancestor", 0)),
            "branch_jump": int(direction_counts.get("branch_jump", 0)),
            "correct": int(direction_counts.get("correct", 0)),
        },
    }

    if wrong_n:
        out["directionality_fraction"] = {
            "descendant_pred_lower_sym": 100.0 * direction_counts.get("descendant", 0) / wrong_n,
            "ancestor_pred_higher_sym": 100.0 * direction_counts.get("ancestor", 0) / wrong_n,
            "branch_jump": 100.0 * direction_counts.get("branch_jump", 0) / wrong_n,
        }
    else:
        out["directionality_fraction"] = None
    return out


def bootstrap_rows(rows, rounds, seed):
    rng = np.random.default_rng(seed)
    n = len(rows)
    stats = {
        "top1_pct": [],
        "pct_wrong_le2": [],
        "mean_wrong_dist": [],
        "descendant_pct_of_wrong": [],
        "ancestor_pct_of_wrong": [],
        "branch_jump_pct_of_wrong": [],
    }

    for _ in range(rounds):
        idx = rng.integers(0, n, size=n)
        sample = [rows[i] for i in idx]
        summary = summarize_rows(sample)
        stats["top1_pct"].append(summary["top1_pct"])
        if summary["pct_wrong_le2"] is not None:
            stats["pct_wrong_le2"].append(summary["pct_wrong_le2"])
        if summary["mean_wrong_dist"] is not None:
            stats["mean_wrong_dist"].append(summary["mean_wrong_dist"])
        if summary["directionality_fraction"] is not None:
            stats["descendant_pct_of_wrong"].append(summary["directionality_fraction"]["descendant_pred_lower_sym"])
            stats["ancestor_pct_of_wrong"].append(summary["directionality_fraction"]["ancestor_pred_higher_sym"])
            stats["branch_jump_pct_of_wrong"].append(summary["directionality_fraction"]["branch_jump"])

    out = {}
    for key, values in stats.items():
        if not values:
            out[key] = None
            continue
        arr = np.asarray(values, dtype=np.float64)
        out[key] = {
            "mean": float(arr.mean()),
            "ci95_low": float(np.quantile(arr, 0.025)),
            "ci95_high": float(np.quantile(arr, 0.975)),
        }
    return out


def main():
    args = parse_args()
    directed, undirected, label_to_node = build_graph(args.graph_json)

    with open(args.failure_json, "r") as handle:
        failure_data = json.load(handle)

    output = {
        "graph_json": args.graph_json,
        "failure_json": args.failure_json,
        "bootstrap_rounds": args.bootstrap,
        "models": {},
    }
    for name, blob in failure_data["models"].items():
        rows = classify_examples(blob["examples"], directed, undirected, label_to_node)
        output["models"][name] = {
            "point_estimate": summarize_rows(rows),
            "bootstrap": bootstrap_rows(rows, args.bootstrap, args.seed),
        }

    with open(args.output_json, "w") as handle:
        json.dump(output, handle, indent=2)
    print(json.dumps(output["models"], indent=2))


if __name__ == "__main__":
    main()
