#!/usr/bin/env python3
"""Compute an exact uniform-wrong-label null on the extinction-group DAG."""

import argparse
import json
from collections import Counter

from analyze_topological_error_distance import (
    build_graph,
    has_directed_path,
    shortest_distance,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Enumerate every wrong extinction-group prediction for each benchmark "
            "example and summarize the resulting DAG distances and directions."
        )
    )
    parser.add_argument("--graph-json", required=True)
    parser.add_argument(
        "--failure-json",
        required=True,
        help="Failure-mode JSON whose examples provide the benchmark true labels.",
    )
    parser.add_argument(
        "--model-key",
        help="Model entry to read. Required when the failure JSON contains multiple models.",
    )
    parser.add_argument(
        "--condition-on-model-errors",
        action="store_true",
        help=(
            "Use only examples for which pred_eg differs from true_eg. This gives "
            "a model-matched null for comparison with wrong-case DAG summaries."
        ),
    )
    parser.add_argument("--num-labels", type=int, default=99)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def select_model(failure_data, requested_key):
    if "examples" in failure_data:
        if requested_key:
            raise ValueError(
                "--model-key is not valid for a flat evaluator result"
            )
        model_key = (
            failure_data.get("checkpoint_label")
            or failure_data.get("model")
            or "flat_evaluator_result"
        )
        return model_key, failure_data

    models = failure_data["models"]
    if requested_key:
        if requested_key not in models:
            raise KeyError(f"Model key {requested_key!r} is not present")
        return requested_key, models[requested_key]
    if len(models) != 1:
        raise ValueError("--model-key is required when multiple models are present")
    return next(iter(models.items()))


def main():
    args = parse_args()
    _, directed, undirected, label_to_node = build_graph(args.graph_json)

    with open(args.failure_json, "r") as handle:
        failure_data = json.load(handle)
    model_key, model_blob = select_model(failure_data, args.model_key)
    examples = model_blob["examples"]
    if args.condition_on_model_errors:
        examples = [
            example
            for example in examples
            if int(example["pred_eg"]) != int(example["true_eg"])
        ]
    true_labels = [int(example["true_eg"]) for example in examples]
    if not true_labels:
        raise ValueError("No benchmark examples satisfy the requested conditioning")

    distances = Counter()
    directions = Counter()
    pair_count = 0

    for true_label in true_labels:
        true_node = label_to_node.get(true_label)
        if true_node is None:
            raise ValueError(f"True label {true_label} is absent from the DAG")

        for predicted_label in range(1, args.num_labels + 1):
            if predicted_label == true_label:
                continue
            predicted_node = label_to_node.get(predicted_label)
            if predicted_node is None:
                raise ValueError(f"Predicted label {predicted_label} is absent from the DAG")

            distance = shortest_distance(undirected, true_node, predicted_node)
            distances["inf" if distance is None else distance] += 1
            pair_count += 1

            if true_node == predicted_node:
                directions["same_condensed_node"] += 1
            elif has_directed_path(directed, true_node, predicted_node):
                directions["descendant_pred_lower_sym"] += 1
            elif has_directed_path(directed, predicted_node, true_node):
                directions["ancestor_pred_higher_sym"] += 1
            else:
                directions["branch_jump"] += 1

    finite_distance_sum = sum(
        distance * count
        for distance, count in distances.items()
        if distance != "inf"
    )
    finite_count = pair_count - distances["inf"]

    direction_counts = {
        key: int(directions[key])
        for key in (
            "descendant_pred_lower_sym",
            "ancestor_pred_higher_sym",
            "branch_jump",
            "same_condensed_node",
        )
    }
    result = {
        "null_definition": (
            "For each benchmark true label, enumerate all 98 incorrect labels "
            "uniformly and summarize their positions on the condensed DAG."
        ),
        "graph_json": args.graph_json,
        "failure_json": args.failure_json,
        "source_model_key": model_key,
        "conditioning": (
            "model_wrong_cases"
            if args.condition_on_model_errors
            else "all_benchmark_cases"
        ),
        "n_source_examples": len(model_blob["examples"]),
        "n_examples": len(true_labels),
        "num_labels": args.num_labels,
        "n_wrong_label_pairs": pair_count,
        "mean_wrong_distance": finite_distance_sum / finite_count,
        "pct_wrong_distance_le2": (
            100.0 * (distances[0] + distances[1] + distances[2]) / pair_count
        ),
        "distance_counts": {
            str(key): int(value)
            for key, value in sorted(
                distances.items(),
                key=lambda item: float("inf") if item[0] == "inf" else item[0],
            )
        },
        "directionality_counts": direction_counts,
        "directionality_pct": {
            key: 100.0 * value / pair_count
            for key, value in direction_counts.items()
        },
        "descendant_including_same_condensed_node_pct": (
            100.0
            * (
                directions["descendant_pred_lower_sym"]
                + directions["same_condensed_node"]
            )
            / pair_count
        ),
    }

    with open(args.output_json, "w") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
