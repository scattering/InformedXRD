#!/usr/bin/env python3
"""Run paired Top-k comparisons from XRDBench failure-mode JSON files."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("model must be NAME=PATH")
    name, raw_path = value.split("=", 1)
    if not name or not raw_path:
        raise argparse.ArgumentTypeError("model must be NAME=PATH")
    return name, Path(raw_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        action="append",
        type=parse_named_path,
        required=True,
        help="Repeat as NAME=PATH for each failure-mode JSON.",
    )
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def load_examples(path: Path) -> dict[str, dict]:
    blob = json.loads(path.read_text())
    if "models" not in blob or len(blob["models"]) != 1:
        raise ValueError(f"{path} must contain exactly one model")
    model = next(iter(blob["models"].values()))
    by_case = {}
    for row in model["examples"]:
        case_id = str(row["case_id"])
        if case_id in by_case:
            raise ValueError(f"duplicate case_id {case_id!r} in {path}")
        top5 = [int(label) for label in row["top5_eg"]]
        if len(top5) < 5 or len(set(top5)) != len(top5):
            raise ValueError(f"invalid Top-5 labels for {case_id!r} in {path}")
        if int(row["pred_eg"]) != top5[0]:
            raise ValueError(f"Top-1 mismatch for {case_id!r} in {path}")
        by_case[case_id] = {
            "true_eg": int(row["true_eg"]),
            "top5_eg": top5,
        }
    return by_case


def exact_mcnemar_p(a_only: int, b_only: int) -> float:
    discordant = a_only + b_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, i) for i in range(min(a_only, b_only) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def holm_adjust(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [0.0] * len(p_values)
    running_max = 0.0
    family_size = len(p_values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (family_size - rank) * p_values[index])
        running_max = max(running_max, candidate)
        adjusted[index] = running_max
    return adjusted


def paired_comparison(
    correct_a: np.ndarray,
    correct_b: np.ndarray,
    bootstrap: int,
    seed: int,
) -> dict:
    both = int(np.sum(correct_a & correct_b))
    a_only = int(np.sum(correct_a & ~correct_b))
    b_only = int(np.sum(~correct_a & correct_b))
    neither = int(np.sum(~correct_a & ~correct_b))
    difference = correct_a.astype(np.int8) - correct_b.astype(np.int8)

    rng = np.random.default_rng(seed)
    draws = np.empty(bootstrap, dtype=np.float64)
    for index in range(bootstrap):
        sample = rng.integers(0, difference.size, size=difference.size)
        draws[index] = difference[sample].mean()

    return {
        "n": int(difference.size),
        "both_correct": both,
        "model_a_only_correct": a_only,
        "model_b_only_correct": b_only,
        "neither_correct": neither,
        "model_a_accuracy_pct": float(100.0 * correct_a.mean()),
        "model_b_accuracy_pct": float(100.0 * correct_b.mean()),
        "model_a_minus_b_percentage_points": float(100.0 * difference.mean()),
        "paired_bootstrap_95_percentage_points": [
            float(100.0 * np.quantile(draws, 0.025)),
            float(100.0 * np.quantile(draws, 0.975)),
        ],
        "mcnemar_exact_two_sided_p": exact_mcnemar_p(a_only, b_only),
    }


def main() -> None:
    args = parse_args()
    if len(args.model) < 2:
        raise ValueError("at least two models are required")
    if len({name for name, _ in args.model}) != len(args.model):
        raise ValueError("model names must be unique")
    if any(k < 1 or k > 5 for k in args.top_k):
        raise ValueError("--top-k values must be between 1 and 5")

    models = {name: load_examples(path) for name, path in args.model}
    reference_name = args.model[0][0]
    case_ids = list(models[reference_name])
    reference_cases = set(case_ids)
    for name, examples in models.items():
        if set(examples) != reference_cases:
            raise ValueError(f"{name} does not contain the same case IDs")
        for case_id in case_ids:
            if examples[case_id]["true_eg"] != models[reference_name][case_id]["true_eg"]:
                raise ValueError(f"true-label mismatch for {case_id!r} in {name}")

    correctness = {}
    for name, examples in models.items():
        correctness[name] = {}
        for k in sorted(set(args.top_k)):
            correctness[name][k] = np.asarray(
                [
                    examples[case_id]["true_eg"]
                    in examples[case_id]["top5_eg"][:k]
                    for case_id in case_ids
                ],
                dtype=bool,
            )

    comparisons = []
    model_pairs = list(itertools.combinations(models, 2))
    for k in sorted(set(args.top_k)):
        family_start = len(comparisons)
        for pair_index, (name_a, name_b) in enumerate(model_pairs):
            result = paired_comparison(
                correctness[name_a][k],
                correctness[name_b][k],
                args.bootstrap,
                args.seed + 1000 * k + pair_index,
            )
            result.update(
                {
                    "top_k": k,
                    "model_a": name_a,
                    "model_b": name_b,
                }
            )
            comparisons.append(result)

        family = comparisons[family_start:]
        adjusted = holm_adjust(
            [row["mcnemar_exact_two_sided_p"] for row in family]
        )
        for row, adjusted_p in zip(family, adjusted):
            row["mcnemar_holm_p_within_topk_family"] = adjusted_p

    output = {
        "benchmark": args.benchmark,
        "models": {
            name: str(path)
            for name, path in args.model
        },
        "n": len(case_ids),
        "bootstrap_replicates": args.bootstrap,
        "bootstrap_seed": args.seed,
        "multiplicity": (
            "Holm adjustment is across the three model pairs separately "
            "within each Top-k metric."
        ),
        "comparisons": comparisons,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
