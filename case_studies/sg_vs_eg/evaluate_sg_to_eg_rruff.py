#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from evaluate_sg_to_eg_control import (
    build_categorical_model,
    build_multilabel_aux_model,
    collect_eg_aux_probs,
    collect_sg_probs,
    collapse_sg_probs_to_eg,
    load_config,
    load_eg_targets,
    load_sg_to_eg_map,
    summarize_probs,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Score the matched synthetic SG and direct-EG checkpoints on real "
            "RRUFF patterns in the common 99-way extinction-group label space."
        )
    )
    parser.add_argument("--sg-checkpoint", required=True)
    parser.add_argument("--sg-config", required=True)
    parser.add_argument("--eg-checkpoint", required=True)
    parser.add_argument("--eg-config", required=True)
    parser.add_argument("--canonical-table-path", required=True)
    parser.add_argument(
        "--benchmark",
        action="append",
        required=True,
        metavar="NAME=H5_PATH",
        help="Repeat for each RRUFF benchmark.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def parse_benchmarks(values):
    benchmarks = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Benchmark must be NAME=H5_PATH, got {value!r}")
        name, path = value.split("=", 1)
        if not name or not path:
            raise ValueError(f"Benchmark must be NAME=H5_PATH, got {value!r}")
        benchmarks.append((name, path))
    return benchmarks


def topk_correct(probs, targets, k):
    topk = np.argpartition(-probs, kth=k - 1, axis=1)[:, :k]
    return np.asarray(
        [target in row for target, row in zip(targets.tolist(), topk.tolist())],
        dtype=bool,
    )


def exact_mcnemar_p(model_a_correct, model_b_correct):
    a_only = int(np.sum(model_a_correct & ~model_b_correct))
    b_only = int(np.sum(~model_a_correct & model_b_correct))
    discordant = a_only + b_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, i) for i in range(min(a_only, b_only) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def paired_topk_comparison(direct_probs, collapsed_probs, targets, k, seed=20260726):
    direct_correct = topk_correct(direct_probs, targets, k)
    collapsed_correct = topk_correct(collapsed_probs, targets, k)
    both = int(np.sum(direct_correct & collapsed_correct))
    direct_only = int(np.sum(direct_correct & ~collapsed_correct))
    collapsed_only = int(np.sum(~direct_correct & collapsed_correct))
    neither = int(np.sum(~direct_correct & ~collapsed_correct))

    rng = np.random.default_rng(seed + k)
    paired_difference = direct_correct.astype(float) - collapsed_correct.astype(float)
    bootstrap = np.empty(10000, dtype=float)
    for i in range(bootstrap.size):
        indices = rng.integers(0, targets.shape[0], size=targets.shape[0])
        bootstrap[i] = paired_difference[indices].mean()

    return {
        "both_correct": both,
        "direct_eg_only_correct": direct_only,
        "collapsed_sg_only_correct": collapsed_only,
        "neither_correct": neither,
        "direct_eg_minus_collapsed_sg": float(paired_difference.mean()),
        "paired_bootstrap_95": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "mcnemar_exact_two_sided_p": exact_mcnemar_p(
            direct_correct, collapsed_correct
        ),
    }


def main():
    args = parse_args()
    device = torch.device(args.device)

    sg_checkpoint = torch.load(args.sg_checkpoint, map_location="cpu")
    sg_config = load_config(args.sg_config)
    sg_model = build_categorical_model(sg_config, sg_checkpoint, device)

    eg_checkpoint = torch.load(args.eg_checkpoint, map_location="cpu")
    eg_config = load_config(args.eg_config)
    eg_model = build_multilabel_aux_model(eg_config, eg_checkpoint, device)

    sg_to_eg = load_sg_to_eg_map(args.canonical_table_path)
    results = {}
    for name, benchmark_path in parse_benchmarks(args.benchmark):
        eg_targets = load_eg_targets(benchmark_path)

        # The RRUFF HDF5 label is already an EG. The SG loader's target is
        # ignored; only its identically sliced intensity array is used.
        sg_probs = collect_sg_probs(
            sg_model,
            sg_config,
            benchmark_path,
            args.num_workers,
            device,
        )
        collapsed_probs = collapse_sg_probs_to_eg(sg_probs, sg_to_eg)
        direct_eg_probs = collect_eg_aux_probs(
            eg_model,
            eg_config,
            benchmark_path,
            args.num_workers,
            device,
        )

        n_samples = eg_targets.shape[0]
        if collapsed_probs.shape != direct_eg_probs.shape:
            raise ValueError(
                f"{name}: collapsed-SG shape {collapsed_probs.shape} does not "
                f"match direct-EG shape {direct_eg_probs.shape}"
            )
        if collapsed_probs.shape != (n_samples, 99):
            raise ValueError(
                f"{name}: expected {(n_samples, 99)} probabilities, got "
                f"{collapsed_probs.shape}"
            )

        collapsed_metrics = summarize_probs(collapsed_probs, eg_targets)
        direct_metrics = summarize_probs(direct_eg_probs, eg_targets)
        results[name] = {
            "path": benchmark_path,
            "n_samples": int(n_samples),
            "represented_extinction_groups": int(np.unique(eg_targets).size),
            "sg_to_eg_collapsed": collapsed_metrics,
            "direct_eg_aux": direct_metrics,
            "direct_eg_minus_collapsed_sg": {
                key: float(direct_metrics[key] - collapsed_metrics[key])
                for key in ("top1", "top3", "top5")
            },
            "paired_comparisons": {
                f"top{k}": paired_topk_comparison(
                    direct_eg_probs, collapsed_probs, eg_targets, k
                )
                for k in (1, 3, 5)
            },
        }

    output = {
        "comparison": (
            "Matched synthetic-uniform checkpoints evaluated on identical "
            "RRUFF inputs in the common 99-way EG label space."
        ),
        "sg_checkpoint": args.sg_checkpoint,
        "sg_config": args.sg_config,
        "eg_checkpoint": args.eg_checkpoint,
        "eg_config": args.eg_config,
        "canonical_table_path": args.canonical_table_path,
        "benchmarks": results,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
