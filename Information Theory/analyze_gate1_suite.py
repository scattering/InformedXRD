#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np

from compute_bits_over_prior import (
    bootstrap_bits_over_prior,
    permutation_null,
    summarize,
)


def parse_args():
    p = argparse.ArgumentParser(description="Run the established real-data Gate 1 analysis on a posterior dump.")
    p.add_argument("--posterior-npz", required=True)
    p.add_argument("--posterior-key", default="calibrated_probs")
    p.add_argument("--label", default="")
    p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("--permutation-rounds", type=int, default=2000)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--output-json", required=True)
    return p.parse_args()


def gate1_decision(point, bootstrap, null):
    bits_positive = point["bits_over_prior_bits"] > 0.0
    ci_above_null = bootstrap["ci95_low_bits"] > null["ci95_high_bits"]
    kl_summary = point["kl_i"]
    kl_spread_width = kl_summary["p95_bits"] - kl_summary["p05_bits"]
    kl_nonflat = (kl_summary["std_bits"] > 1e-9) and (kl_spread_width > 1e-9)
    return {
        "bits_over_prior_positive": bool(bits_positive),
        "bootstrap_ci_above_permutation_null": bool(ci_above_null),
        "kl_i_nonflat": bool(kl_nonflat),
        "kl_i_spread_width_bits": float(kl_spread_width),
        "passes": bool(bits_positive and ci_above_null and kl_nonflat),
    }


def topk_sets(probs, k):
    return np.argpartition(-probs, kth=k - 1, axis=1)[:, :k]


def topk_accuracy(probs, labels, k):
    sets = topk_sets(probs, k)
    return float(np.mean([labels[i] in sets[i] for i in range(len(labels))]))


def true_label_rank(probs, labels):
    true_p = probs[np.arange(len(labels)), labels]
    return 1 + np.sum(probs > true_p[:, None], axis=1)


def empirical_prior(labels, n_classes):
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    return counts / counts.sum()


def prior_topk_baseline(labels, n_classes, k):
    phat = empirical_prior(labels, n_classes)
    top = np.argpartition(-phat, kth=k - 1)[:k]
    top = set(top.tolist())
    return float(np.mean([labels[i] in top for i in range(len(labels))]))


def shuffle_null_topk_closed_form(probs, labels, n_classes, k):
    phat = empirical_prior(labels, n_classes)
    sets = topk_sets(probs, k)
    return float(np.mean(phat[sets].sum(axis=1)))


def permutation_null_stat(stat_fn, probs, labels, n_perm=2000, seed=0, lower_is_better=False):
    rng = np.random.default_rng(seed)
    observed = stat_fn(probs, labels)
    null = np.empty(n_perm)
    y = labels.copy()
    for b in range(n_perm):
        rng.shuffle(y)
        null[b] = stat_fn(probs, y)
    null_mean, null_std = float(null.mean()), float(null.std(ddof=1))
    lo, hi = np.percentile(null, [2.5, 97.5])
    z = (observed - null_mean) / (null_std + 1e-12)
    if lower_is_better:
        z = -z
        p = (np.sum(null <= observed) + 1) / (n_perm + 1)
    else:
        p = (np.sum(null >= observed) + 1) / (n_perm + 1)
    return {
        "observed": float(observed),
        "null_mean": null_mean,
        "null_std": null_std,
        "null_95": [float(lo), float(hi)],
        "z": float(z),
        "p_one_sided": float(p),
    }


def bootstrap_ci(stat_fn, probs, labels, n_boot=2000, seed=1):
    rng = np.random.default_rng(seed)
    n = len(labels)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        vals[b] = stat_fn(probs[idx], labels[idx])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"mean": float(vals.mean()), "ci95": [float(lo), float(hi)]}


def rank_analyze(probs, labels, n_perm=2000, ks=(1, 3, 5)):
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    probs = probs / probs.sum(axis=1, keepdims=True)
    n_classes = probs.shape[1]
    out = {"n_samples": int(len(labels)), "n_classes": int(n_classes), "topk": {}}
    for k in ks:
        obs = topk_accuracy(probs, labels, k)
        prior_base = prior_topk_baseline(labels, n_classes, k)
        cf_null = shuffle_null_topk_closed_form(probs, labels, n_classes, k)
        perm = permutation_null_stat(lambda p, y, k=k: topk_accuracy(p, y, k), probs, labels, n_perm=n_perm)
        boot = bootstrap_ci(lambda p, y, k=k: topk_accuracy(p, y, k), probs, labels, n_boot=n_perm, seed=1)
        out["topk"][k] = {
            "observed": obs,
            "prior_static_baseline": prior_base,
            "shuffle_null_closed_form": cf_null,
            "permutation_null": perm,
            "bootstrap": boot,
            "pattern_conditional_lift": obs - perm["null_mean"],
        }
    mean_rank = lambda p, y: float(np.mean(true_label_rank(p, y)))
    out["mean_true_rank"] = {
        "observed": mean_rank(probs, labels),
        "permutation_null": permutation_null_stat(mean_rank, probs, labels, n_perm=n_perm, lower_is_better=True),
    }
    return out


def main():
    args = parse_args()
    data = np.load(args.posterior_npz, allow_pickle=True)
    probs = np.asarray(data[args.posterior_key], dtype=np.float64)
    targets = np.asarray(data["targets"], dtype=np.int64)

    # Some comparison-model adapters emit canonical EG labels in 1..99 form
    # rather than zero-based class ids. Normalize once here so the Gate 1
    # analysis can run on a common footing across model families.
    if targets.size and targets.min() >= 1 and targets.max() == probs.shape[1]:
        targets = targets - 1

    bits_point = summarize(probs, targets)
    bits_boot = bootstrap_bits_over_prior(probs, targets, args.bootstrap, args.seed)
    bits_null = permutation_null(probs, targets, args.permutation_rounds, args.seed + 1)
    bits_decision = gate1_decision(bits_point, bits_boot, bits_null)
    rank = rank_analyze(probs, targets, n_perm=args.permutation_rounds, ks=(1, 3, 5))

    output = {
        "label": args.label,
        "posterior_npz": args.posterior_npz,
        "posterior_key": args.posterior_key,
        "n_samples": int(targets.shape[0]),
        "bits_over_prior": {
            "point_estimate": bits_point,
            "bootstrap": bits_boot,
            "permutation_null": bits_null,
            "gate1_checks": bits_decision,
        },
        "rank_lift": rank,
        "compact_summary": {
            "bits_over_prior_bits": float(bits_point["bits_over_prior_bits"]),
            "bits_over_prior_null_mean": float(bits_null["mean_bits"]),
            "top1": float(rank["topk"][1]["observed"]),
            "top1_null_mean": float(rank["topk"][1]["permutation_null"]["null_mean"]),
            "top5": float(rank["topk"][5]["observed"]),
            "top5_null_mean": float(rank["topk"][5]["permutation_null"]["null_mean"]),
            "top5_pattern_conditional_lift": float(rank["topk"][5]["pattern_conditional_lift"]),
            "mean_true_rank": float(rank["mean_true_rank"]["observed"]),
            "mean_true_rank_null_mean": float(rank["mean_true_rank"]["permutation_null"]["null_mean"]),
        },
    }

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2))
    print(json.dumps(output["compact_summary"], indent=2))


if __name__ == "__main__":
    main()
