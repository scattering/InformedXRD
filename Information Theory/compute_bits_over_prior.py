#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np


LOG2 = np.log(2.0)


def parse_args():
    parser = argparse.ArgumentParser(description="Compute Gate 1 bits-over-prior metrics from full posterior dumps.")
    parser.add_argument("--posterior-npz", required=True)
    parser.add_argument("--posterior-key", default="calibrated_probs", choices=["calibrated_probs", "raw_probs"])
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--permutation-rounds", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--temperature-grid",
        type=float,
        nargs="+",
        default=[0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0],
        help="Diagnostic temperature grid for NLL-tuned oracle-T analysis using raw_probs plus stored prior.",
    )
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def to_bits_nll(true_probs):
    return -np.log(np.clip(true_probs, 1e-12, 1.0)) / LOG2


def empirical_prior_from_targets(targets, num_classes):
    counts = np.bincount(targets, minlength=num_classes).astype(np.float64)
    probs = counts / counts.sum()
    return probs


def topk_accuracy(probs, targets, k):
    idx = np.argpartition(-probs, kth=k - 1, axis=1)[:, :k]
    return float(np.mean([int(t in row) for t, row in zip(targets.tolist(), idx.tolist())]))


def summarize(probs, targets):
    n, num_classes = probs.shape
    prior = empirical_prior_from_targets(targets, num_classes)
    prior_true_probs = prior[targets]
    model_true_probs = probs[np.arange(n), targets]

    nll_prior = to_bits_nll(prior_true_probs)
    nll_model = to_bits_nll(model_true_probs)
    bits_over_prior = nll_prior.mean() - nll_model.mean()
    delta_i = np.log(np.clip(model_true_probs, 1e-12, 1.0)) / LOG2 - np.log(np.clip(prior_true_probs, 1e-12, 1.0)) / LOG2

    kl_i = np.sum(
        probs * ((np.log(np.clip(probs, 1e-12, 1.0)) - np.log(np.clip(prior[None, :], 1e-12, 1.0))) / LOG2),
        axis=1,
    )
    posterior_entropy = -np.sum(probs * (np.log(np.clip(probs, 1e-12, 1.0)) / LOG2), axis=1)
    h_prior = -np.sum(prior * (np.log(np.clip(prior, 1e-12, 1.0)) / LOG2))

    return {
        "n_examples": int(n),
        "num_classes": int(num_classes),
        "h_prior_bits": float(h_prior),
        "nll_prior_bits": float(nll_prior.mean()),
        "nll_model_bits": float(nll_model.mean()),
        "bits_over_prior_bits": float(bits_over_prior),
        "top1": float(np.mean(probs.argmax(axis=1) == targets)),
        "top5": topk_accuracy(probs, targets, 5),
        "mean_posterior_entropy_bits": float(posterior_entropy.mean()),
        "plugin_mi_bits": float(h_prior - posterior_entropy.mean()),
        "effective_candidates": float(2.0 ** posterior_entropy.mean()),
        "delta_i": {
            "mean_bits": float(delta_i.mean()),
            "std_bits": float(delta_i.std(ddof=1)),
            "p05_bits": float(np.quantile(delta_i, 0.05)),
            "p25_bits": float(np.quantile(delta_i, 0.25)),
            "median_bits": float(np.quantile(delta_i, 0.50)),
            "p75_bits": float(np.quantile(delta_i, 0.75)),
            "p95_bits": float(np.quantile(delta_i, 0.95)),
            "min_bits": float(delta_i.min()),
            "max_bits": float(delta_i.max()),
            "fraction_positive": float(np.mean(delta_i > 0.0)),
        },
        "kl_i": {
            "mean_bits": float(kl_i.mean()),
            "std_bits": float(kl_i.std(ddof=1)),
            "p05_bits": float(np.quantile(kl_i, 0.05)),
            "p25_bits": float(np.quantile(kl_i, 0.25)),
            "median_bits": float(np.quantile(kl_i, 0.50)),
            "p75_bits": float(np.quantile(kl_i, 0.75)),
            "p95_bits": float(np.quantile(kl_i, 0.95)),
            "min_bits": float(kl_i.min()),
            "max_bits": float(kl_i.max()),
        },
        "empirical_prior": prior.tolist(),
    }


def bootstrap_bits_over_prior(probs, targets, rounds, seed):
    rng = np.random.default_rng(seed)
    n = targets.shape[0]
    values = np.empty(rounds, dtype=np.float64)
    for i in range(rounds):
        idx = rng.integers(0, n, size=n)
        values[i] = summarize(probs[idx], targets[idx])["bits_over_prior_bits"]
    return {
        "mean_bits": float(values.mean()),
        "ci95_low_bits": float(np.quantile(values, 0.025)),
        "ci95_high_bits": float(np.quantile(values, 0.975)),
    }


def permutation_null(probs, targets, rounds, seed):
    rng = np.random.default_rng(seed)
    n = targets.shape[0]
    num_classes = probs.shape[1]
    prior = empirical_prior_from_targets(targets, num_classes)
    prior_nll_bits = to_bits_nll(prior[targets]).mean()
    values = np.empty(rounds, dtype=np.float64)
    for i in range(rounds):
        permuted = rng.permutation(targets)
        model_nll_bits = to_bits_nll(probs[np.arange(n), permuted]).mean()
        values[i] = prior_nll_bits - model_nll_bits
    return {
        "mean_bits": float(values.mean()),
        "ci95_low_bits": float(np.quantile(values, 0.025)),
        "ci95_high_bits": float(np.quantile(values, 0.975)),
    }


def gate1_decision(point, bootstrap, null):
    # AGENTS.md decision rule:
    # - GO only if bits_over_prior is positive
    # - and the bootstrap CI sits above the permutation-null band
    # - and KL_i shows real spread rather than collapsing to a flat constant
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


def infer_seen_mask_from_calibrated(calibrated_probs):
    return np.any(np.asarray(calibrated_probs) > 0.0, axis=0)


def retemperature_with_prior(raw_probs, prior_probs, seen_mask, temperature):
    log_raw = np.log(np.clip(raw_probs, 1e-12, 1.0))
    log_prior = np.log(np.clip(prior_probs, 1e-12, 1.0))
    scores = log_raw / float(temperature) + log_prior[None, :]
    if seen_mask is not None:
        scores = np.where(seen_mask[None, :], scores, -np.inf)
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    return probs


def oracle_temperature_sweep(raw_probs, calibrated_probs, prior_probs, targets, temperature_grid):
    seen_mask = infer_seen_mask_from_calibrated(calibrated_probs)
    sweep = []
    best = None
    for temp in temperature_grid:
        probs = retemperature_with_prior(raw_probs, prior_probs, seen_mask, temp)
        summary = summarize(probs, targets)
        row = {
            "temperature": float(temp),
            "nll_model_bits": summary["nll_model_bits"],
            "bits_over_prior_bits": summary["bits_over_prior_bits"],
            "top1": summary["top1"],
            "top5": summary["top5"],
            "delta_i_median_bits": summary["delta_i"]["median_bits"],
            "delta_i_fraction_positive": summary["delta_i"]["fraction_positive"],
        }
        sweep.append(row)
        if best is None or row["nll_model_bits"] < best["nll_model_bits"]:
            best = row
    return {
        "temperature_grid": [float(v) for v in temperature_grid],
        "best_by_nll": best,
        "sweep": sweep,
    }


def main():
    args = parse_args()
    data = np.load(args.posterior_npz, allow_pickle=True)
    probs = np.asarray(data[args.posterior_key], dtype=np.float64)
    targets = np.asarray(data["targets"], dtype=np.int64)
    raw_probs = np.asarray(data["raw_probs"], dtype=np.float64) if "raw_probs" in data else None
    calibrated_probs = np.asarray(data["calibrated_probs"], dtype=np.float64) if "calibrated_probs" in data else None
    prior_probs = np.asarray(data["prior_probs"], dtype=np.float64) if "prior_probs" in data else None

    point = summarize(probs, targets)
    bootstrap = bootstrap_bits_over_prior(probs, targets, args.bootstrap, args.seed)
    null = permutation_null(probs, targets, args.permutation_rounds, args.seed + 1)
    decision = gate1_decision(point, bootstrap, null)

    oracle_t = None
    if raw_probs is not None and calibrated_probs is not None and prior_probs is not None:
        oracle_t = oracle_temperature_sweep(
            raw_probs=raw_probs,
            calibrated_probs=calibrated_probs,
            prior_probs=prior_probs,
            targets=targets,
            temperature_grid=args.temperature_grid,
        )

    output = {
        "posterior_npz": args.posterior_npz,
        "posterior_key": args.posterior_key,
        "point_estimate": point,
        "bootstrap_bits_over_prior": bootstrap,
        "permutation_null_bits_over_prior": null,
        "gate1_decision_rule": {
            "source": "Information Theory/AGENTS.md",
            "summary": "GO only if bits_over_prior is positive, the bootstrap CI sits above the permutation-null band, and KL_i shows real spread.",
        },
        "gate1_checks": decision,
        "gate1_passes": decision["passes"],
    }
    if oracle_t is not None:
        output["oracle_temperature_nll_diagnostic"] = oracle_t
    Path(args.output_json).write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
