"""
rank_lift_null.py
=================

Gate-1 (corrected) significance check for the NARROW claim:

    "The model is not a flat prior ranker; it narrows the extinction-group
     candidate set on a per-pattern basis."

This uses calibration-FREE, rank-based statistics, which match that claim far
better than the strict log-loss lower bound (bits_over_prior). bits_over_prior
stays in the report as a conservative, calibration-sensitive diagnostic; it is
NOT the gate.

Primary statistic: Top-k accuracy vs a LABEL-SHUFFLE null.
    - Shuffling labels keeps the model's per-pattern top-k SETS fixed and breaks
      only the pattern<->label association.
    - Null mean (closed form): mean_i sum_{g in topk(q_i)} phat(g)
      = the average prior mass captured by the model's own top-k sets.
      Because the model favours common groups, this sits ABOVE the static prior
      top-k baseline. The gap (observed - shuffle_null) is the pattern-
      conditional re-ranking signal specifically.
    - A flat prior-ranker has observed == null by construction.

Secondary statistic: mean rank of the true label (1 = top), model vs prior,
with the same shuffle null. A single calibration-free scalar for "narrows the
set."

Inputs: calibrated probs (N, K) and integer labels (N,). If you only have
logits, apply your temperature + log-prior upstream and pass the resulting
probs, so the numbers match the reported run (e.g. T=5 or oracle T=10).

Only dependency: numpy.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

_EPS = 1e-12


# --------------------------------------------------------------------------- #
# Core statistics
# --------------------------------------------------------------------------- #
def topk_sets(probs, k):
    """Return (N, k) array of the top-k class indices per pattern."""
    # argpartition for speed, then no need to sort within the set for membership
    return np.argpartition(-probs, kth=k - 1, axis=1)[:, :k]


def topk_accuracy(probs, labels, k):
    sets = topk_sets(probs, k)
    return float(np.mean([labels[i] in sets[i] for i in range(len(labels))]))


def true_label_rank(probs, labels):
    """Rank of the true label per pattern, 1 = highest prob (lower is better)."""
    # rank = 1 + (#classes strictly more probable than the true class)
    true_p = probs[np.arange(len(labels)), labels]
    return 1 + np.sum(probs > true_p[:, None], axis=1)


def empirical_prior(labels, n_classes):
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    return counts / counts.sum()


def prior_topk_baseline(labels, n_classes, k):
    """Static baseline: the k globally most frequent classes, scored vs labels."""
    phat = empirical_prior(labels, n_classes)
    top = np.argpartition(-phat, kth=k - 1)[:k]
    top = set(top.tolist())
    return float(np.mean([labels[i] in top for i in range(len(labels))]))


def paired_static_prior_comparison(probs, labels, n_classes, k):
    """Case-level overlap between model Top-k and the fixed empirical-prior Top-k."""
    phat = empirical_prior(labels, n_classes)
    prior_set = set(np.argpartition(-phat, kth=k - 1)[:k].tolist())
    model_sets = topk_sets(probs, k)
    model_correct = np.asarray(
        [label in row for label, row in zip(labels.tolist(), model_sets.tolist())],
        dtype=bool,
    )
    prior_correct = np.asarray([label in prior_set for label in labels.tolist()], dtype=bool)
    both = int(np.sum(model_correct & prior_correct))
    model_only = int(np.sum(model_correct & ~prior_correct))
    prior_only = int(np.sum(~model_correct & prior_correct))
    neither = int(np.sum(~model_correct & ~prior_correct))
    return {
        "prior_class_indices_zero_based": sorted(prior_set),
        "both_correct": both,
        "model_only_correct": model_only,
        "prior_only_correct": prior_only,
        "neither_correct": neither,
        "model_correct_total": int(model_correct.sum()),
        "prior_correct_total": int(prior_correct.sum()),
    }


def shuffle_null_topk_closed_form(probs, labels, n_classes, k):
    """E[Top-k] under label shuffling = mean_i (prior mass in top-k set of i)."""
    phat = empirical_prior(labels, n_classes)
    sets = topk_sets(probs, k)
    return float(np.mean(phat[sets].sum(axis=1)))


# --------------------------------------------------------------------------- #
# Permutation null
# --------------------------------------------------------------------------- #
def permutation_null(stat_fn, probs, labels, n_perm=2000, seed=0,
                     lower_is_better=False):
    """
    Monte-Carlo label-shuffle null for an arbitrary statistic stat_fn(probs, y).
    Returns dict with observed, null mean/std, 95% band, z, one-sided p.
    """
    rng = np.random.default_rng(seed)
    observed = stat_fn(probs, labels)
    null = np.empty(n_perm)
    y = labels.copy()
    for b in range(n_perm):
        rng.shuffle(y)
        null[b] = stat_fn(probs, y)
    null_mean, null_std = float(null.mean()), float(null.std(ddof=1))
    lo, hi = np.percentile(null, [2.5, 97.5])
    z = (observed - null_mean) / (null_std + _EPS)
    if lower_is_better:
        z = -z
        p = (np.sum(null <= observed) + 1) / (n_perm + 1)
    else:
        p = (np.sum(null >= observed) + 1) / (n_perm + 1)
    return {
        "observed": observed,
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


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def analyze(probs, labels, n_perm=2000, ks=(1, 3, 5)):
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    probs = probs / probs.sum(axis=1, keepdims=True)
    n_classes = probs.shape[1]
    assert labels.min() >= 0 and labels.max() < n_classes

    out = {"n_samples": int(len(labels)), "n_classes": int(n_classes), "topk": {}}
    for k in ks:
        obs = topk_accuracy(probs, labels, k)
        prior_base = prior_topk_baseline(labels, n_classes, k)
        cf_null = shuffle_null_topk_closed_form(probs, labels, n_classes, k)
        perm = permutation_null(lambda p, y, k=k: topk_accuracy(p, y, k),
                                probs, labels, n_perm=n_perm)
        boot = bootstrap_ci(lambda p, y, k=k: topk_accuracy(p, y, k),
                            probs, labels)
        out["topk"][k] = {
            "observed": obs,
            "prior_static_baseline": prior_base,
            "paired_static_prior": paired_static_prior_comparison(
                probs, labels, n_classes, k
            ),
            "shuffle_null_closed_form": cf_null,
            "permutation_null": perm,
            "bootstrap": boot,
            "pattern_conditional_lift": obs - perm["null_mean"],
        }

    # mean rank of the true label (lower is better) + its shuffle null
    mean_rank = lambda p, y: float(np.mean(true_label_rank(p, y)))
    out["mean_true_rank"] = {
        "observed": mean_rank(probs, labels),
        "permutation_null": permutation_null(mean_rank, probs, labels,
                                              n_perm=n_perm, lower_is_better=True),
        "note": "1 = true label ranked top; lower is better; null = rank under "
                "shuffled labels (random pattern-label pairing).",
    }
    return out


def _load(path, key=None):
    if path.endswith(".npy"):
        return np.load(path)
    if path.endswith(".npz"):
        if key is None:
            raise ValueError("key is required for .npz inputs")
        data = np.load(path, allow_pickle=True)
        return np.asarray(data[key])
    if path.endswith(".json"):
        with open(path) as f:
            d = json.load(f)
        return np.asarray(d[key])
    raise ValueError(f"unsupported file: {path}")


def infer_seen_mask_from_calibrated(calibrated_probs):
    return np.any(np.asarray(calibrated_probs) > 0.0, axis=0)


def retemperature_with_prior(raw_probs, prior_probs, seen_mask, temperature):
    log_raw = np.log(np.clip(raw_probs, _EPS, 1.0))
    log_prior = np.log(np.clip(prior_probs, _EPS, 1.0))
    scores = log_raw / float(temperature) + log_prior[None, :]
    if seen_mask is not None:
        scores = np.where(seen_mask[None, :], scores, -np.inf)
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", help="(N,K) calibrated probs .npy/.npz")
    ap.add_argument("--labels", help="(N,) int labels .npy/.npz")
    ap.add_argument("--probs-key", default=None, help="Key for probs when --probs is .npz/.json")
    ap.add_argument("--labels-key", default=None, help="Key for labels when --labels is .npz/.json")
    ap.add_argument("--posterior-npz", help="Posterior dump .npz with calibrated_probs/raw_probs/targets/prior_probs")
    ap.add_argument("--posterior-key", default="calibrated_probs", help="Key to analyze from --posterior-npz")
    ap.add_argument("--oracle-temperature", type=float, default=None, help="If set with --posterior-npz, reconstruct probs at this T from raw_probs + prior_probs")
    ap.add_argument("--tag", default=None, help="Optional label recorded in output")
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--out", default=None, help="optional JSON output path")
    args = ap.parse_args()

    if args.posterior_npz:
        if args.probs or args.labels:
            raise ValueError("use either --posterior-npz or (--probs, --labels), not both")
        data = np.load(args.posterior_npz, allow_pickle=True)
        labels = np.asarray(data["targets"], dtype=np.int64)
        if args.oracle_temperature is not None:
            raw_probs = np.asarray(data["raw_probs"], dtype=np.float64)
            prior_probs = np.asarray(data["prior_probs"], dtype=np.float64)
            calibrated_probs = np.asarray(data["calibrated_probs"], dtype=np.float64)
            seen_mask = infer_seen_mask_from_calibrated(calibrated_probs)
            probs = retemperature_with_prior(raw_probs, prior_probs, seen_mask, args.oracle_temperature)
        else:
            probs = np.asarray(data[args.posterior_key], dtype=np.float64)
    else:
        if not args.probs or not args.labels:
            raise ValueError("either --posterior-npz or both --probs and --labels are required")
        probs = _load(args.probs, key=args.probs_key)
        labels = _load(args.labels, key=args.labels_key)

    res = analyze(probs, labels, n_perm=args.n_perm)
    if args.tag is not None:
        res["tag"] = args.tag
    if args.posterior_npz:
        res["posterior_npz"] = args.posterior_npz
        res["posterior_key"] = args.posterior_key
        if args.oracle_temperature is not None:
            res["oracle_temperature"] = float(args.oracle_temperature)
    txt = json.dumps(res, indent=2)
    print(txt)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            f.write(txt)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main()
    else:
        # ---- self-test: closed form == MC null; pattern-conditional model
        #      beats null; flat model does not -------------------------------
        rng = np.random.default_rng(42)
        N, K = 325, 99
        # imbalanced prior
        phat = rng.dirichlet(np.ones(K) * 0.3)
        labels = rng.choice(K, size=N, p=phat)

        # (a) pattern-conditional model: sharp posteriors that put the true
        #     label in the top-5 ~43% of the time, top-1 ~10%, often confident
        #     on a wrong-but-adjacent class.
        probs = np.full((N, K), 1e-3)
        for i, y in enumerate(labels):
            # decoy gets most mass; true label gets into the top set ~half time
            decoy = rng.integers(K)
            probs[i, decoy] += rng.uniform(2, 6)
            if rng.random() < 0.45:
                probs[i, y] += rng.uniform(0.5, 2.5)
            # a few random extras to fill the top-5
            probs[i, rng.integers(0, K, 4)] += rng.uniform(0.2, 1.0)
        probs /= probs.sum(axis=1, keepdims=True)

        res = analyze(probs, labels, n_perm=1000)
        t5 = res["topk"][5]
        print("self-test (pattern-conditional model):")
        print(f"  Top-1 obs={res['topk'][1]['observed']:.3f}  "
              f"Top-5 obs={t5['observed']:.3f}")
        print(f"  prior static Top-5      = {t5['prior_static_baseline']:.3f}")
        print(f"  shuffle null (closed)   = {t5['shuffle_null_closed_form']:.3f}")
        print(f"  shuffle null (MC mean)  = {t5['permutation_null']['null_mean']:.3f}")
        print(f"  pattern-cond. lift      = {t5['pattern_conditional_lift']:.3f}"
              f"  (z={t5['permutation_null']['z']:.2f}, "
              f"p={t5['permutation_null']['p_one_sided']:.4f})")
        print(f"  mean true rank obs={res['mean_true_rank']['observed']:.1f}  "
              f"null={res['mean_true_rank']['permutation_null']['null_mean']:.1f}"
              f"  (z={res['mean_true_rank']['permutation_null']['z']:.2f})")

        # closed form must match MC null mean
        assert abs(t5["shuffle_null_closed_form"]
                   - t5["permutation_null"]["null_mean"]) < 0.01, "null mismatch"
        # pattern-conditional model must beat its shuffle null
        assert t5["pattern_conditional_lift"] > 0 and \
            t5["permutation_null"]["p_one_sided"] < 0.05, "should be significant"

        # (b) flat prior-ranker: identical prediction for every pattern ->
        #     observed must equal the shuffle null (no pattern-conditional info)
        flat = np.tile(phat, (N, 1))
        resf = analyze(flat, labels, n_perm=500)
        t5f = resf["topk"][5]
        print("self-test (flat prior-ranker):")
        print(f"  Top-5 obs={t5f['observed']:.3f}  "
              f"null={t5f['permutation_null']['null_mean']:.3f}  "
              f"lift={t5f['pattern_conditional_lift']:.4f}")
        assert abs(t5f["pattern_conditional_lift"]) < 0.02, \
            "flat ranker must have ~zero pattern-conditional lift"
        print("\nALL SELF-TESTS PASSED")
