#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from bayes_ceiling import fit_channel, bayes_topk_exact


N_FAMILIES = 6
N_STATES = 3
FULL_SYMBOL_ALPHABET = N_STATES ** N_FAMILIES


def parse_args():
    p = argparse.ArgumentParser(description="Explicit two-stage Gate 2 decomposition on the matched continuous/degraded surface.")
    p.add_argument("--benchmark-h5", required=True)
    p.add_argument("--benchmark-npz", required=True)
    p.add_argument("--symbols-key", default="symbols_background_noise_like")
    p.add_argument("--stage2c-summary", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--output-md", required=True)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--pca-components", type=int, default=64)
    p.add_argument("--max-iter", type=int, default=4000)
    return p.parse_args()


def decode_symbol(sym: int, n_families: int = N_FAMILIES, n_states: int = N_STATES) -> np.ndarray:
    out = np.zeros(n_families, dtype=np.int64)
    x = int(sym)
    for i in range(n_families - 1, -1, -1):
        out[i] = x % n_states
        x //= n_states
    return out


def encode_symbol(states: np.ndarray, n_states: int = N_STATES) -> np.ndarray:
    states = np.asarray(states, dtype=np.int64)
    out = np.zeros(states.shape[0], dtype=np.int64)
    for j in range(states.shape[1]):
        out = out * n_states + states[:, j]
    return out


def topk_from_symbols(pred_symbols: np.ndarray, W: np.ndarray, prior: np.ndarray, ks=(1, 5)):
    joint = prior[:, None] * W
    pm = joint.sum(axis=0)
    post = np.zeros_like(joint)
    nz = pm > 0
    post[:, nz] = joint[:, nz] / pm[nz]
    if np.any(~nz):
        post[:, ~nz] = prior[:, None]
    scores = post[:, pred_symbols].T
    kmax = max(ks)
    top = np.argsort(-scores, axis=1)[:, :kmax]
    return {k: top[:, :k] for k in ks}


def accuracy_from_topk(topk: np.ndarray, labels: np.ndarray) -> float:
    truth = labels[:, None]
    return float((topk == truth).any(axis=1).mean())


def main():
    args = parse_args()
    with h5py.File(args.benchmark_h5, "r") as f:
        X = np.asarray(f["X_test"], dtype=np.float32)
    data = np.load(args.benchmark_npz, allow_pickle=True)
    labels = np.asarray(data["labels"], dtype=np.int64)
    symbols = np.asarray(data[args.symbols_key], dtype=np.int64)
    stage2c = json.loads(Path(args.stage2c_summary).read_text())

    true_states = np.stack([decode_symbol(int(s)) for s in symbols], axis=0)
    n_examples = int(X.shape[0])

    recovered_states = np.zeros_like(true_states)
    fold_assign = np.full(n_examples, -1, dtype=np.int64)

    cv = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    for fold_idx, (tr, te) in enumerate(cv.split(X)):
        fold_assign[te] = fold_idx
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr])
        Xte = scaler.transform(X[te])
        n_comp = max(2, min(args.pca_components, Xtr.shape[0] - 1, Xtr.shape[1]))
        pca = PCA(n_components=n_comp, random_state=args.seed)
        Ztr = pca.fit_transform(Xtr)
        Zte = pca.transform(Xte)

        for j in range(N_FAMILIES):
            ytr = true_states[tr, j]
            clf = LogisticRegression(
                solver="lbfgs",
                max_iter=args.max_iter,
            )
            clf.fit(Ztr, ytr)
            recovered_states[te, j] = clf.predict(Zte)

    recovered_symbols = encode_symbol(recovered_states)
    exact_symbol_acc = float(np.mean(recovered_symbols == symbols))
    family_state_acc = [float(np.mean(recovered_states[:, j] == true_states[:, j])) for j in range(N_FAMILIES)]
    all_family_acc = float(np.mean(np.all(recovered_states == true_states, axis=1)))

    n_classes = int(labels.max()) + 1
    W, prior = fit_channel(
        labels,
        symbols,
        n_classes=n_classes,
        n_symbols=FULL_SYMBOL_ALPHABET,
        alpha=1.0 / FULL_SYMBOL_ALPHABET,
    )
    ceiling = bayes_topk_exact(W, prior, ks=(1, 5))
    topk = topk_from_symbols(recovered_symbols, W, prior, ks=(1, 5))
    recovered_top1 = accuracy_from_topk(topk[1], labels)
    recovered_top5 = accuracy_from_topk(topk[5], labels)

    stage2c_top1 = float(stage2c["top1"])
    stage2c_top5 = float(stage2c["top5"])
    ceiling_top1 = float(ceiling["top1_bayes_exact"])
    ceiling_top5 = float(ceiling["top5_bayes_exact"])

    result = {
        "benchmark_h5": args.benchmark_h5,
        "benchmark_npz": args.benchmark_npz,
        "symbols_key": args.symbols_key,
        "n_examples": n_examples,
        "n_splits": int(args.n_splits),
        "seed": int(args.seed),
        "pca_components": int(args.pca_components),
        "absence_reader": {
            "exact_symbol_accuracy": exact_symbol_acc,
            "all_family_states_exact_accuracy": all_family_acc,
            "family_state_accuracy": family_state_acc,
            "unique_true_symbols": int(np.unique(symbols).size),
            "unique_recovered_symbols": int(np.unique(recovered_symbols).size),
        },
        "absence_to_eg_ceiling": {
            "top1": ceiling_top1,
            "top5": ceiling_top5,
        },
        "eg_given_recovered_absence": {
            "top1": recovered_top1,
            "top5": recovered_top5,
        },
        "stage2c_end_to_end": {
            "top1": stage2c_top1,
            "top5": stage2c_top5,
        },
        "decomposition_points": {
            "top1_upstream_total_gap": 100.0 * (ceiling_top1 - stage2c_top1),
            "top1_absence_recovery_loss": 100.0 * (ceiling_top1 - recovered_top1),
            "top1_residual_decoder_gap": 100.0 * (recovered_top1 - stage2c_top1),
            "top5_upstream_total_gap": 100.0 * (ceiling_top5 - stage2c_top5),
            "top5_absence_recovery_loss": 100.0 * (ceiling_top5 - recovered_top5),
            "top5_residual_decoder_gap": 100.0 * (recovered_top5 - stage2c_top5),
        },
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2) + "\n")

    fam_lines = "\n".join(
        f"- family {i}: `{100.0 * acc:.2f}%`" for i, acc in enumerate(family_state_acc)
    )
    md = f"""# Gate 2 Two-Stage Decomposition

- benchmark h5: `{args.benchmark_h5}`
- benchmark npz: `{args.benchmark_npz}`
- n examples: `{n_examples}`
- CV folds: `{args.n_splits}`

## Absence reader

- exact symbol accuracy: `{100.0 * exact_symbol_acc:.2f}%`
- all-family exact accuracy: `{100.0 * all_family_acc:.2f}%`
- unique true symbols: `{result["absence_reader"]["unique_true_symbols"]}`
- unique recovered symbols: `{result["absence_reader"]["unique_recovered_symbols"]}`

Per-family state accuracy:
{fam_lines}

## EG accuracies

- discrete absence->EG ceiling Top-1 / Top-5: `{100.0 * ceiling_top1:.2f}% / {100.0 * ceiling_top5:.2f}%`
- EG from recovered absences Top-1 / Top-5: `{100.0 * recovered_top1:.2f}% / {100.0 * recovered_top5:.2f}%`
- end-to-end `stage2c` Top-1 / Top-5: `{100.0 * stage2c_top1:.2f}% / {100.0 * stage2c_top5:.2f}%`

## Decomposition in percentage points

- Top-1 total upstream gap: `{result["decomposition_points"]["top1_upstream_total_gap"]:.2f}`
- Top-1 absence-recovery loss: `{result["decomposition_points"]["top1_absence_recovery_loss"]:.2f}`
- Top-1 residual decoder gap: `{result["decomposition_points"]["top1_residual_decoder_gap"]:.2f}`
- Top-5 total upstream gap: `{result["decomposition_points"]["top5_upstream_total_gap"]:.2f}`
- Top-5 absence-recovery loss: `{result["decomposition_points"]["top5_absence_recovery_loss"]:.2f}`
- Top-5 residual decoder gap: `{result["decomposition_points"]["top5_residual_decoder_gap"]:.2f}`
"""
    Path(args.output_md).write_text(md)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
