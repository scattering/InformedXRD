#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Bootstrap difficulty-controlled partition differential on RRUFF-209.")
    p.add_argument("--alphadiffract-full-json", required=True)
    p.add_argument("--stage2c-full-npz", required=True)
    p.add_argument("--subset-json", required=True, help="92-case no-MP-evidence subset JSON")
    p.add_argument("--n-bootstrap", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-json", required=True)
    p.add_argument("--output-md", required=True)
    return p.parse_args()


def top5_hits_from_probs(probs: np.ndarray, targets: np.ndarray) -> np.ndarray:
    top5 = np.argpartition(-probs, kth=4, axis=1)[:, :5]
    return (top5 == targets[:, None]).any(axis=1)


def percentile_ci(samples: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    lo = float(np.quantile(samples, alpha / 2.0))
    hi = float(np.quantile(samples, 1.0 - alpha / 2.0))
    return lo, hi


def summarize_partition(metric_a: np.ndarray, metric_b: np.ndarray, n_a: int, n_b: int):
    a = float(metric_a.mean())
    b = float(metric_b.mean())
    full = float((metric_a.sum() + metric_b.sum()) / (n_a + n_b))
    return a, b, full


def bootstrap_did(alpha_a, alpha_b, s2c_a, s2c_b, n_bootstrap: int, seed: int):
    rng = np.random.default_rng(seed)
    n_a = len(alpha_a["top1"])
    n_b = len(alpha_b["top1"])
    top1 = np.empty(n_bootstrap, dtype=np.float64)
    top5 = np.empty(n_bootstrap, dtype=np.float64)

    for i in range(n_bootstrap):
        ia = rng.integers(0, n_a, size=n_a)
        ib = rng.integers(0, n_b, size=n_b)

        aa1 = alpha_a["top1"][ia]
        aa5 = alpha_a["top5"][ia]
        ab1 = alpha_b["top1"][ib]
        ab5 = alpha_b["top5"][ib]

        sa1 = s2c_a["top1"][ia]
        sa5 = s2c_a["top5"][ia]
        sb1 = s2c_b["top1"][ib]
        sb5 = s2c_b["top5"][ib]

        alpha_a_top1, _, alpha_full_top1 = summarize_partition(aa1, ab1, n_a, n_b)
        alpha_a_top5, _, alpha_full_top5 = summarize_partition(aa5, ab5, n_a, n_b)
        s2c_a_top1, _, s2c_full_top1 = summarize_partition(sa1, sb1, n_a, n_b)
        s2c_a_top5, _, s2c_full_top5 = summarize_partition(sa5, sb5, n_a, n_b)

        top1[i] = (alpha_a_top1 - alpha_full_top1) - (s2c_a_top1 - s2c_full_top1)
        top5[i] = (alpha_a_top5 - alpha_full_top5) - (s2c_a_top5 - s2c_full_top5)

    return top1, top5


def main():
    args = parse_args()

    alpha_full = json.loads(Path(args.alphadiffract_full_json).read_text())
    subset = json.loads(Path(args.subset_json).read_text())
    subset_case_ids = {str(row["case_id"]) for row in subset["rows"]}

    alpha_examples = alpha_full["models"]["alphadiffract_209"]["examples"]
    alpha_by_index = {}
    case_id_to_index = {}
    for ex in alpha_examples:
        idx = int(ex["index"])
        case_id = str(ex["case_id"])
        top1 = bool(ex["correct"])
        top5 = int(ex["true_eg"]) in [int(v) for v in ex["top5_eg"]]
        alpha_by_index[idx] = {"top1": top1, "top5": top5, "case_id": case_id}
        case_id_to_index[case_id] = idx

    subset_indices = sorted(case_id_to_index[cid] for cid in subset_case_ids)
    subset_index_set = set(subset_indices)
    complement_indices = [i for i in sorted(alpha_by_index) if i not in subset_index_set]

    z = np.load(args.stage2c_full_npz, allow_pickle=False)
    s2c_probs = z["calibrated_probs"]
    s2c_targets = z["targets"]
    s2c_top1 = s2c_probs.argmax(axis=1) == s2c_targets
    s2c_top5 = top5_hits_from_probs(s2c_probs, s2c_targets)

    alpha_top1 = np.array([alpha_by_index[i]["top1"] for i in range(len(alpha_by_index))], dtype=bool)
    alpha_top5 = np.array([alpha_by_index[i]["top5"] for i in range(len(alpha_by_index))], dtype=bool)

    alpha_a = {
        "top1": alpha_top1[subset_indices],
        "top5": alpha_top5[subset_indices],
    }
    alpha_b = {
        "top1": alpha_top1[complement_indices],
        "top5": alpha_top5[complement_indices],
    }
    s2c_a = {
        "top1": s2c_top1[subset_indices],
        "top5": s2c_top5[subset_indices],
    }
    s2c_b = {
        "top1": s2c_top1[complement_indices],
        "top5": s2c_top5[complement_indices],
    }

    n_a = len(subset_indices)
    n_b = len(complement_indices)

    alpha_a_top1, alpha_b_top1, alpha_full_top1 = summarize_partition(alpha_a["top1"], alpha_b["top1"], n_a, n_b)
    alpha_a_top5, alpha_b_top5, alpha_full_top5 = summarize_partition(alpha_a["top5"], alpha_b["top5"], n_a, n_b)
    s2c_a_top1, s2c_b_top1, s2c_full_top1 = summarize_partition(s2c_a["top1"], s2c_b["top1"], n_a, n_b)
    s2c_a_top5, s2c_b_top5, s2c_full_top5 = summarize_partition(s2c_a["top5"], s2c_b["top5"], n_a, n_b)

    diff_in_diff_top1 = (alpha_a_top1 - alpha_full_top1) - (s2c_a_top1 - s2c_full_top1)
    diff_in_diff_top5 = (alpha_a_top5 - alpha_full_top5) - (s2c_a_top5 - s2c_full_top5)

    boot_top1, boot_top5 = bootstrap_did(alpha_a, alpha_b, s2c_a, s2c_b, args.n_bootstrap, args.seed)
    top1_ci = percentile_ci(boot_top1)
    top5_ci = percentile_ci(boot_top5)

    result = {
        "subset_name": subset["subset_name"],
        "n_subset": n_a,
        "n_complement": n_b,
        "n_total": n_a + n_b,
        "alphadiffract": {
            "full_top1": alpha_full_top1,
            "subset_top1": alpha_a_top1,
            "full_top5": alpha_full_top5,
            "subset_top5": alpha_a_top5,
            "delta_top1": alpha_a_top1 - alpha_full_top1,
            "delta_top5": alpha_a_top5 - alpha_full_top5,
        },
        "stage2c": {
            "full_top1": s2c_full_top1,
            "subset_top1": s2c_a_top1,
            "full_top5": s2c_full_top5,
            "subset_top5": s2c_a_top5,
            "delta_top1": s2c_a_top1 - s2c_full_top1,
            "delta_top5": s2c_a_top5 - s2c_full_top5,
        },
        "difference_in_differences": {
            "top1": diff_in_diff_top1,
            "top1_ci95": list(top1_ci),
            "top5": diff_in_diff_top5,
            "top5_ci95": list(top5_ci),
            "n_bootstrap": args.n_bootstrap,
            "seed": args.seed,
        },
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2) + "\n")

    lines = [
        f"# RRUFF-209 Partition Difference-in-Differences",
        "",
        f"- subset: `{subset['subset_name']}`",
        f"- n subset: `{n_a}`",
        f"- n complement: `{n_b}`",
        "",
        "## Top-1",
        f"- AlphaDiffract: `{alpha_full_top1*100:.2f}% -> {alpha_a_top1*100:.2f}%` (`{(alpha_a_top1-alpha_full_top1)*100:+.2f}` pts)",
        f"- stage2c: `{s2c_full_top1*100:.2f}% -> {s2c_a_top1*100:.2f}%` (`{(s2c_a_top1-s2c_full_top1)*100:+.2f}` pts)",
        f"- difference-in-differences: `{diff_in_diff_top1*100:+.2f}` pts",
        f"- bootstrap 95% CI: `[{top1_ci[0]*100:+.2f}, {top1_ci[1]*100:+.2f}]` pts",
        "",
        "## Top-5",
        f"- AlphaDiffract: `{alpha_full_top5*100:.2f}% -> {alpha_a_top5*100:.2f}%` (`{(alpha_a_top5-alpha_full_top5)*100:+.2f}` pts)",
        f"- stage2c: `{s2c_full_top5*100:.2f}% -> {s2c_a_top5*100:.2f}%` (`{(s2c_a_top5-s2c_full_top5)*100:+.2f}` pts)",
        f"- difference-in-differences: `{diff_in_diff_top5*100:+.2f}` pts",
        f"- bootstrap 95% CI: `[{top5_ci[0]*100:+.2f}, {top5_ci[1]*100:+.2f}]` pts",
        "",
    ]
    Path(args.output_md).write_text("\n".join(lines))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
