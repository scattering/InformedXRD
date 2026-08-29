#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np


def normalized_entropy(weights: np.ndarray) -> float:
    p = np.clip(weights, 1e-12, None)
    p = p / p.sum()
    entropy = -(p * np.log(p)).sum()
    return float(entropy / np.log(len(p)))


def concentration_metrics(attn: np.ndarray) -> dict:
    attn = np.asarray(attn, dtype=float)
    attn = np.clip(attn, 0.0, None)
    if attn.sum() <= 0:
        return {
            "norm_entropy": None,
            "top1pct_mass": None,
            "top5pct_mass": None,
            "tail80_mass": None,
            "effective_support_90pct": None,
        }
    p = attn / attn.sum()
    n = len(p)
    sort_desc = np.sort(p)[::-1]
    top1 = max(1, int(np.ceil(0.01 * n)))
    top5 = max(1, int(np.ceil(0.05 * n)))
    tail_start = int(np.floor(0.88 * n))
    csum = np.cumsum(sort_desc)
    support90 = int(np.searchsorted(csum, 0.90) + 1)
    return {
        "norm_entropy": normalized_entropy(p),
        "top1pct_mass": float(sort_desc[:top1].sum()),
        "top5pct_mass": float(sort_desc[:top5].sum()),
        "tail80_mass": float(p[tail_start:].sum()),
        "effective_support_90pct": support90,
    }


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-json", required=True)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-md")
    return ap.parse_args()


def main():
    args = parse_args()
    summary_path = Path(args.summary_json)
    with open(summary_path) as f:
        summary = json.load(f)

    rows = []
    groups = {"correct": [], "descendant_error": []}
    for row in summary:
        case_id = row["case_id"]
        mineral = case_id.split("__")[0]
        pred = int(row["topk_pred_egs"][0])
        true = int(row["true_eg"])
        label = "correct" if pred == true else "descendant_error"
        attn_path = Path(row["attention_npy"])
        if not attn_path.exists():
            attn_path = summary_path.parent / attn_path.name
        attn = np.load(attn_path)
        metrics = concentration_metrics(attn)
        out = {
            "sample_index": int(row["sample_index"]),
            "case_id": case_id,
            "mineral": mineral,
            "label": label,
            "true_eg": true,
            "top1_pred_eg": pred,
            **metrics,
        }
        rows.append(out)
        groups[label].append(out)

    group_summary = {}
    for label, items in groups.items():
        group_summary[label] = {"count": len(items)}
        for key in [
            "norm_entropy",
            "top1pct_mass",
            "top5pct_mass",
            "tail80_mass",
            "effective_support_90pct",
        ]:
            vals = [x[key] for x in items if x[key] is not None]
            group_summary[label][f"mean_{key}"] = float(np.mean(vals)) if vals else None

    output = {"cases": rows, "group_summary": group_summary}
    Path(args.output_json).write_text(json.dumps(output, indent=2))

    if args.output_md:
        lines = [
            "# Attention Concentration Summary",
            "",
            "This is a lightweight diagnostic on the 6-case attention panel.",
            "",
        ]
        for label in ["correct", "descendant_error"]:
            g = group_summary[label]
            lines.extend(
                [
                    f"## {label.replace('_', ' ').title()} ({g['count']} cases)",
                    "",
                    f"- Mean normalized entropy: {g['mean_norm_entropy']:.3f}",
                    f"- Mean top-1% mass: {g['mean_top1pct_mass']:.3f}",
                    f"- Mean top-5% mass: {g['mean_top5pct_mass']:.3f}",
                    f"- Mean high-angle tail mass: {g['mean_tail80_mass']:.3f}",
                    f"- Mean support to reach 90% mass: {g['mean_effective_support_90pct']:.1f} bins",
                    "",
                ]
            )
        Path(args.output_md).write_text("\n".join(lines))


if __name__ == "__main__":
    main()
