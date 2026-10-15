#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Plot grouped temperature-CV calibration sweeps.")
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-png", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.input_json, "r") as handle:
        data = json.load(handle)

    by_temp = defaultdict(lambda: defaultdict(list))
    for fold in data["fold_results"]:
        for row in fold["calibration_sweep"]:
            temp = float(row["temperature"])
            for metric in ("top1", "top5", "ece", "nll", "brier"):
                by_temp[temp][metric].append(float(row[metric]))

    temps = np.array(sorted(by_temp.keys()), dtype=float)
    metrics = {}
    for metric in ("top1", "top5", "ece", "nll", "brier"):
        means = np.array([np.mean(by_temp[t][metric]) for t in temps], dtype=float)
        stds = np.array([np.std(by_temp[t][metric], ddof=0) for t in temps], dtype=float)
        metrics[metric] = (means, stds)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    plot_specs = [
        ("nll", "Calibration NLL", "NLL", axes[0, 0]),
        ("ece", "Calibration ECE", "ECE", axes[0, 1]),
        ("top1", "Calibration Top-1", "Top-1 Accuracy", axes[1, 0]),
        ("top5", "Calibration Top-5", "Top-5 Accuracy", axes[1, 1]),
    ]

    for metric, title, ylabel, ax in plot_specs:
        means, stds = metrics[metric]
        ax.plot(temps, means, marker="o", linewidth=2)
        ax.fill_between(temps, means - stds, means + stds, alpha=0.2)
        ax.set_title(title)
        ax.set_xlabel("Temperature")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Grouped Mineral-Family CV Calibration Sweep", fontsize=14)
    output_path = Path(args.output_png)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    print(output_path)


if __name__ == "__main__":
    main()
