#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot Savitzky-Golay-smoothed derivatives for grouped temperature-CV sweeps."
    )
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-png", required=True)
    parser.add_argument("--step", type=float, default=0.25, help="Uniform temperature step for interpolation.")
    parser.add_argument("--window", type=int, default=21, help="Savitzky-Golay window length on the interpolated grid.")
    parser.add_argument("--polyorder", type=int, default=3)
    parser.add_argument("--x-min", type=float, default=None)
    parser.add_argument("--x-max", type=float, default=None)
    return parser.parse_args()


def load_means(input_json):
    with open(input_json, "r") as handle:
        data = json.load(handle)

    by_temp = defaultdict(lambda: defaultdict(list))
    for fold in data["fold_results"]:
        for row in fold["calibration_sweep"]:
            temp = float(row["temperature"])
            for metric in ("top1", "top5", "ece", "nll"):
                by_temp[temp][metric].append(float(row[metric]))

    temps = np.array(sorted(by_temp.keys()), dtype=float)
    means = {}
    for metric in ("top1", "top5", "ece", "nll"):
        means[metric] = np.array([np.mean(by_temp[t][metric]) for t in temps], dtype=float)
    return temps, means


def odd_at_most(n, limit):
    n = min(n, limit)
    if n % 2 == 0:
        n -= 1
    return max(n, 5)


def main():
    args = parse_args()
    temps, means = load_means(args.input_json)

    grid = np.arange(temps.min(), temps.max() + 0.5 * args.step, args.step)
    window = odd_at_most(args.window, len(grid))
    polyorder = min(args.polyorder, window - 2)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    plot_specs = [
        ("nll", "d(NLL)/dT", axes[0, 0]),
        ("ece", "d(ECE)/dT", axes[0, 1]),
        ("top1", "d(Top-1)/dT", axes[1, 0]),
        ("top5", "d(Top-5)/dT", axes[1, 1]),
    ]

    for metric, ylabel, ax in plot_specs:
        interp = np.interp(grid, temps, means[metric])
        smooth = savgol_filter(interp, window_length=window, polyorder=polyorder)
        deriv = savgol_filter(interp, window_length=window, polyorder=polyorder, deriv=1, delta=args.step)

        ax.plot(grid, deriv, linewidth=2)
        ax.axhline(0.0, color="black", linewidth=1, alpha=0.5)
        ax.set_title(metric.upper())
        ax.set_xlabel("Temperature")
        ax.set_ylabel(ylabel)
        if args.x_min is not None or args.x_max is not None:
            ax.set_xlim(left=args.x_min, right=args.x_max)
        ax.grid(True, alpha=0.3)

        # Mark the original sampled temperatures for reference.
        sample_deriv = np.interp(temps, grid, deriv)
        ax.scatter(temps, sample_deriv, s=18, zorder=3)

    fig.suptitle(
        f"Grouped Mineral-Family CV Derivatives (Savitzky-Golay, window={window}, poly={polyorder})",
        fontsize=14,
    )
    output_path = Path(args.output_png)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    print(output_path)


if __name__ == "__main__":
    main()
