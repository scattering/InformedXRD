#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Plot CNN convergence as a standalone SVG.")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--log", action="append", default=[])
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_records(summary_path, log_paths):
    summary = json.loads(Path(summary_path).read_text())
    records = {int(row["epoch"]): row for row in summary["history"]}
    for log_path in log_paths:
        for line in Path(log_path).read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and {"epoch", "train", "val"} <= row.keys():
                records[int(row["epoch"])] = row
    return [records[epoch] for epoch in sorted(records)]


def esc(text):
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def line(points, color, width=3, dash=None):
    attrs = [
        f'points="{" ".join(f"{x:.2f},{y:.2f}" for x, y in points)}"',
        'fill="none"',
        f'stroke="{color}"',
        f'stroke-width="{width}"',
        'stroke-linejoin="round"',
        'stroke-linecap="round"',
    ]
    if dash:
        attrs.append(f'stroke-dasharray="{dash}"')
    return f"<polyline {' '.join(attrs)}/>"


def circles(points, color, radius=3):
    return "".join(
        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius}" fill="{color}"/>'
        for x, y in points
    )


def nice_bounds(values, fraction=0.08):
    low, high = min(values), max(values)
    pad = max((high - low) * fraction, abs(high) * 0.005, 1e-6)
    return low - pad, high + pad


def render(records, output_path):
    width, height = 1200, 900
    left, right = 105, 35
    plot_width = width - left - right
    panel_height = 205
    panel_tops = [115, 390, 665]
    epochs = [int(row["epoch"]) for row in records]
    epoch_min, epoch_max = min(epochs), max(epochs)
    span = max(epoch_max - epoch_min, 1)

    train_loss = [float(row["train"]["loss"]) for row in records]
    val_loss = [float(row["val"]["loss"]) for row in records]
    train_top1 = [float(row["train"]["top1"]) for row in records]
    val_top1 = [float(row["val"]["top1"]) for row in records]
    val_top5 = [float(row["val"]["top5"]) for row in records]
    first_recorded_lr = next(
        float(row["learning_rate"]) for row in records if "learning_rate" in row
    )
    learning_rates = [
        float(row.get("learning_rate", first_recorded_lr)) for row in records
    ]
    best_index = min(range(len(records)), key=lambda i: val_loss[i])
    best_epoch = epochs[best_index]

    def x_of(epoch):
        return left + (epoch - epoch_min) / span * plot_width

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>',
        'text{font-family:Arial,Helvetica,sans-serif;fill:#20242a;letter-spacing:0}',
        '.title{font-size:27px;font-weight:700}.subtitle{font-size:15px;fill:#59636e}',
        '.axis{font-size:13px;fill:#59636e}.label{font-size:16px;font-weight:600}',
        '.legend{font-size:14px}.note{font-size:13px;fill:#59636e}',
        '</style>',
        '<text x="105" y="42" class="title">1D residual CNN convergence audit</text>',
        (
            f'<text x="105" y="70" class="subtitle">Epochs {epoch_min}-{epoch_max}; '
            f'best validation loss at epoch {best_epoch}. '
            'Checkpoint selection uses synthetic validation only.</text>'
        ),
    ]

    def panel(top, label, series, bounds, ticks, legend, show_best=False):
        low, high = bounds
        bottom = top + panel_height

        def y_of(value):
            return bottom - (value - low) / (high - low) * panel_height

        parts.append(f'<rect x="{left}" y="{top}" width="{plot_width}" height="{panel_height}" fill="#fbfcfd"/>')
        for tick in ticks:
            y = y_of(tick)
            parts.append(
                f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" '
                'stroke="#dce1e6" stroke-width="1"/>'
            )
            parts.append(f'<text x="{left - 12}" y="{y + 5:.2f}" text-anchor="end" class="axis">{tick:g}</text>')
        for epoch in range(5, epoch_max + 1, 5):
            if epoch < epoch_min:
                continue
            x = x_of(epoch)
            parts.append(
                f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{bottom}" '
                'stroke="#edf0f2" stroke-width="1"/>'
            )
            parts.append(f'<text x="{x:.2f}" y="{bottom + 22}" text-anchor="middle" class="axis">{epoch}</text>')
        if show_best:
            best_x = x_of(best_epoch)
            parts.append(
                f'<line x1="{best_x:.2f}" y1="{top}" x2="{best_x:.2f}" y2="{bottom}" '
                'stroke="#33383f" stroke-width="1.5" stroke-dasharray="6 5"/>'
            )
            parts.append(
                f'<text x="{best_x - 7:.2f}" y="{top + 18}" text-anchor="end" '
                f'class="note">best epoch {best_epoch}</text>'
            )
        for values, color in series:
            points = [(x_of(epoch), y_of(value)) for epoch, value in zip(epochs, values)]
            parts.append(line(points, color))
            parts.append(circles(points, color, 2.2))
        parts.append(
            f'<text transform="translate(30 {top + panel_height / 2}) rotate(-90)" '
            f'text-anchor="middle" class="label">{esc(label)}</text>'
        )
        legend_width = 170 * len(legend)
        parts.append(
            f'<rect x="{left + 8}" y="{top + 5}" width="{legend_width}" height="28" '
            'fill="#fbfcfd" opacity="0.94"/>'
        )
        legend_x = left + 15
        for name, color in legend:
            parts.append(
                f'<line x1="{legend_x}" y1="{top + 23}" x2="{legend_x + 28}" y2="{top + 23}" '
                f'stroke="{color}" stroke-width="4"/>'
            )
            parts.append(f'<text x="{legend_x + 36}" y="{top + 28}" class="legend">{esc(name)}</text>')
            legend_x += 165

    loss_low, loss_high = nice_bounds(train_loss + val_loss)
    loss_ticks = [
        round(loss_low + i * (loss_high - loss_low) / 4, 2)
        for i in range(5)
    ]
    panel(
        panel_tops[0],
        "Cross-entropy loss",
        [(train_loss, "#177e89"), (val_loss, "#d95f02")],
        (loss_low, loss_high),
        loss_ticks,
        [("training", "#177e89"), ("validation", "#d95f02")],
        show_best=True,
    )

    acc_low, acc_high = nice_bounds(train_top1 + val_top1 + val_top5)
    acc_ticks = [
        round(acc_low + i * (acc_high - acc_low) / 4, 1)
        for i in range(5)
    ]
    panel(
        panel_tops[1],
        "Accuracy (%)",
        [
            (train_top1, "#177e89"),
            (val_top1, "#d95f02"),
            (val_top5, "#6a51a3"),
        ],
        (acc_low, acc_high),
        acc_ticks,
        [
            ("train Top-1", "#177e89"),
            ("validation Top-1", "#d95f02"),
            ("validation Top-5", "#6a51a3"),
        ],
    )

    lr_logs = [math.log10(value) for value in learning_rates]
    lr_low, lr_high = nice_bounds(lr_logs, fraction=0.12)
    lr_ticks_values = sorted(set(learning_rates), reverse=True)
    top = panel_tops[2]
    bottom = top + panel_height

    def y_lr(value):
        return bottom - (math.log10(value) - lr_low) / (lr_high - lr_low) * panel_height

    parts.append(f'<rect x="{left}" y="{top}" width="{plot_width}" height="{panel_height}" fill="#fbfcfd"/>')
    for tick in lr_ticks_values:
        y = y_lr(tick)
        parts.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" '
            'stroke="#dce1e6" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{left - 12}" y="{y + 5:.2f}" text-anchor="end" class="axis">{tick:.0e}</text>'
        )
    for epoch in range(5, epoch_max + 1, 5):
        if epoch < epoch_min:
            continue
        x = x_of(epoch)
        parts.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{bottom}" '
            'stroke="#edf0f2" stroke-width="1"/>'
        )
        parts.append(f'<text x="{x:.2f}" y="{bottom + 22}" text-anchor="middle" class="axis">{epoch}</text>')
    lr_points = [(x_of(epoch), y_lr(rate)) for epoch, rate in zip(epochs, learning_rates)]
    parts.append(line(lr_points, "#4d4d4d", width=3))
    parts.append(circles(lr_points, "#4d4d4d", radius=2.2))
    parts.append(
        f'<text transform="translate(30 {top + panel_height / 2}) rotate(-90)" '
        'text-anchor="middle" class="label">Learning rate (log scale)</text>'
    )
    parts.append(
        f'<rect x="{left + 8}" y="{top + 5}" width="220" height="28" '
        'fill="#fbfcfd" opacity="0.94"/>'
    )
    parts.append(
        f'<line x1="{left + 15}" y1="{top + 23}" x2="{left + 43}" y2="{top + 23}" '
        'stroke="#4d4d4d" stroke-width="4"/>'
    )
    parts.append(f'<text x="{left + 51}" y="{top + 28}" class="legend">AdamW learning rate</text>')
    parts.append(
        f'<text x="{left + plot_width / 2}" y="{height - 8}" text-anchor="middle" class="label">Epoch</text>'
    )
    parts.append("</svg>")
    Path(output_path).write_text("\n".join(parts) + "\n")


def main():
    args = parse_args()
    records = load_records(args.summary, args.log)
    render(records, args.output)


if __name__ == "__main__":
    main()
