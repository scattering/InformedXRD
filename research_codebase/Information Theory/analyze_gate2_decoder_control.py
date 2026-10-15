#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from bayes_ceiling import bayes_topk_exact, fit_channel


def parse_args():
    p = argparse.ArgumentParser(description="Compare decoder accuracy to the discrete ceiling on the same simulated rung surface.")
    p.add_argument("--benchmark-npz", required=True)
    p.add_argument("--symbols-key", required=True)
    p.add_argument("--decoder-summary", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--output-md", required=True)
    p.add_argument("--alpha", type=float, default=0.0)
    return p.parse_args()


def main():
    args = parse_args()
    data = np.load(args.benchmark_npz, allow_pickle=True)
    labels = np.asarray(data["labels"], dtype=np.int64)
    symbols = np.asarray(data[args.symbols_key], dtype=np.int64)
    n_classes = int(labels.max()) + 1
    n_symbols = int(symbols.max()) + 1
    alpha = float(args.alpha) if args.alpha > 0 else 1.0 / n_symbols

    W, prior = fit_channel(labels, symbols, n_classes=n_classes, n_symbols=n_symbols, alpha=alpha)
    ceiling = bayes_topk_exact(W, prior, ks=(1, 3, 5))

    decoder = json.loads(Path(args.decoder_summary).read_text())
    result = {
        "benchmark_npz": args.benchmark_npz,
        "symbols_key": args.symbols_key,
        "decoder_summary": args.decoder_summary,
        "n_examples": int(labels.shape[0]),
        "support": int(np.unique(labels).size),
        "n_classes_dense": int(n_classes),
        "n_symbols": int(n_symbols),
        "alpha": float(alpha),
        "ceiling": ceiling,
        "decoder": decoder,
        "gap_top1_points": float(100.0 * (ceiling["top1_bayes_exact"] - decoder["top1"])),
        "gap_top5_points": float(100.0 * (ceiling["top5_bayes_exact"] - decoder["top5"])),
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2) + "\n")

    md = f"""# Gate 2 Decoder Control

- benchmark npz: `{args.benchmark_npz}`
- symbols key: `{args.symbols_key}`
- n examples: `{result["n_examples"]}`
- populated classes: `{result["support"]}`
- dense class count: `{result["n_classes_dense"]}`
- symbols: `{result["n_symbols"]}`

## Discrete ceiling on same surface

- Top-1: `{100.0 * ceiling["top1_bayes_exact"]:.2f}%`
- Top-3: `{100.0 * ceiling["top3_bayes_exact"]:.2f}%`
- Top-5: `{100.0 * ceiling["top5_bayes_exact"]:.2f}%`
- effective candidates: `{ceiling["effective_candidates"]:.3f}`

## Decoder

- Top-1: `{100.0 * decoder["top1"]:.2f}%`
- Top-5: `{100.0 * decoder["top5"]:.2f}%`
- mean max prob: `{decoder["mean_max_prob"]:.4f}`

## Gap

- Top-1 gap: `{result["gap_top1_points"]:.2f}` points
- Top-5 gap: `{result["gap_top5_points"]:.2f}` points
"""
    Path(args.output_md).write_text(md)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
