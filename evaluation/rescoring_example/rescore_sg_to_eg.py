#!/usr/bin/env python3
"""Worked example: rescore a 230-way space-group model in extinction-group space.

This script demonstrates the full post-hoc rescoring path promised in the paper:

  1. load a model's per-pattern 230-way space-group probabilities,
  2. collapse them into the 99-way extinction-group label space using the
     released mapping table (``canonical_extinction_to_space_group.csv``),
  3. score Top-k accuracy against the benchmark's extinction-group labels,

and reproduces a reported results row end to end. With no arguments it scores
the PatchTST posterior on the 98-record opXRD Cu-Kalpha surface (released in
``case_studies/opxrd/xrdbench_eval/job_868916/PatchTST/posterior.npz``) and
reproduces the PatchTST row of the opXRD table: 10.20 / 20.41 / 34.69 Top-1/3/5.

The same collapse applies to any (N, 230) probability array: column ``j`` must
hold the probability of space group ``j + 1``, and labels must be
crystallographic extinction-group numbers 1..99 (canonical mapping row ``i``
corresponds to extinction-group number ``i + 1``).

Usage:
  python rescore_sg_to_eg.py                       # reproduce the PatchTST opXRD row
  python rescore_sg_to_eg.py --posterior my.npz \
      --mapping canonical_extinction_to_space_group.csv
"""
import argparse
import ast
import csv
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POSTERIOR = (
    REPO_ROOT
    / "case_studies/opxrd/xrdbench_eval/job_868916/PatchTST/posterior.npz"
)
DEFAULT_MAPPING = REPO_ROOT / "assets/lookups/canonical_extinction_to_space_group.csv"


def load_mapping(mapping_csv: Path) -> list[list[int]]:
    """Return, for each 0-based extinction-group index, its space-group numbers."""
    rows: list[tuple[int, list[int]]] = []
    with open(mapping_csv, newline="") as f:
        for row in csv.DictReader(f):
            idx = int(row["Index"])
            sgs = ast.literal_eval(row["Space Group Numbers"])
            rows.append((idx, [int(s) for s in sgs]))
    rows.sort()
    assert [i for i, _ in rows] == list(range(len(rows))), "non-contiguous mapping indices"
    covered = sorted(sg for _, sgs in rows for sg in sgs)
    assert covered == list(range(1, 231)), "mapping must cover each space group exactly once"
    return [sgs for _, sgs in rows]


def collapse_sg_to_eg(sg_probs: np.ndarray, mapping: list[list[int]]) -> np.ndarray:
    """Sum 230-way space-group probabilities into the 99-way extinction-group space."""
    n_eg = len(mapping)
    eg = np.zeros((sg_probs.shape[0], n_eg), dtype=np.float64)
    for eg_idx, sgs in enumerate(mapping):
        eg[:, eg_idx] = sg_probs[:, [sg - 1 for sg in sgs]].sum(axis=1)
    return eg


def topk_accuracy(eg_probs: np.ndarray, targets_eg_number: np.ndarray, ks=(1, 3, 5)) -> dict:
    order = np.argsort(-eg_probs, axis=1) + 1  # columns -> extinction-group numbers 1..99
    return {
        f"top{k}": float(np.mean([t in order[i, :k] for i, t in enumerate(targets_eg_number)]))
        for k in ks
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--posterior", type=Path, default=DEFAULT_POSTERIOR,
                    help="npz with sg_probs (N,230) and targets_eg_number (N,)")
    ap.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING,
                    help="canonical_extinction_to_space_group.csv")
    args = ap.parse_args()

    data = np.load(args.posterior, allow_pickle=True)
    sg_probs = np.asarray(data["sg_probs"], dtype=np.float64)
    targets = np.asarray(data["targets_eg_number"], dtype=int)

    mapping = load_mapping(args.mapping)
    eg_probs = collapse_sg_to_eg(sg_probs, mapping)
    scores = topk_accuracy(eg_probs, targets)

    report = {
        "posterior": str(args.posterior),
        "n_patterns": int(sg_probs.shape[0]),
        "topk_percent": {k: round(100.0 * v, 2) for k, v in scores.items()},
    }
    if "eg_probs" in data:
        stored = np.asarray(data["eg_probs"], dtype=np.float64)
        report["matches_released_eg_probs"] = bool(
            np.allclose(stored / stored.sum(axis=1, keepdims=True),
                        eg_probs / eg_probs.sum(axis=1, keepdims=True), atol=1e-5)
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
