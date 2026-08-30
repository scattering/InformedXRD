#!/usr/bin/env python3
"""Evaluate an XRDBench SG model on a physically matched RRUFF grid."""

from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path

import h5py
import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval-h5", required=True)
    parser.add_argument("--mapping-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-npz", default="")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--grid-min", type=float, default=10.0)
    parser.add_argument("--grid-max", type=float, default=80.0)
    parser.add_argument("--grid-step", type=float, default=0.02)
    parser.add_argument(
        "--target-label-base",
        type=int,
        choices=(0, 1),
        default=1,
        help="Integer base used by y_test EG labels. The RRUFF benchmark files use 1..99.",
    )
    parser.add_argument(
        "--input-target-max",
        type=float,
        default=100.0,
        help=(
            "Normalize each resampled model-input pattern to this maximum. "
            "The SimXRD paper specifies a maximum of 100."
        ),
    )
    parser.add_argument(
        "--normalization-mode",
        choices=("target-max", "preserve"),
        default="target-max",
        help=(
            "Use target-max to independently rescale each aligned pattern, or "
            "preserve to retain the stored RRUFF intensity scale after grid selection."
        ),
    )
    return parser.parse_args()


def load_sg_to_eg_map(mapping_csv: str) -> np.ndarray:
    sg_to_eg = np.full(231, -1, dtype=np.int64)
    seen_eg_indices = set()
    with open(mapping_csv, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            eg_index = int(row["Index"])
            if eg_index in seen_eg_indices:
                raise ValueError(f"Duplicate EG index in mapping: {eg_index}")
            seen_eg_indices.add(eg_index)
            for sg in ast.literal_eval(row["Space Group Numbers"]):
                sg = int(sg)
                if not 1 <= sg <= 230:
                    raise ValueError(f"SG number outside 1..230 in mapping: {sg}")
                if sg_to_eg[sg] >= 0:
                    raise ValueError(f"SG {sg} appears in more than one EG")
                sg_to_eg[sg] = eg_index
    if seen_eg_indices != set(range(99)):
        raise ValueError(
            "Expected canonical EG indices 0..98; "
            f"found {sorted(seen_eg_indices)}"
        )
    if np.any(sg_to_eg[1:] < 0):
        missing = np.where(sg_to_eg[1:] < 0)[0] + 1
        raise ValueError(f"Missing SG->EG mapping for SGs: {missing.tolist()}")
    return sg_to_eg


def common_grid(grid_min: float, grid_max: float, grid_step: float) -> np.ndarray:
    if grid_step <= 0 or grid_max <= grid_min:
        raise ValueError("Grid requires grid_step > 0 and grid_max > grid_min")
    raw_intervals = (grid_max - grid_min) / grid_step
    intervals = int(round(raw_intervals))
    if not np.isclose(raw_intervals, intervals, atol=1e-10, rtol=0):
        raise ValueError(
            f"Grid span is not an integer number of steps: {raw_intervals}"
        )
    grid = grid_min + np.arange(intervals + 1, dtype=np.float64) * grid_step
    if grid.shape[0] != intervals + 1:
        raise AssertionError("Inclusive target-grid construction has the wrong length")
    if not np.isclose(grid[0], grid_min, atol=1e-10, rtol=0):
        raise ValueError(f"Target grid starts at {grid[0]}, expected {grid_min}")
    if not np.isclose(grid[-1], grid_max, atol=1e-10, rtol=0):
        raise ValueError(f"Target grid ends at {grid[-1]}, expected {grid_max}")
    return grid


def convert_eg_targets(raw_targets: np.ndarray, label_base: int) -> np.ndarray:
    raw_targets = np.asarray(raw_targets, dtype=np.int64).reshape(-1)
    expected_min = label_base
    expected_max = label_base + 98
    if raw_targets.size == 0:
        raise ValueError("Evaluation target array is empty")
    if raw_targets.min() < expected_min or raw_targets.max() > expected_max:
        raise ValueError(
            f"Raw EG labels must be in {expected_min}..{expected_max}; "
            f"observed {raw_targets.min()}..{raw_targets.max()}"
        )
    targets = raw_targets - label_base
    if targets.min() < 0 or targets.max() > 98:
        raise AssertionError("Converted EG targets are not valid posterior columns 0..98")
    return targets


def normalize_patterns(
    patterns: np.ndarray,
    target_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    if patterns.ndim != 2:
        raise ValueError(f"Patterns must be 2-D; found shape {patterns.shape}")
    if target_max <= 0:
        raise ValueError("Input-pattern target maximum must be positive")
    row_maxima = patterns.max(axis=1)
    if not np.isfinite(row_maxima).all() or np.any(row_maxima <= 0):
        raise ValueError("Each source pattern must have a finite, positive maximum")
    normalized = patterns * (np.float32(target_max) / row_maxima[:, None])
    normalized = normalized.astype(np.float32, copy=False)
    if not np.allclose(
        normalized.max(axis=1),
        target_max,
        atol=1e-4,
        rtol=1e-6,
    ):
        raise AssertionError("Input-pattern maximum normalization failed")
    return normalized, row_maxima


def select_aligned_patterns(
    patterns: np.ndarray,
    source_grid: np.ndarray,
    target_grid: np.ndarray,
) -> tuple[np.ndarray, dict]:
    if patterns.ndim != 2:
        raise ValueError(f"Patterns must be 2-D; found shape {patterns.shape}")
    if target_grid.ndim != 1 or np.any(np.diff(target_grid) <= 0):
        raise ValueError("Target two-theta grid must be one-dimensional and increasing")
    if source_grid.ndim == 1:
        if source_grid.shape[0] != patterns.shape[1]:
            raise ValueError(
                f"Source grid shape {source_grid.shape} does not match patterns {patterns.shape}"
            )
        source_grids = np.broadcast_to(source_grid, patterns.shape)
    elif source_grid.shape == patterns.shape:
        source_grids = source_grid
    else:
        raise ValueError(
            f"Source grid shape {source_grid.shape} does not match patterns {patterns.shape}"
        )
    if np.any(np.diff(source_grids, axis=1) <= 0):
        raise ValueError("Source two-theta grid must be strictly increasing")
    if (
        target_grid[0] < source_grids[:, 0].max()
        or target_grid[-1] > source_grids[:, -1].min()
    ):
        raise ValueError("Target grid extends beyond the source two-theta support")

    reference_grid = source_grids[0]
    right = np.searchsorted(reference_grid, target_grid, side="left")
    if np.any(right <= 0) or np.any(right >= reference_grid.shape[0]):
        raise ValueError("Could not bracket every target coordinate on the source grid")
    left = right - 1
    use_right = (
        np.abs(reference_grid[right] - target_grid)
        < np.abs(reference_grid[left] - target_grid)
    )
    source_indices = np.where(use_right, right, left)
    index_steps = np.diff(source_indices)
    if np.any(index_steps <= 0) or not np.all(index_steps == index_steps[0]):
        raise ValueError(
            "Target grid is not a uniform integer decimation of the source grid"
        )

    selected_grids = source_grids[:, source_indices]
    max_mismatch = float(np.abs(selected_grids - target_grid[None, :]).max())
    if max_mismatch > 1e-5:
        raise ValueError(
            "Target coordinates are not present on the stored source grid; "
            f"maximum mismatch is {max_mismatch} degrees"
        )

    result = patterns[:, source_indices].astype(np.float32, copy=True)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite intensity encountered after aligned-grid selection")
    source_steps = np.diff(source_grids, axis=1)
    diagnostics = {
        "method": "exact stored-grid crop and integer decimation; no interpolation",
        "source_grid_step_min": float(source_steps.min()),
        "source_grid_step_max": float(source_steps.max()),
        "source_start_index": int(source_indices[0]),
        "source_end_index": int(source_indices[-1]),
        "source_index_stride": int(index_steps[0]),
        "max_target_coordinate_mismatch_degrees": max_mismatch,
    }
    return result, diagnostics


def collapse_logits_to_eg_probs(logits: np.ndarray, sg_to_eg: np.ndarray) -> np.ndarray:
    if logits.ndim != 2 or logits.shape[1] != 230:
        raise ValueError(f"Expected SG logits with shape (N, 230); found {logits.shape}")
    if sg_to_eg.shape != (231,) or set(sg_to_eg[1:].tolist()) != set(range(99)):
        raise ValueError("SG->EG mapping must cover SG 1..230 and EG indices 0..98")
    sg_probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    eg_probs = np.zeros((sg_probs.shape[0], 99), dtype=np.float32)
    for sg in range(1, 231):
        eg_probs[:, sg_to_eg[sg]] += sg_probs[:, sg - 1]
    if not np.allclose(eg_probs.sum(axis=1), 1.0, atol=1e-6, rtol=1e-6):
        raise AssertionError("SG-to-EG aggregation did not preserve posterior probability")
    return eg_probs


def topk_accuracy(probs: np.ndarray, targets: np.ndarray, k: int) -> float:
    k = min(k, probs.shape[1])
    topk = np.argpartition(-probs, kth=k - 1, axis=1)[:, :k]
    return float((topk == targets[:, None]).any(axis=1).mean())


def main():
    args = parse_args()
    sg_to_eg = load_sg_to_eg_map(args.mapping_csv)
    target_grid = common_grid(args.grid_min, args.grid_max, args.grid_step)

    with h5py.File(args.eval_h5, "r") as handle:
        source_patterns = np.asarray(handle["X_test"], dtype=np.float32)
        source_grid = np.asarray(handle["two_theta"], dtype=np.float64)
        raw_targets = np.asarray(handle["y_test"], dtype=np.int64).reshape(-1)
        case_ids = [
            item.decode() if isinstance(item, bytes) else str(item)
            for item in handle.get(
                "case_ids", np.array([f"row_{i}" for i in range(len(raw_targets))])
            )[:]
        ]
    if source_patterns.shape[0] != len(raw_targets) or len(case_ids) != len(raw_targets):
        raise ValueError(
            "Pattern, target, and case-ID counts differ: "
            f"{source_patterns.shape[0]}, {len(raw_targets)}, {len(case_ids)}"
        )
    targets = convert_eg_targets(raw_targets, args.target_label_base)

    selected_patterns, grid_selection = select_aligned_patterns(
        source_patterns,
        source_grid,
        target_grid,
    )
    selected_row_maxima = selected_patterns.max(axis=1)
    if args.normalization_mode == "target-max":
        patterns, selected_row_maxima = normalize_patterns(
            selected_patterns,
            args.input_target_max,
        )
        normalization_description = (
            "each 3501-point pattern independently max-normalized after "
            "10..80 degree aligned-grid selection"
        )
        applied_input_target_max = float(args.input_target_max)
    else:
        patterns = selected_patterns.astype(np.float32, copy=True)
        normalization_description = (
            "stored RRUFF intensity scale preserved after 10..80 degree "
            "aligned-grid selection"
        )
        applied_input_target_max = None
    device = torch.device(args.device)
    model = torch.load(args.checkpoint, map_location="cpu", weights_only=False).to(device)
    model.eval()

    logits = []
    with torch.no_grad():
        for start in range(0, len(patterns), args.batch_size):
            batch = torch.from_numpy(patterns[start : start + args.batch_size])
            logits.append(model(batch.unsqueeze(1).to(device)).cpu().numpy())
    logits_arr = np.concatenate(logits, axis=0)
    eg_probs = collapse_logits_to_eg_probs(logits_arr, sg_to_eg)
    top1 = np.argmax(eg_probs, axis=1)
    top5 = np.argsort(-eg_probs, axis=1)[:, :5]

    result = {
        "checkpoint": args.checkpoint,
        "eval_h5": args.eval_h5,
        "n": int(len(targets)),
        "label_conventions": {
            "model_output": "230 SG logits; column 0 is SG 1 and column 229 is SG 230",
            "aggregation": "softmax SG probabilities summed into canonical EG indices",
            "raw_target_label_base": int(args.target_label_base),
            "raw_target_min_observed": int(raw_targets.min()),
            "raw_target_max_observed": int(raw_targets.max()),
            "posterior_column_base": 0,
            "posterior_columns": "0..98 correspond to crystallographic EG numbers 1..99",
            "examples_eg_label_base": 1,
        },
        "preprocessing": {
            "source_points": int(source_grid.shape[-1]),
            "source_grid_min": float(source_grid.min()),
            "source_grid_max": float(source_grid.max()),
            "target_points": int(target_grid.shape[0]),
            "target_grid_min": float(target_grid[0]),
            "target_grid_max": float(target_grid[-1]),
            "target_grid_step": float(args.grid_step),
            "grid_selection": grid_selection,
            "intensity_normalization": normalization_description,
            "normalization_mode": args.normalization_mode,
            "input_target_max": applied_input_target_max,
            "source_intensity_min": float(source_patterns.min()),
            "source_intensity_max": float(source_patterns.max()),
            "source_row_max_min": float(source_patterns.max(axis=1).min()),
            "source_row_max_max": float(source_patterns.max(axis=1).max()),
            "selected_row_max_before_normalization_min": float(
                selected_row_maxima.min()
            ),
            "selected_row_max_before_normalization_max": float(
                selected_row_maxima.max()
            ),
            "converted_intensity_min": float(patterns.min()),
            "converted_intensity_max": float(patterns.max()),
            "converted_row_max_min": float(patterns.max(axis=1).min()),
            "converted_row_max_max": float(patterns.max(axis=1).max()),
        },
        "top1_eg": topk_accuracy(eg_probs, targets, 1),
        "top3_eg": topk_accuracy(eg_probs, targets, 3),
        "top5_eg": topk_accuracy(eg_probs, targets, 5),
        "examples": [
            {
                "case_id": case_ids[i],
                "true_eg": int(targets[i]) + 1,
                "pred_eg": int(top1[i]) + 1,
                "top5_eg": [int(value) + 1 for value in top5[i]],
                "correct": bool(top1[i] == targets[i]),
            }
            for i in range(len(targets))
        ],
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2) + "\n")
    if args.output_npz:
        output_npz = Path(args.output_npz)
        output_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_npz,
            eg_probs=eg_probs.astype(np.float32),
            targets_eg_index=targets.astype(np.int64),
            targets_eg_number=(targets + 1).astype(np.int64),
            case_ids=np.asarray(case_ids, dtype=object),
            top1_pred_eg_index=top1.astype(np.int64),
            top1_pred_eg_number=(top1 + 1).astype(np.int64),
            top5_pred_eg_index=top5.astype(np.int64),
            top5_pred_eg_number=(top5 + 1).astype(np.int64),
            two_theta=target_grid.astype(np.float32),
        )
    print(json.dumps({key: value for key, value in result.items() if key != "examples"}, indent=2))


if __name__ == "__main__":
    main()
