#!/usr/bin/env python3
"""Evaluate a SimXRD-trained XRDBench SG model on the opXRD Cu-Kalpha surface."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from eval_xrdbench_sg_model_rruff_common_grid import (
    collapse_logits_to_eg_probs,
    common_grid,
    load_sg_to_eg_map,
    normalize_patterns,
    topk_accuracy,
)


REQUIRED_KEYS = {
    "intensity",
    "two_theta",
    "record",
    "sg",
    "eg",
    "community",
    "community_distance",
    "community_outlier",
    "community_assigned",
    "wavelength_mode",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval-npz", required=True)
    parser.add_argument("--mapping-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-npz", default="")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--grid-min", type=float, default=10.0)
    parser.add_argument("--grid-max", type=float, default=80.0)
    parser.add_argument("--grid-step", type=float, default=0.02)
    parser.add_argument("--input-target-max", type=float, default=100.0)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def topk_indices(probabilities: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(-probabilities, axis=1)[:, :k]


def select_exact_aligned_patterns(
    patterns: np.ndarray,
    source_grid: np.ndarray,
    target_grid: np.ndarray,
) -> tuple[np.ndarray, dict]:
    if patterns.ndim != 2:
        raise ValueError(f"Patterns must be two-dimensional: {patterns.shape}")
    if source_grid.ndim != 1 or source_grid.shape[0] != patterns.shape[1]:
        raise ValueError(
            f"Source grid shape {source_grid.shape} does not match {patterns.shape}"
        )
    if target_grid.ndim != 1:
        raise ValueError(f"Target grid must be one-dimensional: {target_grid.shape}")
    if np.any(np.diff(source_grid) <= 0) or np.any(np.diff(target_grid) <= 0):
        raise ValueError("Source and target grids must be strictly increasing")
    if target_grid[0] < source_grid[0] or target_grid[-1] > source_grid[-1]:
        raise ValueError("Target grid extends beyond the measured source support")

    right = np.searchsorted(source_grid, target_grid, side="left")
    right = np.clip(right, 0, source_grid.shape[0] - 1)
    left = np.clip(right - 1, 0, source_grid.shape[0] - 1)
    use_right = (
        np.abs(source_grid[right] - target_grid)
        <= np.abs(source_grid[left] - target_grid)
    )
    source_indices = np.where(use_right, right, left)
    index_steps = np.diff(source_indices)
    if np.any(index_steps <= 0) or not np.all(index_steps == index_steps[0]):
        raise ValueError(
            "Target grid is not a uniform integer decimation of the source grid"
        )
    selected_grid = source_grid[source_indices]
    max_mismatch = float(np.max(np.abs(selected_grid - target_grid)))
    if max_mismatch > 1e-5:
        raise ValueError(
            "Target coordinates are absent from the stored source grid; "
            f"maximum mismatch is {max_mismatch} degrees"
        )

    selected = patterns[:, source_indices].astype(np.float32, copy=True)
    if not np.all(np.isfinite(selected)):
        raise ValueError("Non-finite intensity encountered after grid selection")
    source_steps = np.diff(source_grid)
    return selected, {
        "method": "exact stored-grid crop and integer decimation; no interpolation",
        "source_grid_step_min": float(source_steps.min()),
        "source_grid_step_max": float(source_steps.max()),
        "source_start_index": int(source_indices[0]),
        "source_end_index": int(source_indices[-1]),
        "source_index_stride": int(index_steps[0]),
        "max_target_coordinate_mismatch_degrees": max_mismatch,
    }


def metric_block(
    mask: np.ndarray,
    sg_probs: np.ndarray,
    eg_probs: np.ndarray,
    sg_targets: np.ndarray,
    eg_targets: np.ndarray,
) -> dict:
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (len(sg_targets),):
        raise ValueError(f"Stratum mask has invalid shape {mask.shape}")
    n = int(mask.sum())
    if n == 0:
        return {"n": 0}
    return {
        "n": n,
        "unique_sg": int(np.unique(sg_targets[mask]).size),
        "unique_eg": int(np.unique(eg_targets[mask]).size),
        "top1_sg": topk_accuracy(sg_probs[mask], sg_targets[mask], 1),
        "top3_sg": topk_accuracy(sg_probs[mask], sg_targets[mask], 3),
        "top5_sg": topk_accuracy(sg_probs[mask], sg_targets[mask], 5),
        "top1_eg": topk_accuracy(eg_probs[mask], eg_targets[mask], 1),
        "top3_eg": topk_accuracy(eg_probs[mask], eg_targets[mask], 3),
        "top5_eg": topk_accuracy(eg_probs[mask], eg_targets[mask], 5),
    }


def main() -> int:
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    eval_npz = Path(args.eval_npz)
    mapping_csv = Path(args.mapping_csv)

    with np.load(eval_npz, allow_pickle=False) as source:
        missing = REQUIRED_KEYS - set(source.files)
        if missing:
            raise ValueError(f"opXRD NPZ is missing required keys: {sorted(missing)}")
        patterns = np.asarray(source["intensity"], dtype=np.float32)
        source_grid = np.asarray(source["two_theta"], dtype=np.float64)
        records = np.asarray(source["record"], dtype=str)
        sg_numbers = np.asarray(source["sg"], dtype=np.int64)
        eg_numbers = np.asarray(source["eg"], dtype=np.int64)
        communities = np.asarray(source["community"], dtype=np.int64)
        community_distances = np.asarray(
            source["community_distance"],
            dtype=np.float32,
        )
        community_outlier = np.asarray(source["community_outlier"], dtype=bool)
        community_assigned = np.asarray(source["community_assigned"], dtype=bool)
        wavelength_modes = np.asarray(source["wavelength_mode"], dtype=str)

    n = patterns.shape[0]
    vectors = {
        "record": records,
        "sg": sg_numbers,
        "eg": eg_numbers,
        "community": communities,
        "community_distance": community_distances,
        "community_outlier": community_outlier,
        "community_assigned": community_assigned,
        "wavelength_mode": wavelength_modes,
    }
    if patterns.ndim != 2:
        raise ValueError(f"Intensity array must be two-dimensional: {patterns.shape}")
    for name, values in vectors.items():
        if values.shape != (n,):
            raise ValueError(f"{name} has shape {values.shape}; expected {(n,)}")
    if n == 0:
        raise ValueError("opXRD evaluation surface is empty")
    if not np.all(np.isfinite(patterns)):
        raise ValueError("opXRD intensity array contains non-finite values")
    if sg_numbers.min() < 1 or sg_numbers.max() > 230:
        raise ValueError("opXRD SG labels must use crystallographic numbers 1..230")
    if eg_numbers.min() < 1 or eg_numbers.max() > 99:
        raise ValueError("opXRD EG labels must use crystallographic numbers 1..99")
    if np.any(community_assigned & (communities < 0)):
        raise ValueError("Assigned community records must have nonnegative IDs")
    if np.any(~community_assigned & community_outlier):
        raise ValueError("Community projection failures cannot be marked as outliers")

    sg_to_eg = load_sg_to_eg_map(str(mapping_csv))
    sg_targets = sg_numbers - 1
    eg_targets = eg_numbers - 1
    mapped_targets = sg_to_eg[sg_numbers]
    if not np.array_equal(mapped_targets, eg_targets):
        bad = np.flatnonzero(mapped_targets != eg_targets)
        raise ValueError(
            "Stored opXRD SG and EG labels disagree with the canonical mapping "
            f"for rows {bad[:20].tolist()}"
        )

    target_grid = common_grid(args.grid_min, args.grid_max, args.grid_step)
    selected_patterns, grid_selection = select_exact_aligned_patterns(
        patterns,
        source_grid,
        target_grid,
    )
    normalized_patterns, selected_row_maxima = normalize_patterns(
        selected_patterns,
        args.input_target_max,
    )

    device = torch.device(args.device)
    model = torch.load(checkpoint, map_location="cpu", weights_only=False).to(device)
    model.eval()

    logits = []
    with torch.no_grad():
        for start in range(0, n, args.batch_size):
            batch = torch.from_numpy(
                normalized_patterns[start : start + args.batch_size]
            )
            output = model(batch.unsqueeze(1).to(device))
            if isinstance(output, (tuple, list)):
                output = output[0]
            logits.append(output.detach().cpu().numpy())
    logits_array = np.concatenate(logits, axis=0)
    if logits_array.shape != (n, 230):
        raise ValueError(
            f"Expected model output shape {(n, 230)}; found {logits_array.shape}"
        )

    sg_probs = torch.softmax(torch.from_numpy(logits_array), dim=1).numpy()
    eg_probs = collapse_logits_to_eg_probs(logits_array, sg_to_eg)
    top1_sg = np.argmax(sg_probs, axis=1)
    top5_sg = topk_indices(sg_probs, 5)
    top1_eg = np.argmax(eg_probs, axis=1)
    top5_eg = topk_indices(eg_probs, 5)

    source_names = np.asarray(
        [record.split("/", 1)[0] for record in records],
        dtype=str,
    )
    strata_masks = {
        "all": np.ones(n, dtype=bool),
        "wavelength_explicit_kalpha1_kalpha2": (
            wavelength_modes == "kalpha1_kalpha2_pair"
        ),
        "wavelength_weighted_average": wavelength_modes == "weighted_average",
        "community_in_basin": community_assigned & ~community_outlier,
        "community_frontier_like": community_assigned & community_outlier,
        "community_unassigned": ~community_assigned,
    }
    for source_name in sorted(np.unique(source_names)):
        strata_masks[f"source_{source_name.lower()}"] = source_names == source_name
    metrics_by_stratum = {
        name: metric_block(mask, sg_probs, eg_probs, sg_targets, eg_targets)
        for name, mask in strata_masks.items()
    }
    all_metrics = metrics_by_stratum["all"]

    result = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "eval_npz": str(eval_npz),
        "eval_npz_sha256": sha256_file(eval_npz),
        "mapping_csv": str(mapping_csv),
        "mapping_csv_sha256": sha256_file(mapping_csv),
        "n": n,
        "label_conventions": {
            "model_output": (
                "230 SG logits; column 0 is SG 1 and column 229 is SG 230"
            ),
            "aggregation": (
                "softmax SG probabilities summed into canonical EG indices"
            ),
            "stored_sg_labels": "crystallographic SG numbers 1..230",
            "stored_eg_labels": "crystallographic EG numbers 1..99",
            "posterior_column_base": 0,
            "examples_eg_label_base": 1,
        },
        "preprocessing": {
            "source_points": int(source_grid.shape[0]),
            "source_grid_min": float(source_grid[0]),
            "source_grid_max": float(source_grid[-1]),
            "target_points": int(target_grid.shape[0]),
            "target_grid_min": float(target_grid[0]),
            "target_grid_max": float(target_grid[-1]),
            "target_grid_step": float(args.grid_step),
            "grid_selection": grid_selection,
            "intensity_normalization": (
                "each exactly decimated 3501-point pattern independently "
                "max-normalized to the SimXRD training maximum"
            ),
            "input_target_max": float(args.input_target_max),
            "selected_row_max_before_normalization_min": float(
                selected_row_maxima.min()
            ),
            "selected_row_max_before_normalization_max": float(
                selected_row_maxima.max()
            ),
            "converted_row_max_min": float(
                normalized_patterns.max(axis=1).min()
            ),
            "converted_row_max_max": float(
                normalized_patterns.max(axis=1).max()
            ),
        },
        "top1_sg": all_metrics["top1_sg"],
        "top3_sg": all_metrics["top3_sg"],
        "top5_sg": all_metrics["top5_sg"],
        "top1_eg": all_metrics["top1_eg"],
        "top3_eg": all_metrics["top3_eg"],
        "top5_eg": all_metrics["top5_eg"],
        "metrics_by_stratum": metrics_by_stratum,
        "examples": [
            {
                "case_id": str(records[index]),
                "true_sg": int(sg_targets[index]) + 1,
                "pred_sg": int(top1_sg[index]) + 1,
                "top5_sg": [int(value) + 1 for value in top5_sg[index]],
                "true_eg": int(eg_targets[index]) + 1,
                "pred_eg": int(top1_eg[index]) + 1,
                "top5_eg": [int(value) + 1 for value in top5_eg[index]],
                "correct": bool(top1_eg[index] == eg_targets[index]),
                "wavelength_mode": str(wavelength_modes[index]),
                "community": (
                    int(communities[index])
                    if community_assigned[index]
                    else None
                ),
                "community_status": (
                    "frontier_like"
                    if community_assigned[index] and community_outlier[index]
                    else "in_basin"
                    if community_assigned[index]
                    else "unassigned"
                ),
            }
            for index in range(n)
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
            sg_probs=sg_probs.astype(np.float32),
            eg_probs=eg_probs.astype(np.float32),
            targets_sg_number=sg_numbers.astype(np.int16),
            targets_eg_number=eg_numbers.astype(np.int16),
            case_ids=records,
            top1_pred_sg_number=(top1_sg + 1).astype(np.int16),
            top1_pred_eg_number=(top1_eg + 1).astype(np.int16),
            top5_pred_sg_number=(top5_sg + 1).astype(np.int16),
            top5_pred_eg_number=(top5_eg + 1).astype(np.int16),
            two_theta=target_grid.astype(np.float32),
            wavelength_mode=wavelength_modes,
            community=communities.astype(np.int32),
            community_assigned=community_assigned,
            community_outlier=community_outlier,
        )

    print(
        json.dumps(
            {
                "n": n,
                "top1_sg": result["top1_sg"],
                "top3_sg": result["top3_sg"],
                "top5_sg": result["top5_sg"],
                "top1_eg": result["top1_eg"],
                "top3_eg": result["top3_eg"],
                "top5_eg": result["top5_eg"],
                "metrics_by_stratum": metrics_by_stratum,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
