#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn

from dataset import get_dataloaders_test
from extinction_multilabel import build_template_bank, template_mask_from_ext_groups
from model import VIT_model, adapt_patch_embed_input_channels


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate grouped cross-validation for auxiliary-head temperature calibration on a real benchmark."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--eval-data-path", required=True)
    parser.add_argument("--prior-data-path", required=True)
    parser.add_argument("--group-key", default="minerals", help="HDF5 dataset used for grouped CV.")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument(
        "--temperature-grid",
        type=float,
        nargs="+",
        default=[1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0],
    )
    parser.add_argument(
        "--selection-metric",
        choices=["nll", "ece", "top1", "top5"],
        default="nll",
        help="Metric optimized on the calibration folds when selecting temperature.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def get_model_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def load_config(args, checkpoint):
    config = dict(checkpoint.get("config", {})) if isinstance(checkpoint, dict) else {}
    if args.config:
        with open(args.config, "r") as handle:
            config.update(json.load(handle))
    config["data_path"] = args.eval_data_path
    return config


def build_model(config, checkpoint, device):
    model = VIT_model(
        spec_length=config["spec_length"],
        num_output=config.get("num_labels", 37),
        patch_size=config["patch_size"],
        embed_dim=config["embed_dim"],
        depth=config["depth"],
        num_heads=config["num_heads"],
        mlp_ratio=config["mlp_ratio"],
        drop_ratio=config["dropout"],
        use_rope=config["use_rope"],
        use_mlp_head=config["use_mlp_head"],
        mlp_head_hidden_dim=config["mlp_head_hidden_dim"],
        use_physics_pe=config.get("use_physics_pe", False),
        physics_pe_mode=config.get("physics_pe_mode", "sin2theta"),
        two_theta_min=float(config.get("two_theta_min", 5.0)),
        two_theta_max=float(config.get("two_theta_max", 90.0)),
        physics_pe_scale=float(config.get("physics_pe_scale", 1.0)),
        use_coordinate_channel=config.get("use_coordinate_channel", False),
        coordinate_mode=config.get("coordinate_mode", "sin2theta"),
    )
    if config.get("use_split_head", False):
        model.head_sys = nn.Linear(model.embed_dim, 7)
        model.head_lat = nn.Linear(model.embed_dim, 5)
        model.head_ops = nn.Linear(model.embed_dim, 25)
    if config.get("use_aux_ext_head", False):
        model.aux_ext_head = nn.Linear(model.embed_dim, config.get("num_classes", 99))
        aux_state = checkpoint.get("aux_ext_head_state_dict") if isinstance(checkpoint, dict) else None
        if aux_state is not None:
            model.aux_ext_head.load_state_dict(aux_state)
    adapted_state_dict = adapt_patch_embed_input_channels(get_model_state_dict(checkpoint), model)
    model.load_state_dict(adapted_state_dict, strict=False)
    return model.to(device).eval()


def load_seen_ext_groups(h5_path):
    with h5py.File(h5_path, "r") as handle:
        return sorted(set(int(v) for v in handle["y_train"][:].tolist()))


def load_log_priors(h5_path, ext_group_order, smoothing=1.0):
    with h5py.File(h5_path, "r") as handle:
        y_train = handle["y_train"][:]
    counts = {int(ext): smoothing for ext in ext_group_order}
    for value in y_train.tolist():
        counts[int(value)] = counts.get(int(value), smoothing) + 1.0
    probs = torch.tensor([counts[int(ext)] for ext in ext_group_order], dtype=torch.float32)
    probs = probs / probs.sum()
    return torch.log(probs)


def decode_strings(values):
    out = []
    for value in values.tolist():
        out.append(value.decode() if isinstance(value, bytes) else str(value))
    return out


def multiclass_brier(probs: np.ndarray, targets: np.ndarray) -> float:
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(targets.shape[0]), targets] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def expected_calibration_error(probs: np.ndarray, targets: np.ndarray, bins: int = 15) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == targets).astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for i in range(bins):
        left, right = edges[i], edges[i + 1]
        if i == bins - 1:
            mask = (conf >= left) & (conf <= right)
        else:
            mask = (conf >= left) & (conf < right)
        if not np.any(mask):
            continue
        acc = correct[mask].mean()
        c = conf[mask].mean()
        ece += mask.mean() * abs(acc - c)
    return float(ece)


def summarize_probs(probs: np.ndarray, targets: np.ndarray) -> dict:
    top1 = float(np.mean(probs.argmax(axis=1) == targets))
    k = min(5, probs.shape[1])
    topk_idx = np.argpartition(-probs, kth=k - 1, axis=1)[:, :k]
    top5 = float(np.mean([t in row for t, row in zip(targets.tolist(), topk_idx.tolist())]))
    true_probs = probs[np.arange(targets.shape[0]), targets]
    nll = float(np.mean(-np.log(np.clip(true_probs, 1e-12, 1.0))))
    return {
        "top1": top1,
        "top5": top5,
        "ece": expected_calibration_error(probs, targets),
        "nll": nll,
        "brier": multiclass_brier(probs, targets),
    }


def make_group_folds(groups, n_folds, seed):
    unique_groups = np.asarray(sorted(set(groups.tolist())), dtype=object)
    rng = np.random.default_rng(seed)
    shuffled = unique_groups.copy()
    rng.shuffle(shuffled)
    buckets = [[] for _ in range(n_folds)]
    for idx, group in enumerate(shuffled.tolist()):
        buckets[idx % n_folds].append(group)
    folds = []
    for bucket in buckets:
        test_groups = set(bucket)
        test_idx = np.asarray([i for i, group in enumerate(groups.tolist()) if group in test_groups], dtype=np.int64)
        calib_idx = np.asarray([i for i, group in enumerate(groups.tolist()) if group not in test_groups], dtype=np.int64)
        folds.append((calib_idx, test_idx, sorted(test_groups)))
    return folds


def scores_from_logits(aux_logits, log_priors, seen_mask, temperature):
    scores = aux_logits / temperature + log_priors[None, :]
    scores = np.where(seen_mask[None, :], scores, -np.inf)
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)
    return probs


def pick_temperature(aux_logits, targets, log_priors, seen_mask, grid, metric):
    direction = 1.0 if metric in {"top1", "top5"} else -1.0
    best_temp = None
    best_value = None
    sweep = []
    for temp in grid:
        probs = scores_from_logits(aux_logits, log_priors, seen_mask, temp)
        summary = summarize_probs(probs, targets)
        value = direction * summary[metric]
        sweep.append({"temperature": float(temp), **summary})
        if best_value is None or value > best_value:
            best_value = value
            best_temp = float(temp)
    return best_temp, sweep


def evaluate_temperature_sweep(aux_logits, targets, log_priors, seen_mask, grid):
    sweep = []
    for temp in grid:
        probs = scores_from_logits(aux_logits, log_priors, seen_mask, temp)
        summary = summarize_probs(probs, targets)
        sweep.append({"temperature": float(temp), **summary})
    return sweep


def main():
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = load_config(args, checkpoint)
    device = torch.device(args.device)

    model = build_model(config, checkpoint, device)
    _, ext_group_order, _ = build_template_bank(
        canonical_table_path=config.get("canonical_table_path"),
        final_table_path=config.get("final_table_path"),
        sg_lookup_path=config.get("sg_lookup_path"),
    )
    log_priors = load_log_priors(args.prior_data_path, ext_group_order).cpu().numpy()
    seen_mask = template_mask_from_ext_groups(ext_group_order, load_seen_ext_groups(args.prior_data_path)).cpu().numpy()

    _, _, test_loader, _, _ = get_dataloaders_test(
        h5_file=args.eval_data_path,
        batch_size=config["batch_size"],
        num_workers=args.num_workers,
        num_classes=config.get("num_classes", 99),
        prefetch_factor=config.get("prefetch_factor", 4) if args.num_workers > 0 else None,
        start_col=config["start_col"],
        end_col=config["end_col"],
        label_mode=config.get("label_mode", "multilabel"),
        canonical_table_path=config.get("canonical_table_path"),
        final_table_path=config.get("final_table_path"),
        sg_lookup_path=config.get("sg_lookup_path"),
        max_samples_test=config.get("max_samples_test"),
    )

    aux_logits = []
    all_targets = []
    with torch.no_grad():
        for inputs, _, ext_targets in test_loader:
            inputs = inputs.to(device)
            ext_targets = ext_targets.to(device)
            cls_embedding = model(inputs, return_cls_embedding=True)
            logits = model.aux_ext_head(cls_embedding)
            aux_logits.append(logits.cpu().numpy())
            all_targets.append(ext_targets.cpu().numpy())
    aux_logits = np.concatenate(aux_logits, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    with h5py.File(args.eval_data_path, "r") as handle:
        case_ids = decode_strings(handle["case_ids"][:]) if "case_ids" in handle else [str(i) for i in range(len(targets))]
        groups = np.asarray(decode_strings(handle[args.group_key][:]), dtype=object)

    if groups.shape[0] != targets.shape[0]:
        raise ValueError(f"group_key '{args.group_key}' has {groups.shape[0]} rows, expected {targets.shape[0]}")

    folds = make_group_folds(groups, args.n_folds, args.seed)
    fold_results = []
    for fold_idx, (calib_idx, test_idx, test_groups) in enumerate(folds, start=1):
        chosen_temp, calibration_sweep = pick_temperature(
            aux_logits[calib_idx],
            targets[calib_idx],
            log_priors,
            seen_mask,
            args.temperature_grid,
            args.selection_metric,
        )
        test_sweep = evaluate_temperature_sweep(
            aux_logits[test_idx],
            targets[test_idx],
            log_priors,
            seen_mask,
            args.temperature_grid,
        )
        test_probs = scores_from_logits(aux_logits[test_idx], log_priors, seen_mask, chosen_temp)
        test_summary = summarize_probs(test_probs, targets[test_idx])
        fold_results.append(
            {
                "fold": fold_idx,
                "n_calibration": int(calib_idx.shape[0]),
                "n_test": int(test_idx.shape[0]),
                "n_test_groups": len(test_groups),
                "test_groups": test_groups,
                "selected_temperature": chosen_temp,
                "selection_metric": args.selection_metric,
                "calibration_sweep": calibration_sweep,
                "test_sweep": test_sweep,
                "test_metrics": test_summary,
                "test_case_ids": [case_ids[i] for i in test_idx.tolist()],
            }
        )

    mean_metrics = {}
    for key in ("top1", "top5", "ece", "nll", "brier"):
        values = np.asarray([fold["test_metrics"][key] for fold in fold_results], dtype=np.float64)
        mean_metrics[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "min": float(values.min()),
            "max": float(values.max()),
        }

    output = {
        "checkpoint": args.checkpoint,
        "config": args.config,
        "eval_data_path": args.eval_data_path,
        "prior_data_path": args.prior_data_path,
        "group_key": args.group_key,
        "n_folds": args.n_folds,
        "selection_metric": args.selection_metric,
        "temperature_grid": [float(v) for v in args.temperature_grid],
        "n_samples": int(targets.shape[0]),
        "n_groups": int(len(set(groups.tolist()))),
        "fold_results": fold_results,
        "aggregate_test_metrics": mean_metrics,
        "selected_temperatures": [fold["selected_temperature"] for fold in fold_results],
    }
    Path(args.output_json).write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
