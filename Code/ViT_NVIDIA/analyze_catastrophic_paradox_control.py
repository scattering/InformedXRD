#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn

from dataset import get_dataloaders_test
from model import VIT_model, adapt_patch_embed_input_channels


def parse_args():
    parser = argparse.ArgumentParser(description="Control the catastrophic paradox for EG composition.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--eval-data-path", required=True)
    parser.add_argument("--prior-data-path", required=True)
    parser.add_argument("--temperature", type=float, default=5.0)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def get_model_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


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
    if config.get("use_aux_ext_head", False):
        model.aux_ext_head = nn.Linear(model.embed_dim, config.get("num_classes", 99))
        aux_state = checkpoint.get("aux_ext_head_state_dict") if isinstance(checkpoint, dict) else None
        if aux_state is not None:
            model.aux_ext_head.load_state_dict(aux_state)
    adapted = adapt_patch_embed_input_channels(get_model_state_dict(checkpoint), model)
    model.load_state_dict(adapted, strict=False)
    return model.to(device).eval()


def load_log_priors(h5_path, num_classes, smoothing=1.0):
    with h5py.File(h5_path, "r") as handle:
        y_train = handle["y_train"][:]
    counts = np.full(num_classes, smoothing, dtype=np.float64)
    for value in y_train.tolist():
        counts[int(value) - 1] += 1.0
    probs = counts / counts.sum()
    return torch.log(torch.tensor(probs, dtype=torch.float32))


def bootstrap_diff(mask_a, mask_b, correct, rounds=2000, seed=123):
    rng = np.random.default_rng(seed)
    idx_a = np.where(mask_a)[0]
    idx_b = np.where(mask_b)[0]
    diffs = []
    for _ in range(rounds):
        samp_a = rng.choice(idx_a, size=idx_a.shape[0], replace=True)
        samp_b = rng.choice(idx_b, size=idx_b.shape[0], replace=True)
        diffs.append(float(correct[samp_a].mean() - correct[samp_b].mean()))
    arr = np.asarray(diffs, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "ci95_low": float(np.quantile(arr, 0.025)),
        "ci95_high": float(np.quantile(arr, 0.975)),
    }


def bootstrap_mean(correct_values, rounds=2000, seed=123):
    rng = np.random.default_rng(seed)
    n = correct_values.shape[0]
    vals = []
    for _ in range(rounds):
        idx = rng.integers(0, n, size=n)
        vals.append(float(correct_values[idx].mean()))
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "ci95_low": float(np.quantile(arr, 0.025)),
        "ci95_high": float(np.quantile(arr, 0.975)),
    }


def bootstrap_reweighted(correct, severity, eg, target_severity, global_weights, rounds=2000, seed=123):
    rng = np.random.default_rng(seed)
    sev_mask = severity == target_severity
    sev_idx = np.where(sev_mask)[0]
    observed_egs = sorted(set(eg[sev_mask].tolist()))
    vals = []
    for _ in range(rounds):
        sample_idx = rng.choice(sev_idx, size=sev_idx.shape[0], replace=True)
        sample_eg = eg[sample_idx]
        sample_correct = correct[sample_idx]
        eg_acc = {}
        for value in observed_egs:
            eg_mask = sample_eg == value
            if not np.any(eg_mask):
                continue
            eg_acc[int(value)] = float(sample_correct[eg_mask].mean())
        covered = sum(global_weights[k] for k in eg_acc)
        vals.append(float(sum(global_weights[k] * eg_acc[k] for k in eg_acc) / covered))
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "ci95_low": float(np.quantile(arr, 0.025)),
        "ci95_high": float(np.quantile(arr, 0.975)),
    }


def main():
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    with open(args.config) as handle:
        config = json.load(handle)
    merged = dict(checkpoint.get("config", {})) if isinstance(checkpoint, dict) else {}
    merged.update(config)
    merged["data_path"] = args.eval_data_path

    device = torch.device(args.device)
    model = build_model(merged, checkpoint, device)
    log_priors = load_log_priors(args.prior_data_path, merged.get("num_classes", 99)).to(device)

    _, _, test_loader, _, _ = get_dataloaders_test(
        h5_file=args.eval_data_path,
        batch_size=merged["batch_size"],
        num_workers=args.num_workers,
        num_classes=merged.get("num_classes", 99),
        prefetch_factor=merged.get("prefetch_factor", 4) if args.num_workers > 0 else None,
        start_col=merged["start_col"],
        end_col=merged["end_col"],
        label_mode=merged.get("label_mode", "multilabel"),
        canonical_table_path=merged.get("canonical_table_path"),
        final_table_path=merged.get("final_table_path"),
        sg_lookup_path=merged.get("sg_lookup_path"),
        max_samples_test=merged.get("max_samples_test"),
    )

    preds = []
    targets = []
    with torch.no_grad():
        for inputs, _, ext_targets in test_loader:
            inputs = inputs.to(device)
            cls = model(inputs, return_cls_embedding=True)
            aux_logits = model.aux_ext_head(cls)
            scores = aux_logits / args.temperature + log_priors.to(dtype=aux_logits.dtype).unsqueeze(0)
            pred = torch.argmax(scores, dim=1)
            preds.append(pred.cpu().numpy())
            targets.append(ext_targets.cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)
    correct = (preds == targets).astype(np.float64)

    with h5py.File(args.eval_data_path, "r") as handle:
        severity = [v.decode() if isinstance(v, bytes) else str(v) for v in handle["severity"][:].tolist()]
        minerals = [v.decode() if isinstance(v, bytes) else str(v) for v in handle["minerals"][:].tolist()]

    severity = np.asarray(severity, dtype=object)
    minerals = np.asarray(minerals, dtype=object)
    eg = targets + 1

    severity_levels = sorted(set(severity.tolist()))
    severity_seed_offset = {sev: idx for idx, sev in enumerate(severity_levels)}
    per_severity = {}
    for sev in severity_levels:
        mask = severity == sev
        per_severity[sev] = {
            "n": int(mask.sum()),
            "top1_acc": float(correct[mask].mean()),
            "bootstrap_top1": bootstrap_mean(
                correct[mask],
                rounds=args.bootstrap,
                seed=args.seed + severity_seed_offset[sev],
            ),
            "unique_egs": int(len(set(eg[mask].tolist()))),
            "unique_minerals": int(len(set(minerals[mask].tolist()))),
        }

    global_eg_weights = defaultdict(float)
    for value in eg.tolist():
        global_eg_weights[int(value)] += 1.0
    total = float(len(eg))
    for key in list(global_eg_weights):
        global_eg_weights[key] /= total

    reweighted = {}
    for sev in severity_levels:
        sev_mask = severity == sev
        eg_acc = {}
        for value in sorted(set(eg[sev_mask].tolist())):
            mask = sev_mask & (eg == value)
            eg_acc[int(value)] = float(correct[mask].mean())
        weighted = sum(global_eg_weights[k] * eg_acc[k] for k in eg_acc)
        covered = sum(global_eg_weights[k] for k in eg_acc)
        reweighted[sev] = {
            "eg_reweighted_top1": float(weighted / covered) if covered > 0 else None,
            "coverage_weight": float(covered),
            "bootstrap_eg_reweighted_top1": bootstrap_reweighted(
                correct,
                severity,
                eg,
                sev,
                global_eg_weights,
                rounds=args.bootstrap,
                seed=args.seed + 1000 + severity_seed_offset[sev],
            ),
        }

    matched_eg_mineral = {}
    shared = {}
    if "catastrophic" in set(severity.tolist()) and "recoverable" in set(severity.tolist()):
        mask_cat = severity == "catastrophic"
        mask_rec = severity == "recoverable"
        keys_cat = set(zip(eg[mask_cat].tolist(), minerals[mask_cat].tolist()))
        keys_rec = set(zip(eg[mask_rec].tolist(), minerals[mask_rec].tolist()))
        shared_keys = keys_cat & keys_rec
        shared["n_pairs"] = len(shared_keys)
        weights = defaultdict(float)
        acc_cat = {}
        acc_rec = {}
        pooled_total = 0.0
        for key in shared_keys:
            eg_k, mineral_k = key
            mask_key_cat = mask_cat & (eg == eg_k) & (minerals == mineral_k)
            mask_key_rec = mask_rec & (eg == eg_k) & (minerals == mineral_k)
            c_cat = int(mask_key_cat.sum())
            c_rec = int(mask_key_rec.sum())
            pooled = c_cat + c_rec
            if pooled == 0:
                continue
            weights[key] = pooled
            pooled_total += pooled
            acc_cat[key] = float(correct[mask_key_cat].mean())
            acc_rec[key] = float(correct[mask_key_rec].mean())
        if pooled_total > 0:
            cat_weighted = sum((weights[k] / pooled_total) * acc_cat[k] for k in weights)
            rec_weighted = sum((weights[k] / pooled_total) * acc_rec[k] for k in weights)
            matched_eg_mineral = {
                "catastrophic_matched_top1": float(cat_weighted),
                "recoverable_matched_top1": float(rec_weighted),
                "difference_cat_minus_rec": float(cat_weighted - rec_weighted),
                "bootstrap_difference": bootstrap_diff(mask_cat, mask_rec, correct, rounds=args.bootstrap, seed=args.seed),
            }

    output = {
        "checkpoint": args.checkpoint,
        "eval_data_path": args.eval_data_path,
        "temperature": args.temperature,
        "per_severity": per_severity,
        "eg_reweighted": reweighted,
        "matched_eg_mineral": matched_eg_mineral,
        "shared_support": shared,
    }
    Path(args.output_json).write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
