#!/usr/bin/env python3
import argparse
import json
import os
import socket
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
VIT_DIR = REPO_ROOT / "Code" / "ViT_NVIDIA"
if str(VIT_DIR) not in sys.path:
    sys.path.insert(0, str(VIT_DIR))

from dataset import get_dataloaders_test  # noqa: E402
from extinction_multilabel import build_template_bank, template_mask_from_ext_groups  # noqa: E402
from model import VIT_model, adapt_patch_embed_input_channels  # noqa: E402


def refuse_tacc_login_node():
    hostname = socket.gethostname().lower()
    on_tacc = bool(os.environ.get("TACC_SYSTEM")) or "vista" in hostname or "stampede" in hostname
    if on_tacc and not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit(
            "Refusing to run inference on a TACC login node. Submit a Slurm launcher with sbatch."
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Dump full calibrated auxiliary-head posteriors for Gate 1 information analysis."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--eval-data-path", required=True)
    parser.add_argument("--prior-data-path", required=True)
    parser.add_argument("--aux-temperature", type=float, default=5.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--output-json", required=True)
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


def decode_strings(values):
    out = []
    for value in values.tolist():
        out.append(value.decode() if isinstance(value, bytes) else str(value))
    return out


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
    return torch.log(probs), probs


def main():
    refuse_tacc_login_node()
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
    log_priors, prior_probs = load_log_priors(args.prior_data_path, ext_group_order)
    seen_mask = template_mask_from_ext_groups(ext_group_order, load_seen_ext_groups(args.prior_data_path))

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

    calibrated_probs = []
    raw_probs = []
    targets = []
    with torch.no_grad():
        for inputs, _, ext_targets in test_loader:
            inputs = inputs.to(device)
            ext_targets = ext_targets.to(device)
            cls_embedding = model(inputs, return_cls_embedding=True)
            aux_logits = model.aux_ext_head(cls_embedding)

            raw = torch.softmax(aux_logits, dim=1)
            scores = aux_logits / args.aux_temperature + log_priors.to(dtype=aux_logits.dtype, device=device).unsqueeze(0)
            scores = scores.masked_fill(~seen_mask.to(device).unsqueeze(0), float("-inf"))
            calibrated = torch.softmax(scores, dim=1)

            raw_probs.append(raw.cpu().numpy())
            calibrated_probs.append(calibrated.cpu().numpy())
            targets.append(ext_targets.cpu().numpy())

    calibrated_probs = np.concatenate(calibrated_probs, axis=0)
    raw_probs = np.concatenate(raw_probs, axis=0)
    targets = np.concatenate(targets, axis=0)

    with h5py.File(args.eval_data_path, "r") as handle:
        case_ids = decode_strings(handle["case_ids"][:]) if "case_ids" in handle else [str(i) for i in range(targets.shape[0])]
        minerals = decode_strings(handle["minerals"][:]) if "minerals" in handle else ["" for _ in range(targets.shape[0])]
        severities = decode_strings(handle["severity"][:]) if "severity" in handle else ["" for _ in range(targets.shape[0])]
        fit_buckets = decode_strings(handle["fit_bucket"][:]) if "fit_bucket" in handle else ["" for _ in range(targets.shape[0])]

    Path(args.output_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_npz,
        calibrated_probs=calibrated_probs.astype(np.float32),
        raw_probs=raw_probs.astype(np.float32),
        targets=targets.astype(np.int64),
        ext_group_order=np.asarray(ext_group_order, dtype=np.int64),
        prior_probs=prior_probs.cpu().numpy().astype(np.float32),
        case_ids=np.asarray(case_ids, dtype=object),
        minerals=np.asarray(minerals, dtype=object),
        severities=np.asarray(severities, dtype=object),
        fit_buckets=np.asarray(fit_buckets, dtype=object),
    )

    metadata = {
        "checkpoint": args.checkpoint,
        "config": args.config,
        "eval_data_path": args.eval_data_path,
        "prior_data_path": args.prior_data_path,
        "aux_temperature": args.aux_temperature,
        "n_examples": int(targets.shape[0]),
        "num_classes": int(calibrated_probs.shape[1]),
        "output_npz": args.output_npz,
    }
    Path(args.output_json).write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
