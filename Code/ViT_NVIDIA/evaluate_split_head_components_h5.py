#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn

from extinction_multilabel import (
    FEATURES,
    OPERATOR_SLICE,
    build_extinction_templates,
    multilabel_targets_to_split_targets,
)
from model import VIT_model, adapt_patch_embed_input_channels


def get_model_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate split-head components directly from HDF5 splits.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--save-logits", default="")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config) as handle:
        config = json.load(handle)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    checkpoint_config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    merged_config = dict(checkpoint_config)
    merged_config.update(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    templates = build_extinction_templates(
        canonical_table_path=merged_config.get("canonical_table_path"),
        final_table_path=merged_config.get("final_table_path"),
        sg_lookup_path=merged_config.get("sg_lookup_path"),
    )

    model = VIT_model(
        spec_length=merged_config["spec_length"],
        num_output=merged_config.get("num_labels", 37),
        patch_size=merged_config["patch_size"],
        embed_dim=merged_config["embed_dim"],
        depth=merged_config["depth"],
        num_heads=merged_config["num_heads"],
        mlp_ratio=merged_config["mlp_ratio"],
        drop_ratio=merged_config["dropout"],
        use_rope=merged_config["use_rope"],
        use_mlp_head=merged_config["use_mlp_head"],
        mlp_head_hidden_dim=merged_config["mlp_head_hidden_dim"],
        use_physics_pe=merged_config.get("use_physics_pe", False),
        physics_pe_mode=merged_config.get("physics_pe_mode", "sin2theta"),
        two_theta_min=float(merged_config.get("two_theta_min", 5.0)),
        two_theta_max=float(merged_config.get("two_theta_max", 90.0)),
        physics_pe_scale=float(merged_config.get("physics_pe_scale", 1.0)),
        use_coordinate_channel=merged_config.get("use_coordinate_channel", False),
        coordinate_mode=merged_config.get("coordinate_mode", "sin2theta"),
    ).to(device)
    model.head_sys = nn.Linear(model.embed_dim, 7).to(device)
    model.head_lat = nn.Linear(model.embed_dim, 5).to(device)
    model.head_ops = nn.Linear(model.embed_dim, 25).to(device)
    adapted_state_dict = adapt_patch_embed_input_channels(get_model_state_dict(checkpoint), model)
    model.load_state_dict(adapted_state_dict, strict=False)
    model.eval()

    x_key = f"X_{args.split}"
    y_key = f"y_{args.split}"
    start_col = merged_config["start_col"]
    end_col = merged_config["end_col"]

    total = 0
    system_correct = 0
    lattice_correct = 0
    operator_correct = 0
    operator_total = 0
    operator_exact_match = 0
    operator_correct_by_bit = None
    operator_total_by_bit = None
    all_system_logits = []
    all_lattice_logits = []
    all_operator_logits = []
    all_system_targets = []
    all_lattice_targets = []
    all_operator_targets = []
    all_ext_groups = []

    with h5py.File(args.data_path, "r") as handle:
        x_ds = handle[x_key]
        y_ds = handle[y_key]
        n = x_ds.shape[0]
        if args.max_samples and args.max_samples > 0:
            n = min(n, args.max_samples)

        for start in range(0, n, args.batch_size):
            stop = min(start + args.batch_size, n)
            x = torch.from_numpy(x_ds[start:stop, start_col:end_col].astype("float32"))
            ext_groups = y_ds[start:stop].astype("int64")
            target_vectors = torch.stack([templates[int(ext)].vector for ext in ext_groups], dim=0)

            x = x.unsqueeze(1).to(device)
            target_vectors = target_vectors.to(device)

            with torch.no_grad():
                cls_embedding = model(x, return_cls_embedding=True)
                system_logits = model.head_sys(cls_embedding)
                lattice_logits = model.head_lat(cls_embedding)
                operator_logits = model.head_ops(cls_embedding)

            system_targets, lattice_targets, operator_targets = multilabel_targets_to_split_targets(target_vectors)
            system_pred = system_logits.argmax(dim=1)
            lattice_pred = lattice_logits.argmax(dim=1)
            operator_pred = (torch.sigmoid(operator_logits) >= 0.5).float()

            total += x.size(0)
            system_correct += (system_pred == system_targets).sum().item()
            lattice_correct += (lattice_pred == lattice_targets).sum().item()
            operator_correct += (operator_pred == operator_targets).sum().item()
            operator_total += operator_targets.numel()
            operator_exact_match += (operator_pred == operator_targets).all(dim=1).sum().item()
            bit_correct = (operator_pred == operator_targets).sum(dim=0).cpu().numpy().astype(np.int64)
            bit_total = np.full(bit_correct.shape, operator_targets.shape[0], dtype=np.int64)
            if operator_correct_by_bit is None:
                operator_correct_by_bit = bit_correct
                operator_total_by_bit = bit_total
            else:
                operator_correct_by_bit += bit_correct
                operator_total_by_bit += bit_total

            all_system_logits.append(system_logits.cpu().numpy().astype(np.float32))
            all_lattice_logits.append(lattice_logits.cpu().numpy().astype(np.float32))
            all_operator_logits.append(operator_logits.cpu().numpy().astype(np.float32))
            all_system_targets.append(system_targets.cpu().numpy().astype(np.int64))
            all_lattice_targets.append(lattice_targets.cpu().numpy().astype(np.int64))
            all_operator_targets.append(operator_targets.cpu().numpy().astype(np.float32))
            all_ext_groups.append(ext_groups)

    operator_feature_names = FEATURES[OPERATOR_SLICE]
    operator_bit_accuracy = []
    for i, name in enumerate(operator_feature_names):
        operator_bit_accuracy.append(
            {
                "feature": name,
                "acc_pct": 100.0 * float(operator_correct_by_bit[i]) / float(operator_total_by_bit[i]),
                "correct": int(operator_correct_by_bit[i]),
                "total": int(operator_total_by_bit[i]),
            }
        )

    output = {
        "checkpoint": args.checkpoint,
        "data_path": args.data_path,
        "split": args.split,
        "total": int(total),
        "component_acc_pct": {
            "system_acc": 100.0 * system_correct / total,
            "lattice_acc": 100.0 * lattice_correct / total,
            "operator_bit_acc": 100.0 * operator_correct / operator_total,
            "operator_exact_match": 100.0 * operator_exact_match / total,
        },
        "operator_bit_accuracy_pct": operator_bit_accuracy,
    }

    text = json.dumps(output, indent=2)
    print(text)
    if args.output_json:
        Path(args.output_json).write_text(text)

    if args.save_logits:
        np.savez_compressed(
            args.save_logits,
            system_logits=np.concatenate(all_system_logits, axis=0),
            lattice_logits=np.concatenate(all_lattice_logits, axis=0),
            operator_logits=np.concatenate(all_operator_logits, axis=0),
            system_targets=np.concatenate(all_system_targets, axis=0),
            lattice_targets=np.concatenate(all_lattice_targets, axis=0),
            operator_targets=np.concatenate(all_operator_targets, axis=0),
            ext_groups=np.concatenate(all_ext_groups, axis=0),
            operator_feature_names=np.asarray(operator_feature_names, dtype=object),
        )


if __name__ == "__main__":
    main()
