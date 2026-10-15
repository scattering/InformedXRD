#!/usr/bin/env python3
import argparse
import json

import torch
import torch.nn as nn

from dataset import get_dataloaders
from extinction_multilabel import multilabel_targets_to_split_targets
from model import VIT_model, adapt_patch_embed_input_channels


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate split-head component accuracies.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config) as handle:
        config = json.load(handle)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    checkpoint_config = checkpoint.get("config", {})
    merged_config = dict(checkpoint_config)
    merged_config.update(config)
    if args.data_path:
        merged_config["data_path"] = args.data_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    if merged_config.get("use_aux_ext_head", False):
        model.aux_ext_head = nn.Linear(model.embed_dim, merged_config.get("num_classes", 99)).to(device)
        aux_state = checkpoint.get("aux_ext_head_state_dict")
        if aux_state is not None:
            model.aux_ext_head.load_state_dict(aux_state)

    adapted_state_dict = adapt_patch_embed_input_channels(checkpoint["model_state_dict"], model)
    model.load_state_dict(adapted_state_dict, strict=False)
    model.eval()

    train_loader, val_loader, test_loader, _, _ = get_dataloaders(
        h5_file=merged_config["data_path"],
        batch_size=merged_config["batch_size"],
        num_workers=1,
        num_classes=merged_config.get("num_classes", 99),
        prefetch_factor=merged_config.get("prefetch_factor", 4),
        start_col=merged_config["start_col"],
        end_col=merged_config["end_col"],
        label_mode=merged_config.get("label_mode", "multilabel"),
        canonical_table_path=merged_config.get("canonical_table_path"),
        final_table_path=merged_config.get("final_table_path"),
        sg_lookup_path=merged_config.get("sg_lookup_path"),
    )

    loader = val_loader if args.split == "val" else test_loader

    total = 0
    system_correct = 0
    lattice_correct = 0
    operator_correct = 0
    operator_total = 0
    operator_exact_match = 0

    with torch.no_grad():
        for inputs, targets, _ in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            cls_embedding = model(inputs, return_cls_embedding=True)
            system_logits = model.head_sys(cls_embedding)
            lattice_logits = model.head_lat(cls_embedding)
            operator_logits = model.head_ops(cls_embedding)

            system_targets, lattice_targets, operator_targets = multilabel_targets_to_split_targets(targets)
            system_targets = system_targets.to(device)
            lattice_targets = lattice_targets.to(device)
            operator_targets = operator_targets.to(device)

            system_pred = system_logits.argmax(dim=1)
            lattice_pred = lattice_logits.argmax(dim=1)
            operator_pred = (torch.sigmoid(operator_logits) >= 0.5).float()

            total += inputs.size(0)
            system_correct += (system_pred == system_targets).sum().item()
            lattice_correct += (lattice_pred == lattice_targets).sum().item()
            operator_correct += (operator_pred == operator_targets).sum().item()
            operator_total += operator_targets.numel()
            operator_exact_match += (operator_pred == operator_targets).all(dim=1).sum().item()

    print(f"split={args.split}")
    print(f"total={total}")
    print(f"system_acc={100.0 * system_correct / total:.2f}")
    print(f"lattice_acc={100.0 * lattice_correct / total:.2f}")
    print(f"operator_bit_acc={100.0 * operator_correct / operator_total:.2f}")
    print(f"operator_exact_match={100.0 * operator_exact_match / total:.2f}")


if __name__ == "__main__":
    main()
