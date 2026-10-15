#!/usr/bin/env python3
import argparse
import csv
import ast
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn

from dataset import get_dataloaders_test
from model import VIT_model, adapt_patch_embed_input_channels


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the matched SG->EG control by collapsing 230-way SG probabilities into EG space."
    )
    parser.add_argument("--sg-checkpoint", required=True)
    parser.add_argument("--sg-config", required=True)
    parser.add_argument("--sg-eval-data-path", required=True)
    parser.add_argument("--eg-truth-data-path", required=True)
    parser.add_argument("--eg-checkpoint", required=True)
    parser.add_argument("--eg-config", required=True)
    parser.add_argument("--canonical-table-path", required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", default=None, help="Optional per-pattern output with SG and collapsed EG top-k.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def materialize_config(config):
    materialized = dict(config)
    parameters = config.get("parameters", {})
    for key, value in parameters.items():
        if not isinstance(value, dict):
            materialized[key] = value
            continue
        if "value" in value:
            materialized[key] = value["value"]
        elif "values" in value and value["values"]:
            materialized[key] = value["values"][0]
    return materialized


def load_config(path):
    with open(path, "r") as handle:
        return materialize_config(json.load(handle))


def get_model_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def build_categorical_model(config, checkpoint, device):
    model = VIT_model(
        spec_length=config["spec_length"],
        num_output=config.get("num_classes", 230),
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
    adapted_state_dict = adapt_patch_embed_input_channels(get_model_state_dict(checkpoint), model)
    model.load_state_dict(adapted_state_dict, strict=False)
    return model.to(device).eval()


def build_multilabel_aux_model(config, checkpoint, device):
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


def load_sg_to_eg_map(canonical_table_path: str):
    sg_to_eg = {}
    with open(canonical_table_path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            eg = int(row["Index"]) + 1
            for sg in ast.literal_eval(row["Space Group Numbers"]):
                sg_to_eg[int(sg)] = eg
    if len(sg_to_eg) != 230:
        raise ValueError(f"Expected 230 SG->EG mappings, found {len(sg_to_eg)}")
    return sg_to_eg


def compute_topk(probs: np.ndarray, targets_zero_based: np.ndarray, k: int) -> float:
    topk = np.argpartition(-probs, kth=k - 1, axis=1)[:, :k]
    return float(np.mean([target in row for target, row in zip(targets_zero_based.tolist(), topk.tolist())]))


def summarize_probs(probs: np.ndarray, targets_zero_based: np.ndarray):
    return {
        "top1": float(np.mean(np.argmax(probs, axis=1) == targets_zero_based)),
        "top3": compute_topk(probs, targets_zero_based, 3),
        "top5": compute_topk(probs, targets_zero_based, 5),
    }


def load_eg_targets(h5_path: str) -> np.ndarray:
    with h5py.File(h5_path, "r") as handle:
        return handle["y_test"][:] - 1


def load_sg_targets(h5_path: str) -> np.ndarray:
    with h5py.File(h5_path, "r") as handle:
        return handle["y_test"][:] - 1


def check_test_alignment(sg_h5_path: str, eg_h5_path: str, rows: int = 16) -> dict:
    with h5py.File(sg_h5_path, "r") as sg_h5, h5py.File(eg_h5_path, "r") as eg_h5:
        sg_x = sg_h5["X_test"][:rows]
        eg_x = eg_h5["X_test"][:rows]
    return {
        "rows_checked": int(rows),
        "allclose": bool(np.allclose(sg_x, eg_x)),
        "max_abs_diff": float(np.max(np.abs(sg_x - eg_x))),
    }


def collect_sg_probs(model, config, h5_path: str, num_workers: int, device):
    _, _, test_loader, _, _ = get_dataloaders_test(
        h5_file=h5_path,
        batch_size=config["batch_size"],
        num_workers=num_workers,
        num_classes=config.get("num_classes", 230),
        prefetch_factor=config.get("prefetch_factor", 4) if num_workers > 0 else None,
        start_col=config["start_col"],
        end_col=config["end_col"],
        label_mode="categorical",
        max_samples_test=config.get("max_samples_test"),
    )
    probs = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(probs, axis=0)


def collect_eg_aux_probs(model, config, h5_path: str, num_workers: int, device):
    _, _, test_loader, _, _ = get_dataloaders_test(
        h5_file=h5_path,
        batch_size=config["batch_size"],
        num_workers=num_workers,
        num_classes=config.get("num_classes", 99),
        prefetch_factor=config.get("prefetch_factor", 4) if num_workers > 0 else None,
        start_col=config["start_col"],
        end_col=config["end_col"],
        label_mode="multilabel",
        canonical_table_path=config.get("canonical_table_path"),
        final_table_path=config.get("final_table_path"),
        sg_lookup_path=config.get("sg_lookup_path"),
        max_samples_test=config.get("max_samples_test"),
    )
    probs = []
    with torch.no_grad():
        for inputs, _, _ in test_loader:
            inputs = inputs.to(device)
            cls_embedding = model(inputs, return_cls_embedding=True)
            aux_logits = model.aux_ext_head(cls_embedding)
            probs.append(torch.softmax(aux_logits, dim=1).cpu().numpy())
    return np.concatenate(probs, axis=0)


def collapse_sg_probs_to_eg(sg_probs: np.ndarray, sg_to_eg: dict[int, int]) -> np.ndarray:
    eg_probs = np.zeros((sg_probs.shape[0], 99), dtype=np.float64)
    for sg_zero_based in range(sg_probs.shape[1]):
        sg_one_based = sg_zero_based + 1
        eg_one_based = sg_to_eg[sg_one_based]
        eg_probs[:, eg_one_based - 1] += sg_probs[:, sg_zero_based]
    return eg_probs


def topk_labels_from_probs(probs: np.ndarray, k: int, one_based: bool = True):
    idx = np.argsort(-probs, axis=1)[:, :k]
    return (idx + 1).tolist() if one_based else idx.tolist()


def main():
    args = parse_args()
    device = torch.device(args.device)

    sg_checkpoint = torch.load(args.sg_checkpoint, map_location="cpu")
    sg_config = load_config(args.sg_config)
    sg_config["data_path"] = args.sg_eval_data_path
    sg_model = build_categorical_model(sg_config, sg_checkpoint, device)

    eg_checkpoint = torch.load(args.eg_checkpoint, map_location="cpu")
    eg_config = load_config(args.eg_config)
    eg_config["data_path"] = args.eg_truth_data_path
    eg_model = build_multilabel_aux_model(eg_config, eg_checkpoint, device)

    sg_to_eg = load_sg_to_eg_map(args.canonical_table_path)
    eg_targets = load_eg_targets(args.eg_truth_data_path)
    sg_targets = load_sg_targets(args.sg_eval_data_path)
    alignment = check_test_alignment(args.sg_eval_data_path, args.eg_truth_data_path)

    sg_probs = collect_sg_probs(sg_model, sg_config, args.sg_eval_data_path, args.num_workers, device)
    eg_from_sg_probs = collapse_sg_probs_to_eg(sg_probs, sg_to_eg)
    eg_aux_probs = collect_eg_aux_probs(eg_model, eg_config, args.eg_truth_data_path, args.num_workers, device)

    if not (sg_probs.shape[0] == eg_from_sg_probs.shape[0] == eg_aux_probs.shape[0] == eg_targets.shape[0]):
        raise ValueError("Sample count mismatch between SG eval, collapsed EG eval, direct EG eval, and EG targets")

    output = {
        "sg_checkpoint": args.sg_checkpoint,
        "eg_checkpoint": args.eg_checkpoint,
        "sg_eval_data_path": args.sg_eval_data_path,
        "eg_truth_data_path": args.eg_truth_data_path,
        "n_samples": int(eg_targets.shape[0]),
        "test_alignment": alignment,
        "sg230_raw_space_group": summarize_probs(sg_probs, sg_targets),
        "sg_to_eg_collapsed": summarize_probs(eg_from_sg_probs, eg_targets),
        "direct_eg_aux": summarize_probs(eg_aux_probs, eg_targets),
    }

    if args.output_csv:
        sg_top5 = topk_labels_from_probs(sg_probs, 5, one_based=True)
        collapsed_eg_top5 = topk_labels_from_probs(eg_from_sg_probs, 5, one_based=True)
        direct_eg_top5 = topk_labels_from_probs(eg_aux_probs, 5, one_based=True)
        with open(args.output_csv, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "row_idx",
                    "true_eg",
                    "sg_top5",
                    "collapsed_eg_top5",
                    "direct_eg_aux_top5",
                ]
            )
            for idx in range(eg_targets.shape[0]):
                writer.writerow(
                    [
                        idx,
                        int(eg_targets[idx] + 1),
                        json.dumps(sg_top5[idx]),
                        json.dumps(collapsed_eg_top5[idx]),
                        json.dumps(direct_eg_top5[idx]),
                    ]
                )

    with open(args.output_json, "w") as handle:
        json.dump(output, handle, indent=2)

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
