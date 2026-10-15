#!/usr/bin/env python3
import argparse
import json

import h5py
import torch
import torch.nn as nn
from tqdm import tqdm

from dataset import get_dataloaders_test
from extinction_multilabel import (
    build_template_bank,
    score_split_head_templates,
    template_mask_from_ext_groups,
)
from model import VIT_model, adapt_patch_embed_input_channels


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate cross-model fusion and temperature sweeps on the 325 benchmark.")
    parser.add_argument("--split-checkpoint", required=True)
    parser.add_argument("--split-config", required=True)
    parser.add_argument("--aux-checkpoint", required=True)
    parser.add_argument("--aux-config", required=True)
    parser.add_argument("--eval-data-path", required=True)
    parser.add_argument("--prior-data-path", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--temps", nargs="*", type=float, default=[1.0, 1.5, 2.0, 3.0, 5.0])
    parser.add_argument("--alphas", nargs="*", type=float, default=[0.25, 0.50, 0.75])
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def load_config(config_path, checkpoint, eval_data_path):
    config = dict(checkpoint.get("config", {})) if isinstance(checkpoint, dict) else {}
    with open(config_path, "r") as handle:
        config.update(json.load(handle))
    config["data_path"] = eval_data_path
    return config


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


def init_metrics():
    return {"top1": 0, "top3": 0, "top5": 0}


def update_metrics(metric_dict, topk_indices_zero_based, targets_zero_based):
    metric_dict["top1"] += (topk_indices_zero_based[:, 0] == targets_zero_based).sum().item()
    metric_dict["top3"] += (topk_indices_zero_based[:, :3] == targets_zero_based.unsqueeze(1)).any(dim=1).sum().item()
    metric_dict["top5"] += (topk_indices_zero_based[:, :5] == targets_zero_based.unsqueeze(1)).any(dim=1).sum().item()


def as_pct(metric_dict, total):
    return {k: 100.0 * v / total for k, v in metric_dict.items()}


def main():
    args = parse_args()
    device = torch.device(args.device)

    split_checkpoint = torch.load(args.split_checkpoint, map_location="cpu")
    aux_checkpoint = torch.load(args.aux_checkpoint, map_location="cpu")
    split_config = load_config(args.split_config, split_checkpoint, args.eval_data_path)
    aux_config = load_config(args.aux_config, aux_checkpoint, args.eval_data_path)

    split_model = build_model(split_config, split_checkpoint, device)
    aux_model = build_model(aux_config, aux_checkpoint, device)

    num_workers = split_config.get("num_workers", 8)
    prefetch_factor = split_config.get("prefetch_factor", 4)
    _, _, test_loader, _, _ = get_dataloaders_test(
        h5_file=split_config["data_path"],
        batch_size=split_config["batch_size"],
        num_workers=num_workers,
        num_classes=split_config.get("num_classes", 99),
        prefetch_factor=prefetch_factor,
        start_col=split_config["start_col"],
        end_col=split_config["end_col"],
        label_mode=split_config.get("label_mode", "multilabel"),
        canonical_table_path=split_config.get("canonical_table_path"),
        final_table_path=split_config.get("final_table_path"),
        sg_lookup_path=split_config.get("sg_lookup_path"),
    )

    template_bank, ext_group_order, _ = build_template_bank(
        canonical_table_path=split_config.get("canonical_table_path"),
        final_table_path=split_config.get("final_table_path"),
        sg_lookup_path=split_config.get("sg_lookup_path"),
    )
    template_bank = template_bank.to(device)
    ext_group_tensor = torch.tensor(ext_group_order, device=device, dtype=torch.long) - 1
    seen_ext_groups = load_seen_ext_groups(args.prior_data_path)
    seen_mask = template_mask_from_ext_groups(ext_group_order, seen_ext_groups).to(device)
    log_priors = load_log_priors(args.prior_data_path, ext_group_order).to(device)

    results = {}
    for temp in args.temps:
        results[f"aux_temp_{temp:g}"] = init_metrics()
        for alpha in args.alphas:
            results[f"frankenstein_temp_{temp:g}_alpha_{alpha:.2f}"] = init_metrics()

    total = 0
    with torch.no_grad():
        for inputs, _, ext_targets in tqdm(test_loader, desc="Cross-evaluating"):
            inputs = inputs.to(device)
            ext_targets = ext_targets.to(device)
            total += ext_targets.size(0)

            split_cls = split_model(inputs, return_cls_embedding=True)
            split_logits = torch.cat(
                [split_model.head_sys(split_cls), split_model.head_lat(split_cls), split_model.head_ops(split_cls)],
                dim=1,
            )
            split_scores = score_split_head_templates(
                split_logits,
                template_bank,
                allowed_mask=seen_mask,
                log_priors=log_priors,
                prior_weight=1.0,
                impossible_operator_masking=True,
                impossible_operator_prob=1e-3,
            )
            split_scores = torch.where(
                torch.isfinite(split_scores),
                split_scores,
                torch.full_like(split_scores, -1e9),
            )

            aux_cls = aux_model(inputs, return_cls_embedding=True)
            aux_logits = aux_model.aux_ext_head(aux_cls)

            for temp in args.temps:
                scaled_aux = aux_logits / temp
                bayes_aux = scaled_aux + log_priors.unsqueeze(0)
                aux_topk = torch.topk(bayes_aux, k=5, dim=1).indices
                update_metrics(results[f"aux_temp_{temp:g}"], aux_topk, ext_targets)

                for alpha in args.alphas:
                    fused_scores = alpha * split_scores + (1.0 - alpha) * bayes_aux
                    fused_topk_idx = torch.topk(fused_scores, k=5, dim=1).indices
                    fused_topk = ext_group_tensor[fused_topk_idx]
                    update_metrics(results[f"frankenstein_temp_{temp:g}_alpha_{alpha:.2f}"], fused_topk, ext_targets)

    summary = {
        "split_checkpoint": args.split_checkpoint,
        "aux_checkpoint": args.aux_checkpoint,
        "eval_data_path": args.eval_data_path,
        "prior_data_path": args.prior_data_path,
        "seen_train_eg_count": len(seen_ext_groups),
        "total_test_samples": total,
        "results": {name: as_pct(metrics, total) for name, metrics in results.items()},
    }

    with open(args.output_json, "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
