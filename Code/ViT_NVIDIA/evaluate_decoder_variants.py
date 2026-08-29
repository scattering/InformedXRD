import argparse
import collections
import json

import h5py
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from dataset import get_dataloaders_test
from extinction_multilabel import (
    build_template_bank,
    decode_multilabel_logits,
    decode_split_head_logits,
    score_split_head_templates,
    template_mask_from_ext_groups,
    topk_decoded_ext_groups,
    topk_decoded_split_head_ext_groups,
)
from model import VIT_model, adapt_patch_embed_input_channels


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate decoder variants for a multilabel checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path")
    parser.add_argument("--config", default=None, help="Optional config override path")
    parser.add_argument("--eval-data-path", default=None, help="Optional HDF5 path to evaluate on instead of config data_path")
    parser.add_argument("--prior-data-path", default=None, help="Optional HDF5 path providing y_train for seen-class mask and priors")
    parser.add_argument("--num-workers", type=int, default=None, help="Optional DataLoader worker override")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--aux-temperature",
        type=float,
        default=1.0,
        help="Temperature applied to aux logits before Bayesian prior fusion.",
    )
    parser.add_argument(
        "--group-keys",
        nargs="*",
        default=[],
        help="Optional HDF5 dataset names to use for subgroup reporting, e.g. severity fit_bucket",
    )
    parser.add_argument(
        "--save-aux-logits",
        default=None,
        help="Optional .npz path: dump per-case aux_ext_head logits + ext_targets for offline analysis.",
    )
    return parser.parse_args()


def load_config(args, checkpoint):
    config = dict(checkpoint.get("config", {})) if isinstance(checkpoint, dict) else {}
    if args.config:
        with open(args.config, "r") as handle:
            config.update(json.load(handle))
    if args.eval_data_path:
        config["data_path"] = args.eval_data_path
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


def compute_topk(decoded_topk, targets_zero_based):
    targets_one_based = targets_zero_based.unsqueeze(1) + 1
    matches = decoded_topk == targets_one_based
    return matches.any(dim=1).sum().item()


def compute_topk_from_indices(topk_indices, targets_zero_based):
    return (topk_indices == targets_zero_based.unsqueeze(1)).any(dim=1).sum().item()


def init_metric_dict():
    return {"top1": 0.0, "top3": 0.0, "top5": 0.0, "n": 0}


def init_group_results(names):
    return collections.defaultdict(lambda: {name: init_metric_dict() for name in names})


def update_group_metrics(metric_dict, topk_indices, target):
    target = int(target)
    metric_dict["n"] += 1
    if int(topk_indices[0]) == target:
        metric_dict["top1"] += 1
    if target in [int(v) for v in topk_indices[:3]]:
        metric_dict["top3"] += 1
    if target in [int(v) for v in topk_indices[:5]]:
        metric_dict["top5"] += 1


def load_group_arrays(h5_path, group_keys):
    if not group_keys:
        return {}
    group_arrays = {}
    with h5py.File(h5_path, "r") as handle:
        for key in group_keys:
            if key not in handle:
                raise KeyError(f"Requested group key '{key}' not found in {h5_path}")
            values = handle[key][:]
            if getattr(values.dtype, "kind", "") in ("S", "O", "U"):
                values = [v.decode() if isinstance(v, bytes) else str(v) for v in values.tolist()]
            else:
                values = [str(v) for v in values.tolist()]
            group_arrays[key] = values
    return group_arrays


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


def main():
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = load_config(args, checkpoint)
    device = torch.device(args.device)

    model = build_model(config, checkpoint, device)

    num_workers = config.get("num_workers", 8) if args.num_workers is None else args.num_workers
    prefetch_factor = config.get("prefetch_factor", 4)
    if num_workers == 0:
        prefetch_factor = None

    _, _, test_loader, _, _ = get_dataloaders_test(
        h5_file=config["data_path"],
        batch_size=config["batch_size"],
        num_workers=num_workers,
        num_classes=config.get("num_classes", 99),
        prefetch_factor=prefetch_factor,
        start_col=config["start_col"],
        end_col=config["end_col"],
        label_mode=config.get("label_mode", "multilabel"),
        canonical_table_path=config.get("canonical_table_path"),
        final_table_path=config.get("final_table_path"),
        sg_lookup_path=config.get("sg_lookup_path"),
        max_samples_test=config.get("max_samples_test"),
    )

    template_bank, ext_group_order, _ = build_template_bank(
        canonical_table_path=config.get("canonical_table_path"),
        final_table_path=config.get("final_table_path"),
        sg_lookup_path=config.get("sg_lookup_path"),
    )

    checkpoint_data_path = checkpoint.get("config", {}).get("data_path") if isinstance(checkpoint, dict) else None
    prior_data_path = args.prior_data_path or checkpoint_data_path or config["data_path"]

    seen_ext_groups = load_seen_ext_groups(prior_data_path)
    seen_mask = template_mask_from_ext_groups(ext_group_order, seen_ext_groups)
    log_priors = load_log_priors(prior_data_path, ext_group_order)
    group_arrays = load_group_arrays(config["data_path"], args.group_keys)

    if config.get("use_split_head", False):
        decoders = {
            "split_map_all99": {"allowed_mask": None},
            "split_map_seen70": {"allowed_mask": seen_mask},
            "split_bayes025_seen70": {"allowed_mask": seen_mask, "log_priors": log_priors, "prior_weight": 0.25},
            "split_bayes050_seen70": {"allowed_mask": seen_mask, "log_priors": log_priors, "prior_weight": 0.50},
            "split_bayes100_seen70": {"allowed_mask": seen_mask, "log_priors": log_priors, "prior_weight": 1.00},
            "split_masked_bayes100_seen70": {
                "allowed_mask": seen_mask,
                "log_priors": log_priors,
                "prior_weight": 1.00,
                "impossible_operator_masking": True,
                "impossible_operator_prob": 1e-3,
            },
        }
    else:
        decoders = {
            "euclidean_all99": {"metric": "euclidean", "allowed_mask": None, "hierarchical": False},
            "bernoulli_all99": {"metric": "bernoulli", "allowed_mask": None, "hierarchical": False},
            "bernoulli_seen70": {"metric": "bernoulli", "allowed_mask": seen_mask, "hierarchical": False},
            "hier_bernoulli_all99": {"metric": "bernoulli", "allowed_mask": None, "hierarchical": True},
            "hier_bernoulli_seen70": {"metric": "bernoulli", "allowed_mask": seen_mask, "hierarchical": True},
            "bayes025_hier_seen70": {"metric": "bernoulli", "allowed_mask": seen_mask, "hierarchical": True, "log_priors": log_priors, "prior_weight": 0.25},
            "bayes050_hier_seen70": {"metric": "bernoulli", "allowed_mask": seen_mask, "hierarchical": True, "log_priors": log_priors, "prior_weight": 0.50},
            "bayes100_hier_seen70": {"metric": "bernoulli", "allowed_mask": seen_mask, "hierarchical": True, "log_priors": log_priors, "prior_weight": 1.00},
        }

    decoder_names = list(decoders.keys())
    if hasattr(model, "aux_ext_head"):
        decoder_names.extend(["aux_head", "bayesian_aux_head"])
        if config.get("use_split_head", False):
            decoder_names.extend(
                [
                    "fused_split_masked_bayes100_aux_bayes_alpha025",
                    "fused_split_masked_bayes100_aux_bayes_alpha050",
                    "fused_split_masked_bayes100_aux_bayes_alpha075",
                ]
            )
    grouped_results = {group_key: init_group_results(decoder_names) for group_key in args.group_keys}

    results = {name: {"top1": 0.0, "top3": 0.0, "top5": 0.0} for name in decoders}
    aux_results = {"top1": 0.0, "top3": 0.0, "top5": 0.0}
    bayesian_aux_results = {"top1": 0.0, "top3": 0.0, "top5": 0.0}
    fused_results = {
        "fused_split_masked_bayes100_aux_bayes_alpha025": {"top1": 0.0, "top3": 0.0, "top5": 0.0},
        "fused_split_masked_bayes100_aux_bayes_alpha050": {"top1": 0.0, "top3": 0.0, "top5": 0.0},
        "fused_split_masked_bayes100_aux_bayes_alpha075": {"top1": 0.0, "top3": 0.0, "top5": 0.0},
    }
    total = 0
    aux_logit_dump = [] if args.save_aux_logits else None
    aux_target_dump = [] if args.save_aux_logits else None
    ml_logit_dump = [] if args.save_aux_logits else None

    with torch.no_grad():
        sample_offset = 0
        for inputs, _, ext_targets in tqdm(test_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            ext_targets = ext_targets.to(device)
            cls_embedding = model(inputs, return_cls_embedding=True)
            if config.get("use_split_head", False):
                logits = torch.cat(
                    [model.head_sys(cls_embedding), model.head_lat(cls_embedding), model.head_ops(cls_embedding)],
                    dim=1,
                )
            else:
                logits = model.head(cls_embedding)
            aux_logits = model.aux_ext_head(cls_embedding) if hasattr(model, "aux_ext_head") else None

            total += ext_targets.size(0)
            if aux_logit_dump is not None and aux_logits is not None:
                aux_logit_dump.append(aux_logits.detach().cpu().numpy())
                aux_target_dump.append(ext_targets.detach().cpu().numpy())
                ml_logit_dump.append(logits.detach().cpu().numpy())

            for name, kwargs in decoders.items():
                if config.get("use_split_head", False):
                    decoded_top1 = decode_split_head_logits(logits, template_bank, ext_group_order, **kwargs) - 1
                    decoded_top3 = topk_decoded_split_head_ext_groups(logits, template_bank, ext_group_order, k=3, **kwargs)
                    decoded_top5 = topk_decoded_split_head_ext_groups(logits, template_bank, ext_group_order, k=5, **kwargs)
                else:
                    decoded_top1 = decode_multilabel_logits(logits, template_bank, ext_group_order, **kwargs) - 1
                    decoded_top3 = topk_decoded_ext_groups(logits, template_bank, ext_group_order, k=3, **kwargs)
                    decoded_top5 = topk_decoded_ext_groups(logits, template_bank, ext_group_order, k=5, **kwargs)
                results[name]["top1"] += (decoded_top1 == ext_targets).sum().item()
                results[name]["top3"] += compute_topk(decoded_top3, ext_targets)
                results[name]["top5"] += compute_topk(decoded_top5, ext_targets)
                if grouped_results:
                    top5_zero_based = decoded_top5 - 1
                    for row in range(ext_targets.size(0)):
                        target = int(ext_targets[row].item())
                        for group_key, values in group_arrays.items():
                            group_value = values[sample_offset + row]
                            update_group_metrics(grouped_results[group_key][group_value][name], top5_zero_based[row].tolist(), target)

            if aux_logits is not None:
                aux_topk = torch.topk(aux_logits, k=5, dim=1).indices
                aux_results["top1"] += (aux_topk[:, 0] == ext_targets).sum().item()
                aux_results["top3"] += (aux_topk[:, :3] == ext_targets.unsqueeze(1)).any(dim=1).sum().item()
                aux_results["top5"] += (aux_topk[:, :5] == ext_targets.unsqueeze(1)).any(dim=1).sum().item()
                scaled_aux_logits = aux_logits / args.aux_temperature
                bayes_aux_logits = scaled_aux_logits + log_priors.to(device=aux_logits.device, dtype=aux_logits.dtype).unsqueeze(0)
                bayes_aux_topk = torch.topk(bayes_aux_logits, k=5, dim=1).indices
                bayesian_aux_results["top1"] += (bayes_aux_topk[:, 0] == ext_targets).sum().item()
                bayesian_aux_results["top3"] += (bayes_aux_topk[:, :3] == ext_targets.unsqueeze(1)).any(dim=1).sum().item()
                bayesian_aux_results["top5"] += (bayes_aux_topk[:, :5] == ext_targets.unsqueeze(1)).any(dim=1).sum().item()
                if grouped_results:
                    for row in range(ext_targets.size(0)):
                        target = int(ext_targets[row].item())
                        aux_row = aux_topk[row].tolist()
                        bayes_row = bayes_aux_topk[row].tolist()
                        for group_key, values in group_arrays.items():
                            group_value = values[sample_offset + row]
                            update_group_metrics(grouped_results[group_key][group_value]["aux_head"], aux_row, target)
                            update_group_metrics(grouped_results[group_key][group_value]["bayesian_aux_head"], bayes_row, target)

                if config.get("use_split_head", False):
                    split_scores = score_split_head_templates(
                        logits,
                        template_bank,
                        allowed_mask=seen_mask,
                        log_priors=log_priors,
                        prior_weight=1.0,
                        impossible_operator_masking=True,
                        impossible_operator_prob=1e-3,
                    )
                    aux_scores = bayes_aux_logits
                    # Preserve masked template exclusions and avoid NaN/Inf propagation
                    # when fusing split-template scores with aux-head logits.
                    split_scores = torch.where(
                        torch.isfinite(split_scores),
                        split_scores,
                        torch.full_like(split_scores, -1e9),
                    )
                    ext_group_tensor = torch.tensor(ext_group_order, device=logits.device, dtype=torch.long) - 1
                    for alpha, name in (
                        (0.25, "fused_split_masked_bayes100_aux_bayes_alpha025"),
                        (0.50, "fused_split_masked_bayes100_aux_bayes_alpha050"),
                        (0.75, "fused_split_masked_bayes100_aux_bayes_alpha075"),
                    ):
                        fused_scores = alpha * split_scores + (1.0 - alpha) * aux_scores
                        fused_topk_idx = torch.topk(fused_scores, k=5, dim=1).indices
                        fused_topk = ext_group_tensor[fused_topk_idx]
                        fused_results[name]["top1"] += (fused_topk[:, 0] == ext_targets).sum().item()
                        fused_results[name]["top3"] += compute_topk_from_indices(fused_topk[:, :3], ext_targets)
                        fused_results[name]["top5"] += compute_topk_from_indices(fused_topk[:, :5], ext_targets)
                        if grouped_results:
                            for row in range(ext_targets.size(0)):
                                target = int(ext_targets[row].item())
                                fused_row = fused_topk[row].tolist()
                                for group_key, values in group_arrays.items():
                                    group_value = values[sample_offset + row]
                                    update_group_metrics(grouped_results[group_key][group_value][name], fused_row, target)

            sample_offset += ext_targets.size(0)

    if aux_logit_dump is not None and aux_logit_dump:
        np.savez_compressed(
            args.save_aux_logits,
            aux_logits=np.concatenate(aux_logit_dump, axis=0),
            multilabel_logits=np.concatenate(ml_logit_dump, axis=0),
            ext_targets=np.concatenate(aux_target_dump, axis=0),
            log_priors=log_priors.detach().cpu().numpy(),
            seen_ext_groups=np.array(sorted(seen_ext_groups), dtype=np.int64),
            ext_group_order=np.array(ext_group_order, dtype=np.int64),
        )
        print(f"Saved per-case aux logits to {args.save_aux_logits}")

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Evaluation data: {config['data_path']}")
    print(f"Prior/seen-class source: {prior_data_path}")
    print(f"Seen train EG count: {len(seen_ext_groups)}")
    print(f"Aux temperature: {args.aux_temperature}")
    print(f"Total test samples: {total}")
    for name, metrics in results.items():
        print(
            f"{name}: top1={100.0 * metrics['top1'] / total:.2f}% "
            f"top3={100.0 * metrics['top3'] / total:.2f}% "
            f"top5={100.0 * metrics['top5'] / total:.2f}%"
        )
    if total > 0 and any(v > 0 for v in aux_results.values()):
        print(
            f"aux_head: top1={100.0 * aux_results['top1'] / total:.2f}% "
            f"top3={100.0 * aux_results['top3'] / total:.2f}% "
            f"top5={100.0 * aux_results['top5'] / total:.2f}%"
        )
        print(
            f"bayesian_aux_head: top1={100.0 * bayesian_aux_results['top1'] / total:.2f}% "
            f"top3={100.0 * bayesian_aux_results['top3'] / total:.2f}% "
            f"top5={100.0 * bayesian_aux_results['top5'] / total:.2f}%"
        )
        if config.get("use_split_head", False):
            for name, metrics in fused_results.items():
                print(
                    f"{name}: top1={100.0 * metrics['top1'] / total:.2f}% "
                    f"top3={100.0 * metrics['top3'] / total:.2f}% "
                    f"top5={100.0 * metrics['top5'] / total:.2f}%"
                )
    for group_key, per_value in grouped_results.items():
        print(f"GROUP_KEY {group_key}")
        for group_value, metric_map in per_value.items():
            print(f"GROUP {group_value}")
            for name in decoder_names:
                metrics = metric_map[name]
                if metrics["n"] == 0:
                    continue
                print(
                    f"{name}: top1={100.0 * metrics['top1'] / metrics['n']:.2f}% "
                    f"top3={100.0 * metrics['top3'] / metrics['n']:.2f}% "
                    f"top5={100.0 * metrics['top5'] / metrics['n']:.2f}% "
                    f"n={metrics['n']}"
                )
            print()


if __name__ == "__main__":
    main()
