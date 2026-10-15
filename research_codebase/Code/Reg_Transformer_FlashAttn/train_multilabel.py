import argparse
import contextlib
import io
import json
import os
import socket
import time
import unicodedata
from pathlib import Path
import sys

import h5py
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, StepLR
from torch.utils.data.distributed import DistributedSampler
from torchinfo import summary
from torchvision.ops import sigmoid_focal_loss
from tqdm import tqdm

from dataset_multilabel import get_dataloaders, get_mixed_dataloaders
from model_multilabel import adapt_input_embedding_channels, transformer_model

VIT_DIR = Path(__file__).resolve().parents[1] / "ViT_NVIDIA"
if str(VIT_DIR) not in sys.path:
    sys.path.insert(0, str(VIT_DIR))

from extinction_multilabel import (
    build_template_bank,
    decode_multilabel_logits,
    decode_split_head_logits,
    ext_group_to_multilabel_target,
    multilabel_targets_to_split_targets,
    topk_decoded_ext_groups,
    topk_decoded_split_head_ext_groups,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument("config_path", nargs="?", help="Optional positional path to the configuration file")
    parser.add_argument("--config", default=None, help="Path to the configuration file")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--node_rank", type=int, default=-1)
    parser.add_argument("--sweep_id", type=str, default="")
    parser.add_argument("--num_sweeps", type=int, default=1)
    parser.add_argument("--disable_wandb", action="store_true")
    args = parser.parse_args()
    if args.config is None:
        args.config = args.config_path
    if args.config is None:
        parser.error("the following arguments are required: --config")
    return args


def load_config(config_path):
    with open(config_path, "r") as f:
        return materialize_config(json.load(f))


def materialize_config(config):
    materialized = dict(config)
    parameters = config.get("parameters", {})
    for key, value in parameters.items():
        if not isinstance(value, dict):
            materialized[key] = value
        elif "value" in value:
            materialized[key] = value["value"]
        elif "values" in value and value["values"]:
            materialized[key] = value["values"][0]
    return materialized


def clean_summary_str(summary_str):
    normalized = unicodedata.normalize("NFKD", summary_str)
    return normalized.encode("ascii", "ignore").decode("ascii")


def get_nested_config_value(config, key, default=None):
    if key in config:
        return config[key]
    return config.get("parameters", {}).get(key, {}).get("value", default)


def get_label_mode(config):
    return get_nested_config_value(config, "label_mode", "categorical")


def get_model_output_dim(config):
    if get_label_mode(config) == "multilabel":
        return get_nested_config_value(config, "num_labels", 37)
    return get_nested_config_value(config, "num_classes", 99)


def use_aux_ext_head(config):
    return bool(get_nested_config_value(config, "use_aux_ext_head", False))


def use_split_head(config):
    return bool(get_nested_config_value(config, "use_split_head", False))


def get_aux_ext_weight(config):
    return float(get_nested_config_value(config, "aux_ext_weight", 1.0))


def get_resume_checkpoint(config):
    return get_nested_config_value(config, "resume_checkpoint", None)


def get_resume_weights_only(config):
    return bool(get_nested_config_value(config, "resume_weights_only", False))


def get_checkpoint_model_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def filter_state_dict_for_model(model_state_dict, model):
    model_ref = model.module if isinstance(model, DDP) else model
    target_state_dict = model_ref.state_dict()
    filtered_state_dict = {}
    dropped_keys = []
    for key, value in model_state_dict.items():
        target_value = target_state_dict.get(key)
        if target_value is None:
            dropped_keys.append((key, "missing_in_target"))
            continue
        if tuple(value.shape) != tuple(target_value.shape):
            dropped_keys.append((key, f"shape_mismatch {tuple(value.shape)} != {tuple(target_value.shape)}"))
            continue
        filtered_state_dict[key] = value
    return filtered_state_dict, dropped_keys


def forward_model_outputs(model, inputs, config):
    if get_label_mode(config) != "multilabel":
        logits = model(inputs)
        return logits, None
    cls_embedding = model(inputs, return_cls_embedding=True)
    model_ref = model.module if isinstance(model, DDP) else model
    if use_split_head(config):
        multilabel_logits = torch.cat(
            [model_ref.head_sys(cls_embedding), model_ref.head_lat(cls_embedding), model_ref.head_ops(cls_embedding)],
            dim=1,
        )
    else:
        multilabel_logits = model_ref.head(cls_embedding)
    aux_logits = model_ref.aux_ext_head(cls_embedding) if use_aux_ext_head(config) else None
    return multilabel_logits, aux_logits


def unpack_batch(batch, label_mode):
    if label_mode == "multilabel":
        return batch[0], batch[1], batch[2]
    return batch[0], batch[1], batch[1]


def compute_primary_loss(outputs, targets, label_mode, criterion, config, split_criteria=None):
    if label_mode != "multilabel" or not use_split_head(config):
        return criterion(outputs, targets)
    system_targets, lattice_targets, operator_targets = multilabel_targets_to_split_targets(targets)
    sys_logits = outputs[:, 0:7]
    lat_logits = outputs[:, 7:12]
    op_logits = outputs[:, 12:]
    return (
        split_criteria["system"](sys_logits, system_targets)
        + split_criteria["lattice"](lat_logits, lattice_targets)
        + split_criteria["operators"](op_logits, operator_targets)
    )


def build_operator_pos_weight(config, templates):
    if not get_nested_config_value(config, "use_operator_pos_weight", False):
        return None
    data_path = get_nested_config_value(config, "data_path", None)
    if not data_path:
        return None
    with h5py.File(data_path, "r") as h5:
        ext_groups = torch.from_numpy(h5["y_train"][:]).long()
    targets = torch.stack([ext_group_to_multilabel_target(int(ext_group), templates) for ext_group in ext_groups], dim=0)
    _, _, operator_targets = multilabel_targets_to_split_targets(targets)
    positives = operator_targets.sum(dim=0)
    total = torch.tensor(float(operator_targets.shape[0]), dtype=torch.float32)
    negatives = total - positives
    pos_weight = negatives / positives.clamp_min(1.0)
    pos_weight_mode = str(get_nested_config_value(config, "operator_pos_weight_mode", "raw")).lower()
    if pos_weight_mode == "sqrt":
        pos_weight = torch.sqrt(pos_weight)
    elif pos_weight_mode != "raw":
        raise ValueError(f"Unsupported operator_pos_weight_mode: {pos_weight_mode}")
    max_pos_weight = get_nested_config_value(config, "operator_pos_weight_max", None)
    if max_pos_weight is not None:
        pos_weight = pos_weight.clamp_max(float(max_pos_weight))
    return pos_weight.to(dtype=torch.float32)


def build_operator_loss(config, templates, device, rank):
    operator_loss_type = str(get_nested_config_value(config, "operator_loss_type", "bce")).lower()
    if operator_loss_type == "bce":
        operator_pos_weight = build_operator_pos_weight(config, templates)
        if operator_pos_weight is not None and rank == 0:
            print("Using operator pos_weight:", ", ".join(f"{v:.2f}" for v in operator_pos_weight.tolist()))
        return nn.BCEWithLogitsLoss(pos_weight=operator_pos_weight.to(device) if operator_pos_weight is not None else None)
    if operator_loss_type == "focal":
        gamma = float(get_nested_config_value(config, "operator_focal_gamma", 2.0))
        loss_weight = float(get_nested_config_value(config, "operator_focal_loss_weight", 1.0))

        def focal_operator_loss(logits, targets):
            targets = targets.float()
            per_element = sigmoid_focal_loss(logits, targets, alpha=-1.0, gamma=gamma, reduction="none")
            return loss_weight * per_element.mean()

        return focal_operator_loss
    raise ValueError(f"Unsupported operator_loss_type: {operator_loss_type}")


def compute_top_k_accuracy(output, target, k=1):
    with torch.no_grad():
        _, pred = output.topk(k, dim=1, largest=True, sorted=True)
        correct = pred.eq(target.view(-1, 1).expand_as(pred))
        return correct.any(dim=1).float().sum().item()


def wait_for_master():
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = int(os.environ.get("MASTER_PORT", "29500"))
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        print(f"[rank{rank}] Master node, skipping connectivity test.")
        return
    for attempt in range(10):
        try:
            with socket.create_connection((master_addr, int(master_port)), timeout=3):
                print(f"[rank{rank}] TCP connection to master {master_addr}:{master_port} succeeded.")
                return
        except Exception as e:
            print(f"[rank{rank}] Waiting for master... attempt {attempt+1}, error: {e}")
            time.sleep(2)
    raise SystemExit(f"[rank{rank}] TCP connection to master {master_addr}:{master_port} failed after 10 attempts.")


def broadcast_config(config, rank):
    obj_list = [config] if rank == 0 else [None]
    dist.broadcast_object_list(obj_list, src=0)
    return obj_list[0]


def train(config, train_loader, val_loader, test_loader, device, rank, local_rank, distributed):
    use_wandb = bool(config.get("use_wandb", True))
    wandb_enabled = rank == 0 and use_wandb
    if wandb_enabled:
        wandb.init(project="project", config=config)
        config = wandb.config

    label_mode = get_label_mode(config)
    sample_batch = next(iter(train_loader))
    actual_spec_length = sample_batch[0].shape[1]
    model = transformer_model(
        spec_length=actual_spec_length,
        num_output=get_model_output_dim(config),
        embed_dim=config["embed_dim"],
        ff_dim=config.get("ff_dim"),
        depth=config["depth"],
        num_heads=config["num_heads"],
        mlp_units=config.get("mlp_units", config["embed_dim"]),
        dropout=config["dropout"],
        pos_encoding=config.get("pos_encoding", "none"),
        use_flash_attn=config.get("use_flash_attn", True),
        use_mlp_head=config.get("use_mlp_head", False),
        mlp_head_hidden_dim=config.get("mlp_head_hidden_dim"),
        use_physics_pe=config.get("use_physics_pe", False),
        physics_pe_mode=config.get("physics_pe_mode", "sin2theta"),
        two_theta_min=float(config.get("two_theta_min", 5.0)),
        two_theta_max=float(config.get("two_theta_max", 90.0)),
        physics_pe_scale=float(config.get("physics_pe_scale", 1.0)),
        use_coordinate_channel=config.get("use_coordinate_channel", False),
        coordinate_mode=config.get("coordinate_mode", "sin2theta"),
        mlp_ratio=config.get("mlp_ratio"),
    )
    if label_mode == "multilabel":
        if use_split_head(config):
            model.head_sys = nn.Linear(model.embed_dim, 7)
            model.head_lat = nn.Linear(model.embed_dim, 5)
            model.head_ops = nn.Linear(model.embed_dim, 25)
        if use_aux_ext_head(config):
            model.aux_ext_head = nn.Linear(model.embed_dim, get_nested_config_value(config, "num_classes", 99))
    model = model.to(device)

    if wandb_enabled:
        wandb.watch(model, log="all", log_freq=100)
        with contextlib.redirect_stdout(io.StringIO()) as f:
            summary(model, input_size=(1, actual_spec_length))
        wandb.log({"model_summary": wandb.Html(f"<pre>{clean_summary_str(f.getvalue())}</pre>")})

    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]) if config["optimizer"] == "Adam" else optim.SGD(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = None
    if config["scheduler"] == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=config["num_epochs"], eta_min=config.get("eta_min", 1e-6))
    elif config["scheduler"] == "cosine_warm_restart":
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=config.get("T_0", 10), T_mult=config.get("T_mult", 1), eta_min=config.get("eta_min", 1e-6))
    elif config["scheduler"] == "step":
        scheduler = StepLR(optimizer, step_size=config.get("step_size", 10), gamma=config.get("decay_rate", 0.9))

    warmup_steps = int(get_nested_config_value(config, "warmup_steps", 0) or 0)
    base_lrs = [group["lr"] for group in optimizer.param_groups]
    global_step = 0
    if warmup_steps > 0:
        for group in optimizer.param_groups:
            group["lr"] = 0.0

    run_id = None
    model_dir = None
    final_model_path = None
    latest_checkpoint_path = None
    best_checkpoint_path = None
    best_val_loss = float("inf")
    start_epoch = 0

    resume_checkpoint = get_resume_checkpoint(config)
    if resume_checkpoint:
        if rank == 0:
            print(f"Resuming from checkpoint: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        resume_weights_only = get_resume_weights_only(config)
        model_state_dict = get_checkpoint_model_state_dict(checkpoint)
        adapted_state_dict = adapt_input_embedding_channels(model_state_dict, model)
        filtered_state_dict, dropped_keys = filter_state_dict_for_model(adapted_state_dict, model)
        if rank == 0 and dropped_keys:
            print("Dropping incompatible checkpoint tensors during resume:")
            for key, reason in dropped_keys:
                print(f"  - {key}: {reason}")
        model.load_state_dict(filtered_state_dict, strict=False)
        if (
            isinstance(checkpoint, dict)
            and "optimizer_state_dict" in checkpoint
            and checkpoint["optimizer_state_dict"] is not None
            and not resume_weights_only
        ):
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        elif rank == 0 and resume_weights_only:
            print("Skipping optimizer/scheduler resume because resume_weights_only is enabled")
        if scheduler is not None and isinstance(checkpoint, dict) and checkpoint.get("scheduler_state_dict") is not None and not resume_weights_only:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if isinstance(checkpoint, dict) and not resume_weights_only:
            start_epoch = int(checkpoint.get("epoch", 0))
            best_val_loss = float(checkpoint.get("val_loss", float("inf")))
            global_step = int(checkpoint.get("global_step", start_epoch * len(train_loader)))

    template_bank = None
    ext_group_order = None
    templates = None
    if label_mode == "multilabel":
        template_bank, ext_group_order, templates = build_template_bank(
            canonical_table_path=get_nested_config_value(config, "canonical_table_path", None),
            final_table_path=get_nested_config_value(config, "final_table_path", None),
            sg_lookup_path=get_nested_config_value(config, "sg_lookup_path", None),
        )

    criterion = nn.BCEWithLogitsLoss() if label_mode == "multilabel" else nn.CrossEntropyLoss()
    aux_criterion = nn.CrossEntropyLoss() if label_mode == "multilabel" and use_aux_ext_head(config) else None
    split_criteria = None
    if label_mode == "multilabel" and use_split_head(config):
        split_criteria = {
            "system": nn.CrossEntropyLoss(),
            "lattice": nn.CrossEntropyLoss(),
            "operators": build_operator_loss(config, templates, device, rank),
        }

    if rank == 0:
        run_id = wandb.run.id if wandb_enabled else f"local_{int(time.time())}"
        model_dir = config["model_path"]
        os.makedirs(model_dir, exist_ok=True)
        final_model_path = os.path.join(model_dir, f"xrd_model_{run_id}.pth")
        latest_checkpoint_path = os.path.join(model_dir, f"xrd_model_{run_id}_latest.pth")
        best_checkpoint_path = os.path.join(model_dir, f"xrd_model_{run_id}_best.pth")

    for epoch in range(start_epoch, config["num_epochs"]):
        train_sampler = getattr(train_loader, "sampler", None)
        if distributed and isinstance(train_sampler, DistributedSampler):
            train_sampler.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['num_epochs']}", disable=(rank != 0))
        for batch in progress_bar:
            inputs, targets, ext_targets = unpack_batch(batch, label_mode)
            inputs = inputs.to(device)
            targets = targets.to(device)
            ext_targets = ext_targets.to(device)
            optimizer.zero_grad()
            outputs, aux_outputs = forward_model_outputs(model, inputs, config)
            loss = compute_primary_loss(outputs, targets, label_mode, criterion, config, split_criteria=split_criteria)
            if aux_criterion is not None:
                loss = loss + get_aux_ext_weight(config) * aux_criterion(aux_outputs, ext_targets)
            loss.backward()
            optimizer.step()
            global_step += 1
            if warmup_steps > 0 and global_step <= warmup_steps:
                warmup_scale = global_step / warmup_steps
                for group, base_lr in zip(optimizer.param_groups, base_lrs):
                    group["lr"] = base_lr * warmup_scale
            elif warmup_steps > 0 and global_step == warmup_steps + 1:
                for group, base_lr in zip(optimizer.param_groups, base_lrs):
                    group["lr"] = base_lr
            if scheduler is not None and config["scheduler"] == "cosine_warm_restart" and global_step > warmup_steps:
                scheduler.step(epoch + global_step / max(len(train_loader), 1))
            running_loss += loss.item()

        running_loss_tensor = torch.tensor(running_loss, device=device)
        if distributed:
            dist.all_reduce(running_loss_tensor, op=dist.ReduceOp.SUM)
        avg_train_loss = running_loss_tensor.item() / (len(train_loader) * (dist.get_world_size() if distributed else 1))
        print(f"Epoch {epoch + 1}/{config['num_epochs']}, Train Loss: {avg_train_loss}")

        model.eval()
        val_loss = 0.0
        val_total = 0
        val_correct = 0
        val_aux_correct = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets, ext_targets = unpack_batch(batch, label_mode)
                inputs = inputs.to(device)
                targets = targets.to(device)
                ext_targets = ext_targets.to(device)
                outputs, aux_outputs = forward_model_outputs(model, inputs, config)
                loss = compute_primary_loss(outputs, targets, label_mode, criterion, config, split_criteria=split_criteria)
                if aux_criterion is not None:
                    loss = loss + get_aux_ext_weight(config) * aux_criterion(aux_outputs, ext_targets)
                val_loss += loss.item()
                if label_mode == "multilabel":
                    pred_ext = decode_split_head_logits(outputs, template_bank, ext_group_order) if use_split_head(config) else decode_multilabel_logits(outputs, template_bank, ext_group_order)
                    val_correct += (pred_ext == (ext_targets + 1)).sum().item()
                    val_total += ext_targets.size(0)
                    if aux_outputs is not None:
                        val_aux_correct += (torch.argmax(aux_outputs, dim=1) == ext_targets).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        if distributed:
            val_stats = torch.tensor([val_loss, float(val_correct), float(val_total), float(val_aux_correct)], device=device, dtype=torch.float64)
            dist.all_reduce(val_stats, op=dist.ReduceOp.SUM)
            avg_val_loss = val_stats[0].item() / (len(val_loader) * dist.get_world_size())
            val_correct = int(val_stats[1].item())
            val_total = int(val_stats[2].item())
            val_aux_correct = int(val_stats[3].item())

        if scheduler is not None and config["scheduler"] != "cosine_warm_restart":
            scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        if rank == 0:
            print(f"Epoch {epoch + 1}/{config['num_epochs']}, Validation Loss: {avg_val_loss}")
            if label_mode == "multilabel" and val_total > 0:
                print(f"Epoch {epoch + 1}/{config['num_epochs']}, Validation Decode Accuracy: {100 * val_correct / val_total:.2f}%")
                if aux_criterion is not None:
                    print(f"Epoch {epoch + 1}/{config['num_epochs']}, Validation Aux EG Accuracy: {100 * val_aux_correct / val_total:.2f}%")

        if wandb_enabled:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "lr": current_lr,
                **({"val_decode_accuracy": 100 * val_correct / val_total} if label_mode == "multilabel" and val_total > 0 else {}),
                **({"val_aux_ext_accuracy": 100 * val_aux_correct / val_total} if aux_criterion is not None and val_total > 0 else {}),
            })

        if rank == 0:
            model_state = model.module.state_dict() if distributed else model.state_dict()
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                "aux_ext_head_state_dict": (model.module.aux_ext_head.state_dict() if distributed and aux_criterion is not None else model.aux_ext_head.state_dict() if aux_criterion is not None else None),
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "config": dict(config),
                "run_id": run_id,
                "global_step": global_step,
            }
            torch.save(checkpoint, latest_checkpoint_path)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(checkpoint, best_checkpoint_path)

    if rank == 0:
        model_state = model.module.state_dict() if distributed else model.state_dict()
        torch.save(model_state, final_model_path)
        if wandb_enabled:
            wandb.save(final_model_path)
            wandb.save(latest_checkpoint_path)
            wandb.save(best_checkpoint_path)
        print("After model save")

    model.eval()
    test_loss, correct, total = 0.0, 0.0, 0.0
    top1_correct, top3_correct, top5_correct = 0.0, 0.0, 0.0
    with torch.no_grad():
        for batch in test_loader:
            inputs, targets, ext_targets = unpack_batch(batch, label_mode)
            inputs = inputs.to(device)
            targets = targets.to(device)
            ext_targets = ext_targets.to(device)
            outputs, aux_outputs = forward_model_outputs(model, inputs, config)
            loss = compute_primary_loss(outputs, targets, label_mode, criterion, config, split_criteria=split_criteria)
            if aux_criterion is not None:
                loss = loss + get_aux_ext_weight(config) * aux_criterion(aux_outputs, ext_targets)
            test_loss += loss.item()
            if label_mode == "multilabel":
                if use_split_head(config):
                    top1_correct += compute_split_head_topk_accuracy(outputs, ext_targets, template_bank, ext_group_order, 1)
                    top3_correct += compute_split_head_topk_accuracy(outputs, ext_targets, template_bank, ext_group_order, 3)
                    top5_correct += compute_split_head_topk_accuracy(outputs, ext_targets, template_bank, ext_group_order, 5)
                else:
                    top1_correct += compute_multilabel_topk_accuracy(outputs, ext_targets, template_bank, ext_group_order, 1)
                    top3_correct += compute_multilabel_topk_accuracy(outputs, ext_targets, template_bank, ext_group_order, 3)
                    top5_correct += compute_multilabel_topk_accuracy(outputs, ext_targets, template_bank, ext_group_order, 5)
                total += ext_targets.size(0)
            else:
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                top1_correct += compute_top_k_accuracy(outputs, targets, 1)
                top3_correct += compute_top_k_accuracy(outputs, targets, 3)
                top5_correct += compute_top_k_accuracy(outputs, targets, 5)
    test_accuracy = 100 * correct / total if label_mode != "multilabel" else 100 * top1_correct / total
    top1_accuracy = 100 * top1_correct / total
    top3_accuracy = 100 * top3_correct / total
    top5_accuracy = 100 * top5_correct / total
    if rank == 0:
        print("After test")
        print(f"Total samples: {int(total)}")
        print(f"Test Accuracy: {test_accuracy:.2f}%")
        print(f"Top-1 Accuracy: {top1_accuracy:.2f}%")
        print(f"Top-3 Accuracy: {top3_accuracy:.2f}%")
        print(f"Top-5 Accuracy: {top5_accuracy:.2f}%")
        if wandb_enabled:
            wandb.log({
                "total_samples": int(total),
                "test_loss": test_loss,
                "test_accuracy_top1": top1_accuracy,
                "test_accuracy_top3": top3_accuracy,
                "test_accuracy_top5": top5_accuracy,
            })
            wandb.finish()


def compute_multilabel_topk_accuracy(logits, ext_targets, template_bank, ext_group_order, k):
    decoded = topk_decoded_ext_groups(logits, template_bank, ext_group_order, k=k)
    return (decoded == (ext_targets + 1).unsqueeze(1)).any(dim=1).float().sum().item()


def compute_split_head_topk_accuracy(logits, ext_targets, template_bank, ext_group_order, k):
    decoded = topk_decoded_split_head_ext_groups(logits, template_bank, ext_group_order, k=k)
    return (decoded == (ext_targets + 1).unsqueeze(1)).any(dim=1).float().sum().item()


def main():
    args = parse_args()
    wait_for_master()
    distributed = args.distributed or int(os.environ.get("WORLD_SIZE", "1")) > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0")) if distributed else 0
    rank = int(os.environ.get("RANK", "0")) if distributed else 0
    if distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    config = load_config(args.config)
    if distributed:
        config = broadcast_config(config, rank)

    if "po_data_path" in config and config.get("po_data_path"):
        train_loader, val_loader, test_loader, _, _ = get_mixed_dataloaders(
            standard_h5_file=config["data_path"],
            po_h5_file=config["po_data_path"],
            batch_size=config["batch_size"],
            world_size=dist.get_world_size() if distributed else 1,
            rank=rank,
            num_workers=config["num_workers"],
            num_classes=config.get("num_classes", 99),
            prefetch_factor=config.get("prefetch_factor", 4),
            start_col=config["start_col"],
            end_col=config["end_col"],
            label_mode=config.get("label_mode", "multilabel"),
            canonical_table_path=config.get("canonical_table_path"),
            final_table_path=config.get("final_table_path"),
            sg_lookup_path=config.get("sg_lookup_path"),
            max_samples_train=config.get("max_samples_train"),
            max_samples_val=config.get("max_samples_val"),
            max_samples_test=config.get("max_samples_test"),
            po_train_ratio=config.get("po_train_ratio", 0.2),
            po_val_ratio=config.get("po_val_ratio", 0.2),
            po_test_ratio=config.get("po_test_ratio", 0.2),
            mixed_seed=config.get("mixed_seed", 1337),
        )
    else:
        train_loader, val_loader, test_loader, _, _ = get_dataloaders(
            h5_file=config["data_path"],
            batch_size=config["batch_size"],
            world_size=dist.get_world_size() if distributed else 1,
            rank=rank,
            num_workers=config["num_workers"],
            num_classes=config.get("num_classes", 99),
            prefetch_factor=config.get("prefetch_factor", 4),
            start_col=config["start_col"],
            end_col=config["end_col"],
            label_mode=config.get("label_mode", "multilabel"),
            canonical_table_path=config.get("canonical_table_path"),
            final_table_path=config.get("final_table_path"),
            sg_lookup_path=config.get("sg_lookup_path"),
            max_samples_train=config.get("max_samples_train"),
            max_samples_val=config.get("max_samples_val"),
            max_samples_test=config.get("max_samples_test"),
        )

    train(config, train_loader, val_loader, test_loader, device, rank, local_rank, distributed)
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
