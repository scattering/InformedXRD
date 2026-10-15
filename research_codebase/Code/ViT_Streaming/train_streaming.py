"""
train_streaming.py
------------------
Streaming-only training entrypoint.
Patterns generated in-process via xrd_generator.worker() + Pool.
No subprocess, no JSON, no HDF5 during training.

Checkpoint saved at end of every epoch. Only latest is kept.
Resume via --resume_checkpoint.

Launch:
  torchrun --nnodes=1 --nproc_per_node=4 train_streaming.py \
      --config sweep_config_streaming.json                  \
      --sweep_id <id> --num_sweeps 5                        \
      --dataset_size 20m --val_size 200k                    \
      --checkpoint_dir $SCRATCH/checkpoints

  # Resume:
  torchrun ... train_streaming.py ... \
      --resume_checkpoint $SCRATCH/checkpoints/ckpt_epoch_004.pt
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import socket
import time
import unicodedata

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
from torchinfo import summary
from tqdm import tqdm
import argparse

from model import VIT_model
from streaming_dataset import get_streaming_dataloader, NUM_EXTINCTION_GROUPS


# -----------------------------------------------------------------------
# Dataset size parser  — freeform: 218k, 1.5m, 2000000, 10m, etc.
# -----------------------------------------------------------------------

def parse_dataset_size(size_str: str) -> int:
    """
    Parse a human-readable dataset size into an integer.

    Accepts:
      - Plain integers:          '218000', '1000000'
      - k/K suffix (thousands):  '218k', '500K'
      - m/M suffix (millions):   '1m', '1.5M', '20m'
      - g/G suffix (billions):   '1g', '2G'

    Examples:
      '218k'  → 218_000
      '1.5m'  → 1_500_000
      '20m'   → 20_000_000
      '500k'  → 500_000
      '2000000' → 2_000_000
    """
    import re
    s = size_str.strip().lower()
    m = re.fullmatch(r"([0-9]*\.?[0-9]+)\s*([kmg]?)", s)
    if not m:
        raise argparse.ArgumentTypeError(
            f"Invalid dataset size '{size_str}'. "
            f"Examples: 218k, 1.5m, 20m, 500000"
        )
    value, suffix = float(m.group(1)), m.group(2)
    multiplier = {"k": 1_000, "m": 1_000_000, "g": 1_000_000_000, "": 1}[suffix]
    result = int(value * multiplier)
    if result <= 0:
        raise argparse.ArgumentTypeError(f"Dataset size must be > 0, got '{size_str}'")
    return result


# -----------------------------------------------------------------------
# Args
# -----------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="XRD ViT Streaming Training")

    # Core
    parser.add_argument("--config",      required=True)
    parser.add_argument("--sweep_id",    required=True)
    parser.add_argument("--num_sweeps",  type=int, required=True)
    parser.add_argument("--distributed", action="store_true")

    # Dataset scale
    parser.add_argument("--dataset_size", type=parse_dataset_size, default="10m",
                        help="Train patterns per epoch (e.g. 10m, 20m). "
                             "Accepts: 1k 10k 100k 1m 5m 10m 20m 50m 100m or int.")
    parser.add_argument("--val_size",     type=parse_dataset_size, default="100k",
                        help="Val/test patterns per epoch (default: 100k)")

    # Generation parallelism
    parser.add_argument("--n_workers",      type=int, default=None,
                        help="Pool workers for pattern generation. "
                             "Default: cpu_count - 4")
    parser.add_argument("--pool_chunksize", type=int, default=16,
                        help="imap_unordered chunksize (default: 16)")

    # Checkpointing
    parser.add_argument("--checkpoint_dir", type=str,
                        default=os.path.join(
                            os.environ.get("SCRATCH", "."), "checkpoints"
                        ),
                        help="Directory for checkpoints (default: $SCRATCH/checkpoints)")
    parser.add_argument("--resume_checkpoint", type=str, default=None,
                        help="Path to .pt checkpoint to resume from.")

    return parser.parse_args()


# -----------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------

def load_config(path):
    with open(path) as f:
        return json.load(f)


def clean_summary_str(s):
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")


def compute_top_k_accuracy(output, target, k=1):
    with torch.no_grad():
        _, pred = output.topk(k, dim=1, largest=True, sorted=True)
        return pred.eq(target.view(-1, 1).expand_as(pred)).any(dim=1).float().sum().item()


def wait_for_master():
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = int(os.environ.get("MASTER_PORT", "29500"))
    rank        = int(os.environ.get("RANK", "0"))
    if rank == 0:
        print("[rank0] Master node.")
        return
    for attempt in range(10):
        try:
            with socket.create_connection((master_addr, master_port), timeout=3):
                print(f"[rank{rank}] TCP to {master_addr}:{master_port} ok")
                return
        except Exception as e:
            print(f"[rank{rank}] Waiting for master attempt {attempt+1}: {e}")
            time.sleep(2)
    print(f"[rank{rank}] Could not reach master.")
    exit(1)


def test_internode_connectivity():
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    device     = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    tensor     = torch.tensor([rank], device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print(f"[Rank {rank}] all_reduce={tensor.item()} "
          f"(expected {sum(range(world_size))})")
    dist.barrier()
    time.sleep(1)


def broadcast_config(config, rank):
    if not dist.is_available() or not dist.is_initialized():
        return config
    obj_list = [config] if rank == 0 else [None]
    dist.broadcast_object_list(obj_list, src=0)
    return obj_list[0]


# -----------------------------------------------------------------------
# Checkpoint helpers
# -----------------------------------------------------------------------

def save_checkpoint(checkpoint_dir, epoch, model, optimizer, scheduler,
                    scaler, wandb_run_id, metrics, prev_checkpoint_path) -> str:
    os.makedirs(checkpoint_dir, exist_ok=True)
    raw   = model.module if hasattr(model, "module") else model
    path  = os.path.join(checkpoint_dir, f"ckpt_epoch_{epoch:04d}.pt")

    torch.save({
        "epoch":                epoch,
        "model_state_dict":     raw.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict":    scaler.state_dict() if scaler else None,
        "metrics":              metrics,
        "wandb_run_id":         wandb_run_id,
    }, path)

    print(f"[ckpt] saved  → {path}  "
          f"epoch={epoch+1}  val_loss={metrics.get('val_loss', float('nan')):.4f}")

    if prev_checkpoint_path and os.path.isfile(prev_checkpoint_path):
        os.remove(prev_checkpoint_path)
        print(f"[ckpt] deleted → {prev_checkpoint_path}")

    return path


def load_checkpoint(path, model, optimizer, scheduler, scaler, device):
    print(f"[ckpt] loading ← {path}")
    ckpt     = torch.load(path, map_location=device)
    raw      = model.module if hasattr(model, "module") else model
    raw.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler and ckpt.get("scheduler_state_dict"):
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler and ckpt.get("scaler_state_dict"):
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    start_epoch  = ckpt["epoch"] + 1
    wandb_run_id = ckpt.get("wandb_run_id")
    metrics      = ckpt.get("metrics", {})
    print(f"[ckpt] restored — next epoch={start_epoch+1}  "
          f"val_loss={metrics.get('val_loss', float('nan')):.4f}")
    return start_epoch, wandb_run_id, metrics


# -----------------------------------------------------------------------
# Loader builder
# -----------------------------------------------------------------------

def build_loaders(config, args):
    p = config["parameters"]

    batch_size   = p["batch_size"]["value"]
    est_batches  = args.dataset_size // batch_size
    est_val_batches = args.val_size  // batch_size

    print(f"\n{'='*60}")
    print(f"  Streaming dataset  (RRUFF-weighted EG sampling)")
    print(f"  train     : {args.dataset_size:>12,}  (~{est_batches:,} batches)")
    print(f"  val/test  : {args.val_size:>12,}  (~{est_val_batches:,} batches)")
    print(f"  batch_size: {batch_size}")
    print(f"  n_workers : {args.n_workers or 'auto (cpu_count-4)'}")
    print(f"  chunksize : {args.pool_chunksize}")
    print(f"{'='*60}\n")

    shared = dict(
        batch_size     = batch_size,
        n_workers      = args.n_workers,
        pool_chunksize = args.pool_chunksize,
        num_classes    = NUM_EXTINCTION_GROUPS,
    )

    train_loader, num_classes, num_points = get_streaming_dataloader(
        num_samples=args.dataset_size, **shared
    )
    val_loader,  _, _ = get_streaming_dataloader(
        num_samples=args.val_size, **shared
    )
    test_loader, _, _ = get_streaming_dataloader(
        num_samples=args.val_size, **shared
    )

    return train_loader, val_loader, test_loader, num_classes, num_points


# -----------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------

def train(config, train_loader, val_loader, test_loader,
          device, rank, local_rank, distributed, args):

    # W&B — peek checkpoint for run ID before init
    resumed_run_id = None
    if args.resume_checkpoint and rank == 0:
        peek = torch.load(args.resume_checkpoint, map_location="cpu")
        resumed_run_id = peek.get("wandb_run_id")
        del peek

    if rank == 0:
        wandb.init(
            project = "project",
            config  = config,
            id      = resumed_run_id,
            resume  = "allow",
        )
        config = wandb.config

    # Model
    model = VIT_model(
        spec_length         = config["spec_length"],
        num_output          = config["num_classes"],
        patch_size          = config["patch_size"],
        embed_dim           = config["embed_dim"],
        depth               = config["depth"],
        num_heads           = config["num_heads"],
        mlp_ratio           = config["mlp_ratio"],
        drop_ratio          = config["dropout"],
        use_rope            = config["use_rope"],
        use_mlp_head        = config["use_mlp_head"],
        mlp_head_hidden_dim = config["mlp_head_hidden_dim"],
    ).to(device)

    if rank == 0:
        wandb.watch(model, log="all", log_freq=100)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            summary(model, input_size=(1, 1, model.spec_length))
        wandb.log({"model_summary": wandb.Html(
            f"<pre>{clean_summary_str(buf.getvalue())}</pre>"
        )})

    # Optimizer
    optimizer = (
        optim.Adam(model.parameters(),
                   lr=config["learning_rate"],
                   weight_decay=config["weight_decay"])
        if config["optimizer"] == "Adam"
        else optim.SGD(model.parameters(),
                       lr=config["learning_rate"],
                       weight_decay=config["weight_decay"])
    )

    # Scheduler
    scheduler = None
    if config["scheduler"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config["num_epochs"],
            eta_min=config.get("eta_min", 1e-6),
        )
    elif config["scheduler"] == "cosine_warm_restart":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.get("T_0", 10), T_mult=config.get("T_mult", 1),
            eta_min=config.get("eta_min", 1e-6),
        )
    elif config["scheduler"] == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.get("step_size", 10),
            gamma=config.get("decay_rate", 0.9),
        )

    # Mixed precision — fp16 + GradScaler (matches num_precision=16 in config)
    use_amp = (config.get("num_precision", 32) == 16) and device.type == "cuda"
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)
    if rank == 0:
        print(f"[train] mixed precision: {'fp16 + GradScaler' if use_amp else 'fp32'}")

    # Resume
    start_epoch    = 0
    prev_ckpt_path = None

    if args.resume_checkpoint:
        start_epoch, _, _ = load_checkpoint(
            args.resume_checkpoint, model, optimizer, scheduler, scaler, device
        )
        if distributed:
            t = torch.tensor(start_epoch, device=device)
            dist.broadcast(t, src=0)
            start_epoch = int(t.item())
        prev_ckpt_path = args.resume_checkpoint
        print(f"[rank{rank}] resuming at epoch {start_epoch+1}/{config['num_epochs']}")

    # DDP after loading weights
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    criterion = nn.CrossEntropyLoss()

    # ---- Training loop -------------------------------------------------
    for epoch in range(start_epoch, config["num_epochs"]):
        model.train()
        running_loss = num_batches = 0
        epoch_start  = time.time()

        loader_iter = (
            tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
            if rank == 0 else train_loader
        )

        for i, (inputs, targets) in enumerate(loader_iter):
            inputs  = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if inputs.ndim == 2:
                inputs = inputs.unsqueeze(1)   # (B,L) → (B,1,L)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = criterion(model(inputs), targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if scheduler is not None and config["scheduler"] == "cosine_warm_restart":
                scheduler.step(epoch + i / max(1, num_batches))

            running_loss += loss.item()
            num_batches  += 1

        avg_train_loss = running_loss / max(1, num_batches)
        epoch_secs     = time.time() - epoch_start
        throughput     = (num_batches * config.get("batch_size",
                          config["parameters"]["batch_size"]["value"]
                          if "parameters" in config else 256)) / max(1, epoch_secs)

        # Validation
        model.eval()
        val_loss = val_batches = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs  = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                if inputs.ndim == 2:
                    inputs = inputs.unsqueeze(1)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    val_loss += criterion(model(inputs), targets).item()
                val_batches += 1

        avg_val_loss = val_loss / max(1, val_batches)

        if scheduler is not None and config["scheduler"] != "cosine_warm_restart":
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        if rank == 0:
            print(f"Epoch {epoch+1}/{config['num_epochs']}  "
                  f"train={avg_train_loss:.4f}  val={avg_val_loss:.4f}  "
                  f"lr={current_lr:.2e}  batches={num_batches:,}  "
                  f"throughput={throughput:.0f} patterns/s")

            wandb.log({
                "epoch":            epoch + 1,
                "train_loss":       avg_train_loss,
                "val_loss":         avg_val_loss,
                "lr":               current_lr,
                "train_batches":    num_batches,
                "throughput_pps":   throughput,
                "scaler_scale":     scaler.get_scale() if use_amp else 1.0,
            })

            prev_ckpt_path = save_checkpoint(
                checkpoint_dir       = args.checkpoint_dir,
                epoch                = epoch,
                model                = model,
                optimizer            = optimizer,
                scheduler            = scheduler,
                scaler               = scaler,
                wandb_run_id         = wandb.run.id,
                metrics              = {"train_loss": avg_train_loss,
                                        "val_loss":   avg_val_loss},
                prev_checkpoint_path = prev_ckpt_path,
            )

        if distributed:
            dist.barrier()

    # ---- Test ----------------------------------------------------------
    if rank == 0:
        model_path = f"{config['model_path']}xrd_streaming_{wandb.run.id}.pth"
        raw_model  = model.module if hasattr(model, "module") else model
        torch.save(raw_model.state_dict(), model_path)
        wandb.save(model_path)

        model.eval()
        test_loss = total = correct = 0
        top1_correct = top3_correct = top5_correct = 0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs  = inputs.to(device)
                targets = targets.to(device)
                if inputs.ndim == 2:
                    inputs = inputs.unsqueeze(1)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    outputs    = model(inputs)
                    test_loss += criterion(outputs, targets).item()

                _, predicted = torch.max(outputs, 1)
                total   += targets.size(0)
                correct += (predicted == targets).sum().item()

                top1_correct += compute_top_k_accuracy(outputs, targets, k=1)
                top3_correct += compute_top_k_accuracy(outputs, targets, k=3)
                top5_correct += compute_top_k_accuracy(outputs, targets, k=5)

                all_preds.append(predicted.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        top1_acc = 100 * top1_correct / total
        top3_acc = 100 * top3_correct / total
        top5_acc = 100 * top5_correct / total

        print(f"\nTest  n={total:,}  "
              f"top1={top1_acc:.2f}%  top3={top3_acc:.2f}%  top5={top5_acc:.2f}%")

        wandb.log({
            "total_samples":      total,
            "test_loss":          test_loss,
            "test_accuracy_top1": top1_acc,
            "test_accuracy_top3": top3_acc,
            "test_accuracy_top5": top5_acc,
        })
        wandb.log({
            f"conf_mat_{wandb.run.name}": wandb.plot.confusion_matrix(
                probs=None,
                y_true=np.concatenate(all_targets),
                preds=np.concatenate(all_preds),
                class_names=[str(i) for i in range(config["num_classes"])],
            )
        })
        wandb.finish()


# -----------------------------------------------------------------------
# Sweep plumbing
# -----------------------------------------------------------------------

def run_sweep(sweep_id, config, train_loader, val_loader, test_loader,
              device, num_sweeps, rank, local_rank, distributed, args):
    if rank == 0:
        wandb.agent(
            sweep_id,
            function=lambda: train_entry_point(
                train_loader=train_loader, val_loader=val_loader,
                test_loader=test_loader,  device=device,
                rank=rank, local_rank=local_rank,
                distributed=distributed, args=args,
            ),
            count=num_sweeps, project="project",
        )
    else:
        for _ in range(num_sweeps):
            train(broadcast_config(None, rank),
                  train_loader, val_loader, test_loader,
                  device, rank, local_rank, distributed, args)

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


def train_entry_point(config=None, train_loader=None, val_loader=None,
                      test_loader=None, device=None, rank=None,
                      local_rank=None, distributed=None, args=None):
    wandb.init()
    if config is None:
        config = dict(wandb.config)
    broadcast_config(dict(config), rank)
    train(config, train_loader, val_loader, test_loader,
          device, rank, local_rank, distributed, args)


# -----------------------------------------------------------------------
# main
# -----------------------------------------------------------------------

def main():
    args   = parse_args()
    config = load_config(args.config)

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    print(f"local rank: {local_rank}")

    wait_for_master()

    if args.distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        device     = torch.device(f"cuda:{local_rank}")
        world_size = dist.get_world_size()
        rank       = dist.get_rank()
    else:
        device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        world_size = 1
        rank       = 0
        local_rank = 0

    if args.distributed:
        test_internode_connectivity()

    train_loader, val_loader, test_loader, num_classes, num_points = \
        build_loaders(config, args)

    if rank == 0:
        print(f"num_classes={num_classes}  num_points={num_points}")
        if args.resume_checkpoint:
            print(f"[main] Resuming: {args.resume_checkpoint}")
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        print(f"[main] Checkpoints → {args.checkpoint_dir}")

    run_sweep(
        args.sweep_id, config,
        train_loader, val_loader, test_loader,
        device, args.num_sweeps,
        rank, local_rank, args.distributed,
        args,
    )


if __name__ == "__main__":
    main()
