#!/usr/bin/env python3
import argparse
import importlib
import json
import math
import random
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


def add_repo_paths(script_path: Path) -> None:
    regular_cnn = script_path.parent
    repo_root = regular_cnn.parent.parent
    for path in (regular_cnn, repo_root / "Code" / "ViT_NVIDIA"):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


SCRIPT_PATH = Path(__file__).resolve()
add_repo_paths(SCRIPT_PATH)

from dataset import get_dataloaders_test  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Train a 1D bottleneck ResNet-50 on the realistic RRUFF-conditioned EG task.")
    parser.add_argument("--train-h5", required=True)
    parser.add_argument("--eval-h5-473", required=True)
    parser.add_argument("--eval-h5-325", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name", default="resnet50_rruff_eg")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-classes", type=int, default=99)
    parser.add_argument("--start-col", type=int, default=1)
    parser.add_argument("--end-col", type=int, default=8501)
    parser.add_argument("--save-every-epoch", action="store_true")
    parser.add_argument("--log-interval", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--resume-checkpoint", default="")
    parser.add_argument(
        "--external-model-dir",
        default="/home1/ALLOC2/UserC/ResNet50-Training",
        help="Directory containing resnet50.py from the student handoff.",
    )
    return parser.parse_args()


def load_resnet50_ctor(external_model_dir: str):
    model_dir = Path(external_model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"External model dir not found: {model_dir}")
    model_dir_str = str(model_dir)
    if model_dir_str not in sys.path:
        sys.path.insert(0, model_dir_str)
    module = importlib.import_module("resnet50")
    return module.ResNet50


def compute_topk(logits, targets, k):
    topk = torch.topk(logits, k=min(k, logits.shape[1]), dim=1).indices
    return topk.eq(targets.unsqueeze(1)).any(dim=1).sum().item()


def maybe_add_channel(inputs: torch.Tensor) -> torch.Tensor:
    if inputs.dim() == 2:
        return inputs.unsqueeze(1)
    return inputs


def iterate_h5_batches(h5_path, split, batch_size, start_col, end_col, shuffle, seed, max_batches=0):
    x_key = f"X_{split}"
    y_key = f"y_{split}"
    col_start = start_col - 1
    col_end = end_col
    with h5py.File(h5_path, "r") as handle:
        x_ds = handle[x_key]
        y_ds = handle[y_key]
        total_rows = int(x_ds.shape[0])
        num_batches = total_rows // batch_size
        batch_ids = list(range(num_batches))
        if shuffle:
            random.Random(seed).shuffle(batch_ids)
        emitted = 0
        for batch_id in batch_ids:
            start = batch_id * batch_size
            stop = start + batch_size
            inputs = np.asarray(x_ds[start:stop, col_start:col_end], dtype=np.float32)
            targets = np.asarray(y_ds[start:stop], dtype=np.int64) - 1
            yield torch.from_numpy(inputs), torch.from_numpy(targets)
            emitted += 1
            if max_batches and emitted >= max_batches:
                break


@torch.no_grad()
def evaluate_h5_split(model, h5_path, split, batch_size, start_col, end_col, device, criterion, max_batches=0):
    model.eval()
    loss_sum = 0.0
    total = 0
    top1 = top3 = top5 = 0
    batches = 0
    for inputs, targets in iterate_h5_batches(
        h5_path=h5_path,
        split=split,
        batch_size=batch_size,
        start_col=start_col,
        end_col=end_col,
        shuffle=False,
        seed=0,
        max_batches=max_batches,
    ):
        inputs = maybe_add_channel(inputs).to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        bsz = targets.shape[0]
        total += bsz
        loss_sum += loss.item() * bsz
        top1 += compute_topk(logits, targets, 1)
        top3 += compute_topk(logits, targets, 3)
        top5 += compute_topk(logits, targets, 5)
        batches += 1
    return {
        "loss": loss_sum / max(total, 1),
        "top1": 100.0 * top1 / max(total, 1),
        "top3": 100.0 * top3 / max(total, 1),
        "top5": 100.0 * top5 / max(total, 1),
        "samples": total,
        "batches": batches,
    }


@torch.no_grad()
def evaluate_loader(model, loader, device, criterion):
    model.eval()
    loss_sum = 0.0
    total = 0
    top1 = top3 = top5 = 0
    batches = 0
    for inputs, targets in loader:
        inputs = maybe_add_channel(inputs).to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        bsz = targets.shape[0]
        total += bsz
        loss_sum += loss.item() * bsz
        top1 += compute_topk(logits, targets, 1)
        top3 += compute_topk(logits, targets, 3)
        top5 += compute_topk(logits, targets, 5)
        batches += 1
    return {
        "loss": loss_sum / max(total, 1),
        "top1": 100.0 * top1 / max(total, 1),
        "top3": 100.0 * top3 / max(total, 1),
        "top5": 100.0 * top5 / max(total, 1),
        "samples": total,
        "batches": batches,
    }


def make_scheduler(optimizer, epochs: int, warmup_epochs: int, min_lr_ratio: float):
    if epochs <= 1:
        return None
    warmup_epochs = max(0, min(warmup_epochs, epochs - 1))
    cosine_epochs = max(1, epochs - warmup_epochs)
    schedulers = []
    milestones = []
    if warmup_epochs > 0:
        schedulers.append(LinearLR(optimizer, start_factor=0.2, end_factor=1.0, total_iters=warmup_epochs))
        milestones.append(warmup_epochs)
    schedulers.append(CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=optimizer.param_groups[0]["lr"] * min_lr_ratio))
    if len(schedulers) == 1:
        return schedulers[0]
    return SequentialLR(optimizer, schedulers=schedulers, milestones=milestones)


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output_dir) / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "train_h5": args.train_h5,
        "eval_h5_473": args.eval_h5_473,
        "eval_h5_325": args.eval_h5_325,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "prefetch_factor": args.prefetch_factor,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "warmup_epochs": args.warmup_epochs,
        "min_learning_rate": args.min_learning_rate,
        "num_classes": args.num_classes,
        "start_col": args.start_col,
        "end_col": args.end_col,
        "device": args.device,
        "external_model_dir": args.external_model_dir,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))

    device = torch.device(args.device)
    ResNet50 = load_resnet50_ctor(args.external_model_dir)
    model = ResNet50(in_channels=1, classes=args.num_classes).to(device)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scheduler = make_scheduler(
        optimizer=optimizer,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        min_lr_ratio=max(args.min_learning_rate / args.learning_rate, 0.0),
    )

    history = []
    best_val = math.inf
    best_val_epoch = None
    best_325_top1 = -1.0
    best_325_epoch = None
    total_steps = 0
    start_epoch = 1

    with h5py.File(args.train_h5, "r") as handle:
        train_size = int(handle["X_train"].shape[0])
    steps_per_epoch = train_size // args.batch_size

    if args.resume_checkpoint:
        ckpt = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt and scheduler is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        history = ckpt.get("history", [])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        if history:
            total_steps = int(history[-1].get("total_steps", 0))
            best_val_record = min(history, key=lambda r: r["val"]["loss"])
            best_val = float(best_val_record["val"]["loss"])
            best_val_epoch = int(best_val_record["epoch"])
            best_325_record = max(history, key=lambda r: r.get("rruff325", {}).get("top1", float("-inf")))
            best_325_top1 = float(best_325_record.get("rruff325", {}).get("top1", float("-inf")))
            best_325_epoch = int(best_325_record["epoch"])

    _, _, loader_473, _, _ = get_dataloaders_test(
        args.eval_h5_473,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_classes=args.num_classes,
        prefetch_factor=args.prefetch_factor,
        start_col=args.start_col,
        end_col=args.end_col,
        label_mode="categorical",
    )
    _, _, loader_325, _, _ = get_dataloaders_test(
        args.eval_h5_325,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_classes=args.num_classes,
        prefetch_factor=args.prefetch_factor,
        start_col=args.start_col,
        end_col=args.end_col,
        label_mode="categorical",
    )

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_samples = 0
        epoch_top1 = 0
        epoch_top3 = 0
        epoch_top5 = 0
        start_time = time.time()
        interval_start = time.time()
        interval_samples = 0
        interval_steps = 0

        train_iter = iterate_h5_batches(
            h5_path=args.train_h5,
            split="train",
            batch_size=args.batch_size,
            start_col=args.start_col,
            end_col=args.end_col,
            shuffle=True,
            seed=args.seed + epoch,
            max_batches=max(args.max_train_steps - total_steps, 0) if args.max_train_steps else 0,
        )

        for inputs, targets in train_iter:
            inputs = maybe_add_channel(inputs).to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            bsz = targets.shape[0]
            epoch_loss += loss.item() * bsz
            epoch_samples += bsz
            epoch_top1 += compute_topk(logits, targets, 1)
            epoch_top3 += compute_topk(logits, targets, 3)
            epoch_top5 += compute_topk(logits, targets, 5)
            total_steps += 1
            interval_samples += bsz
            interval_steps += 1

            if args.log_interval and interval_steps >= args.log_interval:
                interval_seconds = time.time() - interval_start
                print(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "step": total_steps,
                            "interval_steps": interval_steps,
                            "interval_samples": interval_samples,
                            "interval_seconds": interval_seconds,
                            "interval_samples_per_second": interval_samples / max(interval_seconds, 1e-9),
                            "learning_rate": optimizer.param_groups[0]["lr"],
                        }
                    ),
                    flush=True,
                )
                interval_start = time.time()
                interval_samples = 0
                interval_steps = 0

            if args.max_train_steps and total_steps >= args.max_train_steps:
                break

        train_seconds = time.time() - start_time
        train_metrics = {
            "loss": epoch_loss / max(epoch_samples, 1),
            "top1": 100.0 * epoch_top1 / max(epoch_samples, 1),
            "top3": 100.0 * epoch_top3 / max(epoch_samples, 1),
            "top5": 100.0 * epoch_top5 / max(epoch_samples, 1),
            "samples": epoch_samples,
            "seconds": train_seconds,
            "samples_per_second": epoch_samples / max(train_seconds, 1e-9),
            "learning_rate": optimizer.param_groups[0]["lr"],
        }

        val_metrics = evaluate_h5_split(
            model=model,
            h5_path=args.train_h5,
            split="val",
            batch_size=args.batch_size,
            start_col=args.start_col,
            end_col=args.end_col,
            device=device,
            criterion=criterion,
            max_batches=args.max_val_batches,
        )
        metrics_473 = evaluate_loader(model, loader_473, device, criterion)
        metrics_325 = evaluate_loader(model, loader_325, device, criterion)

        record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "rruff473": metrics_473,
            "rruff325": metrics_325,
            "total_steps": total_steps,
            "steps_per_epoch": steps_per_epoch,
            "train_size": train_size,
        }
        history.append(record)
        print(json.dumps(record), flush=True)

        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "history": history,
            "config": config,
        }
        torch.save(state, output_dir / "checkpoint_latest.pt")

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_val_epoch = epoch
            torch.save(state, output_dir / "checkpoint_best_val.pt")

        if metrics_325["top1"] > best_325_top1:
            best_325_top1 = metrics_325["top1"]
            best_325_epoch = epoch
            torch.save(state, output_dir / "checkpoint_best_rruff325.pt")

        if args.save_every_epoch:
            torch.save(state, output_dir / f"checkpoint_epoch_{epoch:02d}.pt")

        if scheduler is not None:
            scheduler.step()

        if args.max_train_steps and total_steps >= args.max_train_steps:
            break

    summary = {
        "best_val_epoch": best_val_epoch,
        "best_val_loss": best_val,
        "best_rruff325_epoch": best_325_epoch,
        "best_rruff325_top1": best_325_top1,
        "history": history,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
