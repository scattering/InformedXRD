#!/usr/bin/env python3
import argparse
import h5py
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau


def add_repo_paths(script_path: Path) -> None:
    regular_cnn = script_path.parent
    repo_root = regular_cnn.parent.parent
    vit_dir = repo_root / "Code" / "ViT_NVIDIA"
    for path in (regular_cnn, vit_dir):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


SCRIPT_PATH = Path(__file__).resolve()
add_repo_paths(SCRIPT_PATH)

from resnet_model import ResnetClassifier  # noqa: E402
def parse_args():
    parser = argparse.ArgumentParser(description="Train a 1D ResNet on the realistic RRUFF-conditioned EG task.")
    parser.add_argument("--train-h5", required=True)
    parser.add_argument("--eval-h5-473", required=True)
    parser.add_argument("--eval-h5-325", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name", default="resnet_rruff_eg")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-classes", type=int, default=99)
    parser.add_argument("--start-col", type=int, default=1)
    parser.add_argument("--end-col", type=int, default=8501)
    parser.add_argument("--res-dims", default="32,64,64,64")
    parser.add_argument("--res-kernel", default="5,7,17,13")
    parser.add_argument("--res-stride", default="4,4,5,3")
    parser.add_argument("--num-blocks", default="2,2,2,2")
    parser.add_argument("--first-kernel-size", type=int, default=13)
    parser.add_argument("--first-stride", type=int, default=1)
    parser.add_argument("--first-pool-kernel-size", type=int, default=7)
    parser.add_argument("--first-pool-stride", type=int, default=7)
    parser.add_argument("--save-every-epoch", action="store_true")
    parser.add_argument("--log-interval", type=int, default=0)
    parser.add_argument("--io-mode", choices=["dataloader", "contiguous_h5"], default="contiguous_h5")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--resume-checkpoint", default="")
    parser.add_argument("--lr-factor", type=float, default=0.3)
    parser.add_argument("--lr-patience", type=int, default=2)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-3)
    parser.add_argument("--minimum-epochs", type=int, default=18)
    return parser.parse_args()


def parse_int_list(raw: str):
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def compute_topk(logits, targets, k):
    topk = torch.topk(logits, k=min(k, logits.shape[1]), dim=1).indices
    return topk.eq(targets.unsqueeze(1)).any(dim=1).sum().item()


def iterate_h5_batches(h5_path, split, batch_size, start_col, end_col, shuffle, seed, max_batches=0):
    x_key = f"X_{split}"
    y_key = f"y_{split}"
    col_start = start_col - 1
    col_end = end_col
    with h5py.File(h5_path, "r") as handle:
        x_ds = handle[x_key]
        y_ds = handle[y_key]
        total_rows = int(x_ds.shape[0])
        num_batches = math.ceil(total_rows / batch_size)
        batch_ids = list(range(num_batches))
        if shuffle:
            random.Random(seed).shuffle(batch_ids)
        emitted = 0
        for batch_id in batch_ids:
            start = batch_id * batch_size
            stop = min(start + batch_size, total_rows)
            inputs = np.asarray(x_ds[start:stop, col_start:col_end], dtype=np.float32)
            targets = np.asarray(y_ds[start:stop], dtype=np.int64) - 1
            yield torch.from_numpy(inputs), torch.from_numpy(targets)
            emitted += 1
            if max_batches and emitted >= max_batches:
                break


def evaluate_h5_split(model, h5_path, split, batch_size, start_col, end_col, device, criterion, max_batches=0):
    model.eval()
    loss_sum = 0.0
    total = 0
    top1 = top3 = top5 = 0
    batches = 0
    with torch.no_grad():
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
            inputs = inputs.to(device, non_blocking=True)
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
def evaluate_loader(model, loader, device, criterion, max_batches=0):
    model.eval()
    loss_sum = 0.0
    total = 0
    top1 = top3 = top5 = 0
    batches = 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
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
        if max_batches and batches >= max_batches:
            break
    return {
        "loss": loss_sum / max(total, 1),
        "top1": 100.0 * top1 / max(total, 1),
        "top3": 100.0 * top3 / max(total, 1),
        "top5": 100.0 * top5 / max(total, 1),
        "samples": total,
        "batches": batches,
    }


def main():
    args = parse_args()
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
        "num_classes": args.num_classes,
        "start_col": args.start_col,
        "end_col": args.end_col,
        "res_dims": parse_int_list(args.res_dims),
        "res_kernel": parse_int_list(args.res_kernel),
        "res_stride": parse_int_list(args.res_stride),
        "num_blocks": parse_int_list(args.num_blocks),
        "first_kernel_size": args.first_kernel_size,
        "first_stride": args.first_stride,
        "first_pool_kernel_size": args.first_pool_kernel_size,
        "first_pool_stride": args.first_pool_stride,
        "lr_factor": args.lr_factor,
        "lr_patience": args.lr_patience,
        "min_lr": args.min_lr,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_min_delta": args.early_stopping_min_delta,
        "minimum_epochs": args.minimum_epochs,
        "device": args.device,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))

    device = torch.device(args.device)
    intensity_points = args.end_col - args.start_col + 1
    if args.io_mode != "contiguous_h5":
        raise ValueError("This convergence run requires --io-mode contiguous_h5")

    model = ResnetClassifier(
        input_dim=1,
        res_dims=parse_int_list(args.res_dims),
        res_kernel=parse_int_list(args.res_kernel),
        res_stride=parse_int_list(args.res_stride),
        num_blocks=parse_int_list(args.num_blocks),
        first_kernel_size=args.first_kernel_size,
        first_stride=args.first_stride,
        first_pool_kernel_size=args.first_pool_kernel_size,
        first_pool_stride=args.first_pool_stride,
        num_classes=args.num_classes,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
        min_lr=args.min_lr,
    )
    criterion = nn.CrossEntropyLoss()

    history = []
    best_val = math.inf
    best_epoch = None
    early_stop_reference = math.inf
    epochs_without_improvement = 0
    total_steps = 0
    start_epoch = 1
    with h5py.File(args.train_h5, "r") as handle:
        train_size = int(handle["X_train"].shape[0])
    steps_per_epoch = math.ceil(train_size / args.batch_size)

    if args.resume_checkpoint:
        ckpt = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        history = ckpt.get("history", [])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        if history:
            total_steps = int(history[-1].get("total_steps", 0))
            best_record = min(history, key=lambda r: r["val"]["loss"])
            best_val = float(best_record["val"]["loss"])
            best_epoch = int(best_record["epoch"])
            early_stop_reference = float(ckpt.get("early_stop_reference", best_val))
            epochs_without_improvement = int(ckpt.get("epochs_without_improvement", 0))
        print(
            json.dumps(
                {
                    "resume_checkpoint": args.resume_checkpoint,
                    "start_epoch": start_epoch,
                    "best_val_loss": best_val,
                    "best_epoch": best_epoch,
                    "total_steps": total_steps,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "epochs_without_improvement": epochs_without_improvement,
                }
            ),
            flush=True,
        )

    def checkpoint_state(epoch):
        return {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "history": history,
            "config": config,
            "best_val_loss": best_val,
            "best_epoch": best_epoch,
            "early_stop_reference": early_stop_reference,
            "epochs_without_improvement": epochs_without_improvement,
        }

    best_path = output_dir / "checkpoint_best.pt"
    if args.resume_checkpoint and best_epoch == start_epoch - 1 and not best_path.exists():
        torch.save(checkpoint_state(start_epoch - 1), best_path)

    converged = (
        start_epoch - 1 >= args.minimum_epochs
        and epochs_without_improvement >= args.early_stopping_patience
    )
    for epoch in range(start_epoch, args.epochs + 1):
        if converged:
            break
        model.train()
        epoch_learning_rate = optimizer.param_groups[0]["lr"]
        epoch_loss = 0.0
        epoch_samples = 0
        epoch_top1 = 0
        epoch_top3 = 0
        epoch_top5 = 0
        start_time = time.time()

        interval_start = time.time()
        interval_samples = 0
        interval_steps = 0
        remaining_steps = 0
        if args.max_train_steps:
            remaining_steps = max(args.max_train_steps - total_steps, 0)
        train_iter = iterate_h5_batches(
            h5_path=args.train_h5,
            split="train",
            batch_size=args.batch_size,
            start_col=args.start_col,
            end_col=args.end_col,
            shuffle=True,
            seed=args.seed + epoch,
            max_batches=remaining_steps,
        )

        for inputs, targets in train_iter:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
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

        val_loss = val_metrics["loss"]
        if val_loss < early_stop_reference - args.early_stopping_min_delta:
            early_stop_reference = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        scheduler.step(val_loss)

        record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "total_steps": total_steps,
            "steps_per_epoch": steps_per_epoch,
            "train_size": train_size,
            "intensity_points": intensity_points,
            "learning_rate": epoch_learning_rate,
            "next_learning_rate": optimizer.param_groups[0]["lr"],
            "epochs_without_improvement": epochs_without_improvement,
        }
        history.append(record)
        print(json.dumps(record), flush=True)

        latest_path = output_dir / "checkpoint_latest.pt"
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            torch.save(checkpoint_state(epoch), best_path)

        torch.save(checkpoint_state(epoch), latest_path)

        if args.save_every_epoch:
            torch.save(checkpoint_state(epoch), output_dir / f"checkpoint_epoch_{epoch:02d}.pt")

        if args.max_train_steps and total_steps >= args.max_train_steps:
            break
        converged = (
            epoch >= args.minimum_epochs
            and epochs_without_improvement >= args.early_stopping_patience
        )

    selected_checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(selected_checkpoint["model"])
    selected_epoch = int(selected_checkpoint["epoch"])
    benchmark_metrics = {}
    for name, eval_path in (("rruff473", args.eval_h5_473), ("rruff325", args.eval_h5_325)):
        benchmark_metrics[name] = evaluate_h5_split(
            model=model,
            h5_path=eval_path,
            split="test",
            batch_size=args.batch_size,
            start_col=args.start_col,
            end_col=args.end_col,
            device=device,
            criterion=criterion,
            max_batches=0,
        )
        print(
            json.dumps(
                {"benchmark": name, "selected_epoch": selected_epoch, **benchmark_metrics[name]}
            ),
            flush=True,
        )

    summary = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "selected_epoch": selected_epoch,
        "converged": converged,
        "epochs_without_improvement": epochs_without_improvement,
        "final_learning_rate": optimizer.param_groups[0]["lr"],
        "history": history,
        "benchmarks": benchmark_metrics,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
