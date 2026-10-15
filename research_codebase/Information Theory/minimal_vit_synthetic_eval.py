#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
VIT_DIR = REPO_ROOT / "Code" / "ViT_NVIDIA"
import sys
if str(VIT_DIR) not in sys.path:
    sys.path.insert(0, str(VIT_DIR))

from model import VIT_model, adapt_patch_embed_input_channels  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Minimal inference-only ViT synthetic evaluator.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--eval-h5", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--output-npz", default="")
    p.add_argument("--subset-json", default="")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--aux-temperature", type=float, default=5.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--label-indexing",
        choices=("auto", "zero_based", "one_based_raw"),
        default="auto",
        help=(
            "How to interpret y_test labels in the eval HDF5. "
            "'zero_based' uses labels as-is, "
            "'one_based_raw' keeps 1..99 raw EG ids as model targets, "
            "'auto' preserves historical behavior and shifts labels down by 1 "
            "when the minimum label is >= 1."
        ),
    )
    return p.parse_args()


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
    adapted_state = adapt_patch_embed_input_channels(get_model_state_dict(checkpoint), model)
    model.load_state_dict(adapted_state, strict=False)
    return model.to(device).eval()


def topk_accuracy(probs: np.ndarray, targets: np.ndarray, k: int) -> float:
    k = min(k, probs.shape[1])
    topk = np.argpartition(-probs, kth=k - 1, axis=1)[:, :k]
    return float((topk == targets[:, None]).any(axis=1).mean())


def load_case_ids(handle: h5py.File, n_rows: int) -> list[str]:
    if "case_ids" not in handle:
        return [f"row_{i}" for i in range(n_rows)]
    raw = handle["case_ids"][:]
    return [x.decode() if isinstance(x, bytes) else str(x) for x in raw]


def maybe_filter_subset(x, y, case_ids, subset_json: str):
    if not subset_json:
        return x, y, case_ids
    subset = json.loads(Path(subset_json).read_text())
    allowed = {str(row["case_id"]) for row in subset["rows"]}
    keep_idx = [i for i, cid in enumerate(case_ids) if cid in allowed]
    x = x[keep_idx]
    y = y[keep_idx]
    case_ids = [case_ids[i] for i in keep_idx]
    return x, y, case_ids


def main():
    args = parse_args()
    config = json.loads(Path(args.config).read_text())
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    device = torch.device(args.device)
    model = build_model(config, checkpoint, device)

    with h5py.File(args.eval_h5, "r") as h5:
        x = np.asarray(h5["X_test"][:], dtype=np.float32)
        y = np.asarray(h5["y_test"][:], dtype=np.int64)
        case_ids = load_case_ids(h5, x.shape[0])
    x, y, case_ids = maybe_filter_subset(x, y, case_ids, args.subset_json)
    if args.label_indexing == "auto" and y.size and y.min() >= 1:
        y = y - 1
    elif args.label_indexing == "zero_based":
        pass
    elif args.label_indexing == "one_based_raw":
        pass

    probs = []
    with torch.no_grad():
        for start in range(0, len(x), args.batch_size):
            batch = torch.from_numpy(x[start:start + args.batch_size]).to(device)
            cls_embedding = model(batch, return_cls_embedding=True)
            aux_logits = model.aux_ext_head(cls_embedding) / args.aux_temperature
            probs.append(torch.softmax(aux_logits, dim=1).cpu().numpy())
    probs = np.concatenate(probs, axis=0)

    summary = {
        "checkpoint": args.checkpoint,
        "config": args.config,
        "eval_h5": args.eval_h5,
        "subset_json": args.subset_json,
        "n": int(len(y)),
        "top1": topk_accuracy(probs, y, 1),
        "top5": topk_accuracy(probs, y, 5),
        "mean_max_prob": float(probs.max(axis=1).mean()),
        "aux_temperature": float(args.aux_temperature),
        "mode": "minimal_aux_temperature_only",
        "label_indexing": args.label_indexing,
        "sample_case_ids": case_ids[:10],
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    if args.output_npz:
        np.savez_compressed(
            args.output_npz,
            calibrated_probs=probs.astype(np.float32),
            targets=y.astype(np.int64),
            case_ids=np.asarray(case_ids, dtype=object),
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
