#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
CODE_DIR = Path(os.environ.get("AI_DIFFRACTION_CNN_CODE_DIR", str(THIS_DIR)))
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from resnet_model import ResnetClassifier


def topk_acc(logits: np.ndarray, y: np.ndarray, k: int) -> float:
    order = np.argpartition(-logits, kth=k - 1, axis=1)[:, :k]
    return float(np.mean([yy in row for yy, row in zip(y, order)]))


def eval_logits(logits: np.ndarray, y: np.ndarray) -> dict:
    return {"top1": topk_acc(logits, y, 1), "top5": topk_acc(logits, y, 5)}


def load_prior(path: Path, num_classes: int) -> np.ndarray:
    prior_data = json.loads(path.read_text())
    if isinstance(prior_data, dict) and "ext_group_priors" in prior_data:
        prior_data = prior_data["ext_group_priors"]
    elif isinstance(prior_data, dict) and "rows" in prior_data:
        rows = prior_data["rows"]
        prior = np.zeros(num_classes, dtype=np.float64)
        for i, row in enumerate(rows[:num_classes]):
            if isinstance(row, dict):
                if "probability" in row:
                    prior[i] = float(row["probability"])
                elif "prior" in row:
                    prior[i] = float(row["prior"])
                elif "count" in row:
                    prior[i] = float(row["count"])
                else:
                    raise RuntimeError("Unsupported row format in prior file")
            else:
                prior[i] = float(row)
        prior = np.clip(prior, 1e-12, None)
        return prior / prior.sum()

    prior = np.zeros(num_classes, dtype=np.float64)
    if isinstance(prior_data, dict):
        for k, v in prior_data.items():
            kk = int(k)
            if 0 <= kk < num_classes:
                prior[kk] = float(v)
            elif 1 <= kk <= num_classes:
                prior[kk - 1] = float(v)
    elif isinstance(prior_data, list) and len(prior_data) == num_classes:
        prior = np.asarray(prior_data, dtype=np.float64)
    else:
        raise RuntimeError("Unsupported prior format")

    prior = np.clip(prior, 1e-12, None)
    return prior / prior.sum()


def build_model(cfg: dict) -> ResnetClassifier:
    return ResnetClassifier(
        input_dim=1,
        res_dims=cfg["res_dims"],
        res_kernel=cfg["res_kernel"],
        res_stride=cfg["res_stride"],
        num_blocks=cfg["num_blocks"],
        first_kernel_size=cfg["first_kernel_size"],
        first_stride=cfg["first_stride"],
        first_pool_kernel_size=cfg["first_pool_kernel_size"],
        first_pool_stride=cfg["first_pool_stride"],
        num_classes=cfg["num_classes"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--eval-h5", required=True)
    parser.add_argument("--prior-json", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    with h5py.File(args.eval_h5, "r") as h5:
        x = np.asarray(h5["X_test"], dtype=np.float32)
        y = np.asarray(h5["y_test"], dtype=np.int64) - 1

    start_col = int(cfg.get("start_col", 0))
    end_col = int(cfg.get("end_col", x.shape[1]))
    x = x[:, start_col:end_col]

    model = build_model(cfg)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    model.eval()

    with torch.no_grad():
        logits = model(torch.from_numpy(x)).numpy()

    log_prior = np.log(load_prior(Path(args.prior_json), cfg["num_classes"]))

    out = {
        "checkpoint": args.checkpoint,
        "config": cfg,
        "input_shape": list(x.shape),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "raw": eval_logits(logits, y),
        "temp_only": [],
        "temp_plus_prior": [],
    }
    for T in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0]:
        out["temp_only"].append({"T": T, **eval_logits(logits / T, y)})
        out["temp_plus_prior"].append({"T": T, **eval_logits(logits / T + log_prior[None, :], y)})

    Path(args.output_json).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
