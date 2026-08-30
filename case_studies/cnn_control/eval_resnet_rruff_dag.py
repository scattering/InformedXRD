#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
VIT_DIR = REPO_ROOT / "Code" / "ViT_NVIDIA"
for path in (SCRIPT_PATH.parent, VIT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analyze_topological_error_distance import build_graph, summarize_model  # noqa: E402
from resnet_model import ResnetClassifier  # noqa: E402
from resnet_rruff_train import iterate_h5_batches  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a 1D ResNet on RRUFF and summarize EG DAG errors."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-label", required=True)
    parser.add_argument("--eval-h5", required=True)
    parser.add_argument("--graph-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def build_model(config):
    return ResnetClassifier(
        input_dim=1,
        res_dims=config["res_dims"],
        res_kernel=config["res_kernel"],
        res_stride=config["res_stride"],
        num_blocks=config["num_blocks"],
        first_kernel_size=config["first_kernel_size"],
        first_stride=config["first_stride"],
        first_pool_kernel_size=config["first_pool_kernel_size"],
        first_pool_stride=config["first_pool_stride"],
        num_classes=config["num_classes"],
    )


def main():
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if not isinstance(checkpoint, dict) or "config" not in checkpoint:
        raise ValueError("Expected a convergence checkpoint containing its training config")

    config = checkpoint["config"]
    model = build_model(config)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise ValueError(
            f"Checkpoint/model mismatch: missing={list(missing)}, "
            f"unexpected={list(unexpected)}"
        )
    model = model.to(device).eval()

    examples = []
    top1_correct = 0
    top3_correct = 0
    top5_correct = 0
    with torch.no_grad():
        sample_index = 0
        for inputs, targets in iterate_h5_batches(
            h5_path=args.eval_h5,
            split="test",
            batch_size=args.batch_size,
            start_col=int(config["start_col"]),
            end_col=int(config["end_col"]),
            shuffle=False,
            seed=0,
        ):
            logits = model(inputs.to(device))
            top5 = torch.topk(logits, k=5, dim=1).indices.cpu().numpy()
            targets_np = targets.numpy()
            top1_correct += int(np.sum(top5[:, 0] == targets_np))
            top3_correct += int(
                np.sum([target in row for target, row in zip(targets_np, top5[:, :3])])
            )
            top5_correct += int(
                np.sum([target in row for target, row in zip(targets_np, top5)])
            )
            for row, target in enumerate(targets_np.tolist()):
                prediction = int(top5[row, 0])
                examples.append(
                    {
                        "index": sample_index + row,
                        "true_eg": int(target) + 1,
                        "pred_eg": prediction + 1,
                        "top5_eg": [int(value) + 1 for value in top5[row].tolist()],
                        "correct": prediction == int(target),
                    }
                )
            sample_index += targets_np.shape[0]

    n_samples = len(examples)
    model_blob = {
        "accuracy_top1_pct": 100.0 * top1_correct / n_samples,
        "examples": examples,
    }
    _, directed, undirected, label_to_node = build_graph(args.graph_json)
    topology = summarize_model(model_blob, directed, undirected, label_to_node)

    output = {
        "checkpoint": args.checkpoint,
        "checkpoint_label": args.checkpoint_label,
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "best_epoch_recorded": int(checkpoint.get("best_epoch", -1)),
        "eval_h5": args.eval_h5,
        "n_samples": n_samples,
        "flat_metrics_pct": {
            "top1": 100.0 * top1_correct / n_samples,
            "top3": 100.0 * top3_correct / n_samples,
            "top5": 100.0 * top5_correct / n_samples,
        },
        "topology": topology,
        "examples": examples,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))
    print(json.dumps({key: value for key, value in output.items() if key != "examples"}, indent=2))


if __name__ == "__main__":
    main()
