#!/usr/bin/env python3
import argparse
import ast
import csv
import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Dump per-case ML4pXRDs SG->EG predictions in compare_325_failure_modes format."
    )
    parser.add_argument("--model-dir", required=True, help="Path to the SavedModel directory.")
    parser.add_argument("--eval-h5", required=True, help="RRUFF benchmark HDF5.")
    parser.add_argument("--mapping-csv", required=True, help="canonical_extinction_to_space_group.csv.")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--batch-size", type=int, default=145)
    parser.add_argument("--model-name", default="ml4pxrds")
    parser.add_argument(
        "--sqrt",
        choices=("on", "off"),
        default="on",
        help="Apply sqrt intensity compression before inference.",
    )
    return parser.parse_args()


def load_sg_to_eg_map(mapping_csv: str) -> np.ndarray:
    sg_to_eg = np.full(231, -1, dtype=np.int64)
    with open(mapping_csv, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            eg_index = int(row["Index"])
            sg_list = ast.literal_eval(row["Space Group Numbers"])
            for sg in sg_list:
                sg_to_eg[int(sg)] = eg_index
    missing = np.where(sg_to_eg[1:] < 0)[0]
    if len(missing):
        raise ValueError(f"Missing SG->EG mapping for SGs: {(missing + 1).tolist()}")
    return sg_to_eg


def collapse_sg_probs_to_eg(sg_probs: np.ndarray, sg_to_eg: np.ndarray) -> np.ndarray:
    eg_count = int(sg_to_eg.max()) + 1
    eg_probs = np.zeros((sg_probs.shape[0], eg_count), dtype=np.float32)
    for sg in range(1, 231):
        eg_probs[:, sg_to_eg[sg]] += sg_probs[:, sg - 1]
    return eg_probs


def load_savedmodel_predictor(model_dir: str):
    import tensorflow as tf

    loaded = tf.saved_model.load(model_dir)
    if "serving_default" in loaded.signatures:
        infer = loaded.signatures["serving_default"]

        def _predict(batch: np.ndarray) -> np.ndarray:
            outputs = infer(tf.convert_to_tensor(batch, dtype=tf.float32))
            if isinstance(outputs, dict):
                if len(outputs) != 1:
                    raise ValueError(f"Expected single SavedModel output, found keys: {list(outputs.keys())}")
                tensor = next(iter(outputs.values()))
            else:
                tensor = outputs
            return tf.convert_to_tensor(tensor).numpy()

        return _predict

    if callable(loaded):

        def _predict(batch: np.ndarray) -> np.ndarray:
            outputs = loaded(tf.convert_to_tensor(batch, dtype=tf.float32))
            if isinstance(outputs, dict):
                if len(outputs) != 1:
                    raise ValueError(f"Expected single SavedModel output, found keys: {list(outputs.keys())}")
                tensor = next(iter(outputs.values()))
            else:
                tensor = outputs
            return tf.convert_to_tensor(tensor).numpy()

        return _predict

    raise RuntimeError(f"Could not find a callable inference endpoint in SavedModel: {model_dir}")


def load_case_ids(handle: h5py.File, n_rows: int) -> list[str]:
    if "case_ids" not in handle:
        return [f"row_{i}" for i in range(n_rows)]
    raw = handle["case_ids"][:]
    return [x.decode() if isinstance(x, bytes) else str(x) for x in raw]


def summarize_counter(counter, topn=10):
    return [{"eg": int(k), "count": int(v)} for k, v in counter.most_common(topn)]


def main():
    args = parse_args()
    import tensorflow as tf

    sg_to_eg = load_sg_to_eg_map(args.mapping_csv)

    with h5py.File(args.eval_h5, "r") as handle:
        x = np.asarray(handle["X_test"], dtype=np.float32)
        y = np.asarray(handle["y_test"], dtype=np.int64)
        case_ids = load_case_ids(handle, x.shape[0])

    if y.size and y.min() >= 1 and y.max() == (int(sg_to_eg.max()) + 1):
        y = y - 1

    if args.sqrt == "on":
        x = np.sqrt(np.clip(x, a_min=0.0, a_max=None))
    else:
        x = np.clip(x, a_min=0.0, a_max=None)
    x = np.expand_dims(x, axis=-1)

    predict_fn = load_savedmodel_predictor(args.model_dir)
    chunks = []
    for start in range(0, x.shape[0], args.batch_size):
        chunks.append(predict_fn(x[start : start + args.batch_size]))
    sg_logits = np.concatenate(chunks, axis=0)
    sg_probs = tf.nn.softmax(sg_logits, axis=1).numpy().astype(np.float32)
    eg_probs = collapse_sg_probs_to_eg(sg_probs, sg_to_eg)

    top5_eg = np.argpartition(-eg_probs, kth=4, axis=1)[:, :5]
    top1_eg = eg_probs.argmax(axis=1)

    top1_pred_counts = Counter()
    wrong_pred_counts = Counter()
    true_on_wrong_counts = Counter()
    examples = []
    correct = 0

    for i, (true_eg0, pred_eg0) in enumerate(zip(y.tolist(), top1_eg.tolist())):
        true_eg = int(true_eg0) + 1
        pred_eg = int(pred_eg0) + 1
        top5 = [int(v) + 1 for v in top5_eg[i].tolist()]
        top1_pred_counts[pred_eg] += 1
        is_correct = pred_eg == true_eg
        if is_correct:
            correct += 1
        else:
            wrong_pred_counts[pred_eg] += 1
            true_on_wrong_counts[true_eg] += 1
        examples.append(
            {
                "index": i,
                "case_id": case_ids[i],
                "true_eg": true_eg,
                "pred_eg": pred_eg,
                "top5_eg": top5,
                "correct": is_correct,
            }
        )

    summary = {
        "eval_data_path": args.eval_h5,
        "sqrt": args.sqrt,
        "models": {
            args.model_name: {
                "accuracy_top1_pct": 100.0 * correct / len(y),
                "total": int(len(y)),
                "top1_pred_counts": summarize_counter(top1_pred_counts),
                "wrong_pred_counts": summarize_counter(wrong_pred_counts),
                "true_on_wrong_counts": summarize_counter(true_on_wrong_counts),
                "eg98_wrong_count": int(wrong_pred_counts.get(98, 0)),
                "eg14_wrong_count": int(wrong_pred_counts.get(14, 0)),
                "examples": examples,
            }
        },
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {
                args.model_name: {
                    "accuracy_top1_pct": summary["models"][args.model_name]["accuracy_top1_pct"],
                    "eg98_wrong_count": summary["models"][args.model_name]["eg98_wrong_count"],
                    "eg14_wrong_count": summary["models"][args.model_name]["eg14_wrong_count"],
                    "sqrt": args.sqrt,
                }
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
