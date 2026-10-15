#!/usr/bin/env python3
import argparse
import ast
import csv
import json
import sys
from pathlib import Path

import h5py
import numpy as np


CU_KA_WAVELENGTH = 1.5406
ALPHADIFFRACT_WAVELENGTH = 12.398419843320026 / 20.0  # 20 keV


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the released OpenAlphaDiffract model on RRUFF-473."
    )
    parser.add_argument("--model-dir", required=True, help="Directory containing model.py, model.safetensors, config.json, maxsub.json")
    parser.add_argument("--eval-h5", required=True, help="RRUFF-473 benchmark HDF5")
    parser.add_argument("--mapping-csv", required=True, help="canonical_extinction_to_space_group.csv for SG->EG collapse")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-npz", default="")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--subset-json",
        help="Optional JSON with rows containing case_id; only those benchmark cases will be evaluated.",
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


def topk_accuracy(probs: np.ndarray, targets: np.ndarray, k: int) -> float:
    topk = np.argpartition(-probs, kth=min(k, probs.shape[1]) - 1, axis=1)[:, :k]
    hits = (topk == targets[:, None]).any(axis=1)
    return float(hits.mean())


def two_theta_to_q(two_theta_deg: np.ndarray, wavelength: float) -> np.ndarray:
    theta_rad = np.deg2rad(two_theta_deg / 2.0)
    return (4.0 * np.pi * np.sin(theta_rad)) / wavelength


def normalize_to_0_100(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, a_min=0.0, a_max=None)
    mins = x.min(axis=1, keepdims=True)
    maxs = x.max(axis=1, keepdims=True)
    denom = np.maximum(maxs - mins, 1e-10)
    return ((x - mins) / denom) * 100.0


def resample_rruff_to_alphadiffract_grid(x: np.ndarray, source_two_theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    src_q = two_theta_to_q(source_two_theta, CU_KA_WAVELENGTH)
    target_two_theta = np.linspace(5.0, 20.0, 8192, dtype=np.float32)
    target_q = two_theta_to_q(target_two_theta, ALPHADIFFRACT_WAVELENGTH)
    out = np.empty((x.shape[0], target_q.shape[0]), dtype=np.float32)
    for i in range(x.shape[0]):
        out[i] = np.interp(target_q, src_q, x[i], left=0.0, right=0.0)
    out = normalize_to_0_100(out)
    return out, target_two_theta, target_q


def load_case_ids(handle: h5py.File, n_rows: int) -> list[str]:
    if "case_ids" not in handle:
        return [f"row_{i}" for i in range(n_rows)]
    raw = handle["case_ids"][:]
    return [x.decode() if isinstance(x, bytes) else str(x) for x in raw]


def maybe_filter_subset(x, y, case_ids, subset_json):
    if not subset_json:
        return x, y, case_ids
    subset = json.loads(Path(subset_json).read_text())
    allowed = {row["case_id"] for row in subset["rows"]}
    keep_idx = [i for i, cid in enumerate(case_ids) if cid in allowed]
    return x[keep_idx], y[keep_idx], [case_ids[i] for i in keep_idx]


def main():
    args = parse_args()

    sys.path.insert(0, str(Path(args.model_dir).resolve()))
    import torch
    from model import AlphaDiffract

    sg_to_eg = load_sg_to_eg_map(args.mapping_csv)

    with h5py.File(args.eval_h5, "r") as handle:
        x = np.asarray(handle["X_test"], dtype=np.float32)
        y = np.asarray(handle["y_test"], dtype=np.int64)
        two_theta = np.asarray(handle["two_theta"], dtype=np.float32)
        case_ids = load_case_ids(handle, x.shape[0])

    # Benchmark exports may use canonical EG labels in 1..99 form, and some
    # subsets do not contain class 1. Normalize based on the full class range
    # rather than the minimum alone so the direct eval and Gate 1 path agree.
    if y.size and y.min() >= 1 and y.max() == (int(sg_to_eg.max()) + 1):
        y = y - 1

    if two_theta.ndim == 2:
        source_two_theta = two_theta[0]
    else:
        source_two_theta = two_theta

    x, y, case_ids = maybe_filter_subset(x, y, case_ids, args.subset_json)
    x_resampled, target_two_theta, target_q = resample_rruff_to_alphadiffract_grid(x, source_two_theta)

    model = AlphaDiffract.from_pretrained(args.model_dir, device="cpu")

    sg_chunks = []
    cs_chunks = []
    lp_chunks = []
    with torch.no_grad():
        for start in range(0, x_resampled.shape[0], args.batch_size):
            batch = torch.from_numpy(x_resampled[start : start + args.batch_size])
            out = model(batch)
            sg_chunks.append(torch.softmax(out["sg_logits"], dim=-1).cpu().numpy().astype(np.float32))
            cs_chunks.append(torch.softmax(out["cs_logits"], dim=-1).cpu().numpy().astype(np.float32))
            lp_chunks.append(out["lp"].cpu().numpy().astype(np.float32))

    sg_probs = np.concatenate(sg_chunks, axis=0)
    cs_probs = np.concatenate(cs_chunks, axis=0)
    lp_pred = np.concatenate(lp_chunks, axis=0)
    eg_probs = collapse_sg_probs_to_eg(sg_probs, sg_to_eg)

    result = {
        "model_dir": str(Path(args.model_dir).resolve()),
        "eval_h5": args.eval_h5,
        "subset_json": args.subset_json,
        "mapping_csv": args.mapping_csv,
        "num_cases": int(x_resampled.shape[0]),
        "source_grid_min": float(source_two_theta[0]),
        "source_grid_max": float(source_two_theta[-1]),
        "source_grid_len": int(source_two_theta.shape[0]),
        "target_grid_min": float(target_two_theta[0]),
        "target_grid_max": float(target_two_theta[-1]),
        "target_grid_len": int(target_two_theta.shape[0]),
        "target_q_min": float(target_q[0]),
        "target_q_max": float(target_q[-1]),
        "top1_eg": topk_accuracy(eg_probs, y, 1),
        "top5_eg": topk_accuracy(eg_probs, y, 5),
        "mean_lp_pred": lp_pred.mean(axis=0).tolist(),
        "mean_max_sg_prob": float(sg_probs.max(axis=1).mean()),
        "mean_max_eg_prob": float(eg_probs.max(axis=1).mean()),
        "sample_case_ids": case_ids[:10],
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    if args.output_npz:
        np.savez_compressed(
            args.output_npz,
            calibrated_probs=eg_probs.astype(np.float32),
            targets=y.astype(np.int64),
            case_ids=np.asarray(case_ids, dtype=object),
            sg_probs=sg_probs.astype(np.float32),
            cs_probs=cs_probs.astype(np.float32),
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
