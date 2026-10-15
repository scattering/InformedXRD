#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


"""
Reconstruct the RRUFF Option-1 family/scan labeling stage from raw per-pattern HDF5.

Recovered pieces from the historical workflow:
- exact pattern length 8501
- standard 5-90 degree grid with 0.01 degree step
- Cu-like wavelength gate
- max intensity <= 1000
- Option-1 semantics:
  * label repeated-mineral families as tight / mixed / heterogeneous
  * label scans as representative / core / outlier
  * keep all tight-family scans
  * keep only representative/core scans from mixed families
  * keep no scans from heterogeneous families

Missing from the recovered original code:
- exact low-angle window
- exact numerical thresholds for family and scan labels

This script therefore implements a documented replacement, not the original code.
Defaults are chosen to be physically plausible and easy to tune against the preserved
473-scan target.
"""


EXPECTED_LENGTH = 8501
EXPECTED_TWOTHETA_MIN = 5.0
EXPECTED_TWOTHETA_MAX = 90.0
EXPECTED_STEP = 0.01
DEFAULT_LOW_ANGLE_MAX = 35.0
DEFAULT_INTENSITY_MAX = 1000.0
CU_LIKE_MIN = 1.53
CU_LIKE_MAX = 1.55


@dataclass
class PatternRecord:
    case_id: str
    mineral: str
    intensity: np.ndarray
    two_theta: np.ndarray
    eg: int | None
    wavelength_text: str | None
    wavelength_values: list[float]
    metadata: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct the family/scan labeling step used to build the Option-1 473-scan RRUFF subset."
    )
    parser.add_argument("--input-h5", required=True, help="Raw per-pattern RRUFF HDF5")
    parser.add_argument("--output-dir", required=True, help="Directory for JSON/CSV/HDF5 outputs")
    parser.add_argument("--output-h5-name", default="RRUFF_option1_473_reconstructed.hdf5")
    parser.add_argument(
        "--reference-manifest-json",
        help="Optional preserved Option-1 manifest JSON for overlap diagnostics",
    )
    parser.add_argument("--low-angle-max", type=float, default=DEFAULT_LOW_ANGLE_MAX)
    parser.add_argument("--max-intensity", type=float, default=DEFAULT_INTENSITY_MAX)
    parser.add_argument("--require-cu-like", dest="require_cu_like", action="store_true")
    parser.add_argument("--no-require-cu-like", dest="require_cu_like", action="store_false")
    parser.set_defaults(require_cu_like=True)
    parser.add_argument("--allow-nonstandard-grid", action="store_true")
    parser.add_argument("--full-corr-core-min", type=float, default=0.93)
    parser.add_argument("--low-corr-core-min", type=float, default=0.95)
    parser.add_argument("--rank1-core-max", type=float, default=0.18)
    parser.add_argument("--full-corr-tight-min", type=float, default=0.96)
    parser.add_argument("--low-corr-tight-min", type=float, default=0.97)
    parser.add_argument("--rank1-tight-max", type=float, default=0.12)
    parser.add_argument("--heterogeneous-outlier-frac", type=float, default=0.5)
    parser.add_argument("--heterogeneous-mean-full-corr-max", type=float, default=0.88)
    parser.add_argument("--heterogeneous-mean-low-corr-max", type=float, default=0.90)
    return parser.parse_args()


def normalize_vector(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    max_val = float(np.max(np.abs(x)))
    if max_val <= 0:
        return np.zeros_like(x, dtype=np.float64)
    return x / max_val


def parse_numeric_values(text: str | None) -> list[float]:
    if not text:
        return []
    vals: list[float] = []
    cur = []
    for ch in str(text):
        if ch.isdigit() or ch in {".", "-", "+"}:
            cur.append(ch)
        else:
            if cur:
                try:
                    vals.append(float("".join(cur)))
                except ValueError:
                    pass
                cur = []
    if cur:
        try:
            vals.append(float("".join(cur)))
        except ValueError:
            pass
    return vals


def is_cu_like(vals: list[float]) -> bool:
    return any(CU_LIKE_MIN <= v <= CU_LIKE_MAX for v in vals)


def infer_mineral(case_id: str, attrs: dict[str, Any]) -> str:
    for key in ("mineral", "mineral_name", "name", "phase_name"):
        if key in attrs and str(attrs[key]).strip():
            return str(attrs[key]).strip()
    if "__" in case_id:
        return case_id.split("__", 1)[0]
    return case_id


def decode_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, np.ndarray) and value.dtype.kind in {"S", "O"} and value.size == 1:
        item = value.reshape(-1)[0]
        if isinstance(item, bytes):
            return item.decode("utf-8", errors="ignore")
    if isinstance(value, np.generic):
        return value.item()
    return value


def get_dataset(group: h5py.Group, names: list[str]) -> np.ndarray | None:
    for name in names:
        if name in group:
            return np.asarray(group[name][...])
    return None


def get_eg(attrs: dict[str, Any]) -> int | None:
    for key in ("eg", "extinction_group", "y", "label"):
        if key in attrs:
            try:
                return int(attrs[key])
            except Exception:
                return None
    return None


def get_wavelength_text(attrs: dict[str, Any]) -> str | None:
    for key in ("wavelength", "xray_wavelength", "x_ray_wavelength", "wavelengths"):
        if key in attrs:
            return str(attrs[key])
    return None


def load_reference_ids(path: str | None) -> set[str]:
    if not path:
        return set()
    obj = json.loads(Path(path).read_text())
    if isinstance(obj, dict):
        rows = obj.get("rows") or obj.get("cases") or obj.get("data") or []
    else:
        rows = obj
    ref_ids: set[str] = set()
    for row in rows:
        rid = row.get("rid") if isinstance(row, dict) else None
        if rid:
            ref_ids.add(str(rid))
    return ref_ids


def validate_grid(two_theta: np.ndarray) -> bool:
    if two_theta.ndim != 1 or len(two_theta) != EXPECTED_LENGTH:
        return False
    if not np.isclose(float(two_theta[0]), EXPECTED_TWOTHETA_MIN, atol=5e-3):
        return False
    if not np.isclose(float(two_theta[-1]), EXPECTED_TWOTHETA_MAX, atol=5e-3):
        return False
    diffs = np.diff(two_theta)
    return bool(np.allclose(diffs, EXPECTED_STEP, atol=1e-4))


def load_raw_patterns(path: Path, args: argparse.Namespace) -> tuple[list[PatternRecord], dict[str, int]]:
    kept: list[PatternRecord] = []
    skipped = Counter()
    with h5py.File(path, "r") as handle:
        for case_id in handle.keys():
            item = handle[case_id]
            if not isinstance(item, h5py.Group):
                skipped["not_group"] += 1
                continue

            intensity = get_dataset(item, ["intensity", "I", "xrd", "pattern"])
            if intensity is None:
                skipped["missing_intensity"] += 1
                continue
            intensity = np.asarray(intensity, dtype=np.float64).reshape(-1)
            if intensity.shape[0] != EXPECTED_LENGTH:
                skipped["bad_length"] += 1
                continue
            if float(np.max(intensity)) > args.max_intensity:
                skipped["intensity_over_max"] += 1
                continue

            two_theta = get_dataset(item, ["two_theta", "2theta", "x", "twotheta"])
            if two_theta is None:
                two_theta = np.linspace(
                    EXPECTED_TWOTHETA_MIN,
                    EXPECTED_TWOTHETA_MAX,
                    EXPECTED_LENGTH,
                    dtype=np.float64,
                )
            else:
                two_theta = np.asarray(two_theta, dtype=np.float64).reshape(-1)
            if not args.allow_nonstandard_grid and not validate_grid(two_theta):
                skipped["nonstandard_grid"] += 1
                continue

            attrs = {k: decode_attr(v) for k, v in item.attrs.items()}
            wavelength_text = get_wavelength_text(attrs)
            wavelength_values = parse_numeric_values(wavelength_text)
            if args.require_cu_like and not is_cu_like(wavelength_values):
                skipped["non_cu_like"] += 1
                continue

            kept.append(
                PatternRecord(
                    case_id=case_id,
                    mineral=infer_mineral(case_id, attrs),
                    intensity=normalize_vector(intensity),
                    two_theta=two_theta,
                    eg=get_eg(attrs),
                    wavelength_text=wavelength_text,
                    wavelength_values=wavelength_values,
                    metadata=attrs,
                )
            )
    return kept, dict(skipped)


def corrcoef_safe(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def family_metrics(records: list[PatternRecord], low_angle_max: float) -> dict[str, Any]:
    X = np.stack([r.intensity for r in records], axis=0)
    tt = records[0].two_theta
    low_mask = tt <= low_angle_max

    n = X.shape[0]
    full_corr = np.eye(n, dtype=np.float64)
    low_corr = np.eye(n, dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            fc = corrcoef_safe(X[i], X[j])
            lc = corrcoef_safe(X[i, low_mask], X[j, low_mask])
            full_corr[i, j] = full_corr[j, i] = fc
            low_corr[i, j] = low_corr[j, i] = lc

    X_centered = X - X.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(X_centered, full_matrices=False)
    rank1 = np.outer(u[:, 0] * s[0], vt[0]) if s.size else np.zeros_like(X_centered)
    residual = X_centered - rank1
    denom = np.linalg.norm(X_centered, axis=1)
    denom[denom < 1e-12] = 1.0
    rank1_resid = np.linalg.norm(residual, axis=1) / denom

    mean_full = full_corr.mean(axis=1)
    mean_low = low_corr.mean(axis=1)
    combined_score = 0.5 * mean_full + 0.5 * mean_low - 0.25 * rank1_resid
    medoid_idx = int(np.argmax(combined_score))

    offdiag = ~np.eye(n, dtype=bool)
    return {
        "full_corr": full_corr,
        "low_corr": low_corr,
        "rank1_resid": rank1_resid,
        "mean_full_corr": mean_full,
        "mean_low_corr": mean_low,
        "medoid_idx": medoid_idx,
        "family_mean_full_corr": float(full_corr[offdiag].mean()) if n > 1 else 1.0,
        "family_mean_low_corr": float(low_corr[offdiag].mean()) if n > 1 else 1.0,
        "family_max_rank1_resid": float(rank1_resid.max()),
    }


def classify_family(records: list[PatternRecord], metrics: dict[str, Any], args: argparse.Namespace) -> tuple[str, list[str], list[bool]]:
    n = len(records)
    if n == 1:
        return "tight", ["representative"], [True]

    medoid_idx = metrics["medoid_idx"]
    rank1_resid = metrics["rank1_resid"]
    full_to_medoid = metrics["full_corr"][:, medoid_idx]
    low_to_medoid = metrics["low_corr"][:, medoid_idx]

    roles = ["outlier"] * n
    keep = [False] * n
    roles[medoid_idx] = "representative"
    keep[medoid_idx] = True

    for i in range(n):
        if i == medoid_idx:
            continue
        if (
            full_to_medoid[i] >= args.full_corr_core_min
            and low_to_medoid[i] >= args.low_corr_core_min
            and rank1_resid[i] <= args.rank1_core_max
        ):
            roles[i] = "core"
            keep[i] = True

    core_count = sum(r in {"representative", "core"} for r in roles)
    outlier_frac = 1.0 - (core_count / n)
    family_mean_full = metrics["family_mean_full_corr"]
    family_mean_low = metrics["family_mean_low_corr"]
    family_max_rank1 = metrics["family_max_rank1_resid"]

    if (
        outlier_frac >= args.heterogeneous_outlier_frac
        or family_mean_full <= args.heterogeneous_mean_full_corr_max
        or family_mean_low <= args.heterogeneous_mean_low_corr_max
    ):
        return "heterogeneous", roles, [False] * n

    if (
        core_count == n
        and family_mean_full >= args.full_corr_tight_min
        and family_mean_low >= args.low_corr_tight_min
        and family_max_rank1 <= args.rank1_tight_max
    ):
        return "tight", ["representative" if i == medoid_idx else "core" for i in range(n)], [True] * n

    return "mixed", roles, keep


def write_outputs(records: list[PatternRecord], families: dict[str, list[PatternRecord]], args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    case_rows: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []
    selected_records: list[PatternRecord] = []
    reference_ids = load_reference_ids(args.reference_manifest_json)

    for mineral, fam_records in sorted(families.items()):
        metrics = family_metrics(fam_records, args.low_angle_max)
        family_label, roles, keep = classify_family(fam_records, metrics, args)

        family_rows.append(
            {
                "mineral": mineral,
                "n_cases": len(fam_records),
                "family_label": family_label,
                "medoid_case_id": fam_records[metrics["medoid_idx"]].case_id,
                "family_mean_full_corr": round(metrics["family_mean_full_corr"], 6),
                "family_mean_low_corr": round(metrics["family_mean_low_corr"], 6),
                "family_max_rank1_resid": round(metrics["family_max_rank1_resid"], 6),
                "n_selected": int(sum(keep)),
            }
        )

        for rec, role, keep_flag, rank1, mean_full, mean_low in zip(
            fam_records,
            roles,
            keep,
            metrics["rank1_resid"],
            metrics["mean_full_corr"],
            metrics["mean_low_corr"],
        ):
            row = {
                "case_id": rec.case_id,
                "mineral": rec.mineral,
                "family_label": family_label,
                "scan_label": role,
                "selected": bool(keep_flag),
                "eg": rec.eg,
                "mean_full_corr": round(float(mean_full), 6),
                "mean_low_corr": round(float(mean_low), 6),
                "rank1_resid": round(float(rank1), 6),
                "wavelength": rec.wavelength_text,
            }
            case_rows.append(row)
            if keep_flag:
                selected_records.append(rec)

    summary = {
        "input_h5": str(Path(args.input_h5)),
        "upstream_pool_size": len(records),
        "n_families": len(families),
        "selected_cases": len(selected_records),
        "selected_minerals": len({r.mineral for r in selected_records}),
        "family_label_counts": dict(Counter(row["family_label"] for row in family_rows)),
        "scan_label_counts": dict(Counter(row["scan_label"] for row in case_rows)),
        "notes": {
            "status": "documented reconstructed replacement",
            "recovered_steps": [
                "8501-point length gate",
                "5-90 degree standard grid",
                "0.01 degree step",
                "Cu-like wavelength gate",
                "max intensity <= 1000",
                "Option-1 keep/drop semantics",
            ],
            "reconstructed_thresholds": {
                "low_angle_max": args.low_angle_max,
                "full_corr_core_min": args.full_corr_core_min,
                "low_corr_core_min": args.low_corr_core_min,
                "rank1_core_max": args.rank1_core_max,
                "full_corr_tight_min": args.full_corr_tight_min,
                "low_corr_tight_min": args.low_corr_tight_min,
                "rank1_tight_max": args.rank1_tight_max,
                "heterogeneous_outlier_frac": args.heterogeneous_outlier_frac,
                "heterogeneous_mean_full_corr_max": args.heterogeneous_mean_full_corr_max,
                "heterogeneous_mean_low_corr_max": args.heterogeneous_mean_low_corr_max,
            },
        },
    }

    if reference_ids:
        selected_ids = {r.case_id for r in selected_records}
        overlap = selected_ids & reference_ids
        summary["reference_overlap"] = {
            "reference_cases": len(reference_ids),
            "selected_cases": len(selected_ids),
            "intersection": len(overlap),
            "selected_only": len(selected_ids - reference_ids),
            "reference_only": len(reference_ids - selected_ids),
            "recall_vs_reference": round(len(overlap) / len(reference_ids), 6) if reference_ids else 0.0,
            "precision_vs_reference": round(len(overlap) / len(selected_ids), 6) if selected_ids else 0.0,
        }

    (out_dir / "option1_reconstruction_summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "option1_reconstruction_cases.json").write_text(json.dumps(case_rows, indent=2))
    (out_dir / "option1_reconstruction_families.json").write_text(json.dumps(family_rows, indent=2))

    with (out_dir / "option1_reconstruction_cases.csv").open("w", newline="") as handle:
        columns = list(case_rows[0].keys()) if case_rows else []
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in case_rows:
            writer.writerow(row)

    with (out_dir / "option1_reconstruction_families.csv").open("w", newline="") as handle:
        columns = list(family_rows[0].keys()) if family_rows else []
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in family_rows:
            writer.writerow(row)

    output_h5 = out_dir / args.output_h5_name
    if selected_records:
        X = np.stack([r.intensity.astype(np.float32) for r in selected_records], axis=0)
        y = np.asarray([(-1 if r.eg is None else r.eg) for r in selected_records], dtype=np.int64)
        case_ids = np.asarray([r.case_id for r in selected_records], dtype=h5py.string_dtype("utf-8"))
        minerals = np.asarray([r.mineral for r in selected_records], dtype=h5py.string_dtype("utf-8"))
        two_theta = selected_records[0].two_theta.astype(np.float32)
        with h5py.File(output_h5, "w") as handle:
            handle.create_dataset("X_test", data=X, compression="gzip")
            handle.create_dataset("y_test", data=y)
            handle.create_dataset("case_ids", data=case_ids)
            handle.create_dataset("minerals", data=minerals)
            handle.create_dataset("two_theta", data=two_theta)
            handle.attrs["source_format"] = "reconstructed_option1"
            handle.attrs["status"] = "documented reconstructed replacement"
            handle.attrs["selected_cases"] = len(selected_records)
            handle.attrs["selected_minerals"] = len({r.mineral for r in selected_records})
            handle.attrs["low_angle_max"] = args.low_angle_max


def main() -> None:
    args = parse_args()
    records, skipped = load_raw_patterns(Path(args.input_h5), args)

    families: dict[str, list[PatternRecord]] = defaultdict(list)
    for rec in records:
        families[rec.mineral].append(rec)

    write_outputs(records, families, args)

    summary_path = Path(args.output_dir) / "option1_reconstruction_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["skipped"] = skipped
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
