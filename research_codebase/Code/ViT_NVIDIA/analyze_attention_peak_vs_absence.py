#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import find_peaks


def parse_args():
    p = argparse.ArgumentParser(
        description="Analyze whether attention falls on observed peaks or inter-peak / absence-sensitive regions."
    )
    p.add_argument("--summary-json", required=True)
    p.add_argument("--eval-data-path", required=True)
    p.add_argument("--dif-root", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--output-md")
    p.add_argument("--peak-window-deg", type=float, default=0.25)
    p.add_argument("--allowed-window-deg", type=float, default=0.20)
    p.add_argument("--absence-window-deg", type=float, default=0.14)
    p.add_argument("--absence-min-gap-deg", type=float, default=0.35)
    p.add_argument("--absence-max-gap-deg", type=float, default=1.20)
    p.add_argument("--peak-prominence-frac", type=float, default=0.06)
    p.add_argument("--absence-low-intensity-frac", type=float, default=0.12)
    p.add_argument("--max-hkl-index", type=int, default=3)
    return p.parse_args()


def parse_dif_metadata(path: Path) -> dict:
    text = path.read_text(errors="ignore")
    sg_match = re.search(r"SPACE GROUP:\s*([^\n\r]+)", text)
    cell_match = re.search(
        r"CELL PARAMETERS:\s*([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)",
        text,
    )
    if not sg_match or not cell_match:
        raise ValueError(f"Could not parse space group/cell from {path}")
    return {
        "space_group": sg_match.group(1).strip(),
        "cell": tuple(float(cell_match.group(i)) for i in range(1, 7)),
    }


def infer_dif_path(case_id: str, dif_root: Path) -> Path:
    parts = case_id.split("__")
    if len(parts) < 3:
        raise ValueError(f"Unexpected case_id format: {case_id}")
    dif_name = f"{parts[0]}__{parts[1]}__Powder__DIF_File__{parts[2]}.txt"
    dif_path = dif_root / dif_name
    if not dif_path.exists():
        raise FileNotFoundError(f"Could not find DIF file for {case_id}: {dif_path}")
    return dif_path


def get_hkls_from_dif(case_id: str, dif_root: Path, two_theta_min, two_theta_max, max_hkl_index=3, wavelength=1.541838):
    dif_meta = parse_dif_metadata(infer_dif_path(case_id, dif_root))
    a, b, c, alpha, beta, gamma = dif_meta["cell"]
    alpha = np.deg2rad(alpha)
    beta = np.deg2rad(beta)
    gamma = np.deg2rad(gamma)
    cos_a, cos_b, cos_g = np.cos(alpha), np.cos(beta), np.cos(gamma)
    sin_a, sin_b, sin_g = np.sin(alpha), np.sin(beta), np.sin(gamma)
    volume_factor = np.sqrt(
        max(
            1.0
            - cos_a * cos_a
            - cos_b * cos_b
            - cos_g * cos_g
            + 2.0 * cos_a * cos_b * cos_g,
            1e-12,
        )
    )
    volume = a * b * c * volume_factor
    a_star = b * c * sin_a / volume
    b_star = a * c * sin_b / volume
    c_star = a * b * sin_g / volume
    cos_a_star = (cos_b * cos_g - cos_a) / max(sin_b * sin_g, 1e-12)
    cos_b_star = (cos_a * cos_g - cos_b) / max(sin_a * sin_g, 1e-12)
    cos_g_star = (cos_a * cos_b - cos_g) / max(sin_a * sin_b, 1e-12)

    def d_hkl(h: int, k: int, l: int) -> float:
        inv_d2 = (
            (h * a_star) ** 2
            + (k * b_star) ** 2
            + (l * c_star) ** 2
            + 2.0 * h * k * a_star * b_star * cos_g_star
            + 2.0 * h * l * a_star * c_star * cos_b_star
            + 2.0 * k * l * b_star * c_star * cos_a_star
        )
        if inv_d2 <= 0:
            raise ValueError("Non-positive reciprocal metric")
        return 1.0 / np.sqrt(inv_d2)

    hkls = []
    for h in range(-max_hkl_index, max_hkl_index + 1):
        for k in range(-max_hkl_index, max_hkl_index + 1):
            for l in range(-max_hkl_index, max_hkl_index + 1):
                if h == k == l == 0:
                    continue
                try:
                    d = d_hkl(h, k, l)
                    sin_theta = wavelength / (2.0 * d)
                    if 0 < sin_theta <= 1:
                        tt = np.degrees(2.0 * np.arcsin(sin_theta))
                        if two_theta_min <= tt <= two_theta_max:
                            hkls.append((tt, (h, k, l)))
                except Exception:
                    continue
    hkls.sort(key=lambda x: x[0])
    return hkls


def attention_mass(attn: np.ndarray, mask: np.ndarray) -> float:
    attn = np.asarray(attn, dtype=float)
    attn = np.clip(attn, 0.0, None)
    s = attn.sum()
    if s <= 0:
        return 0.0
    return float(attn[mask].sum() / s)


def window_mask(two_theta: np.ndarray, centers: np.ndarray, half_width_deg: float) -> np.ndarray:
    mask = np.zeros_like(two_theta, dtype=bool)
    for c in centers:
        mask |= np.abs(two_theta - c) <= half_width_deg
    return mask


def midpoint_absence_windows(
    two_theta: np.ndarray,
    spectrum: np.ndarray,
    allowed_tts: np.ndarray,
    observed_peak_tts: np.ndarray,
    half_width_deg: float,
    min_gap_deg: float,
    max_gap_deg: float,
    low_intensity_frac: float,
) -> tuple[np.ndarray, list[float]]:
    mask = np.zeros_like(two_theta, dtype=bool)
    centers = []
    if len(allowed_tts) < 2:
        return mask, centers
    max_intensity = float(np.max(spectrum)) if len(spectrum) else 0.0
    for left, right in zip(allowed_tts[:-1], allowed_tts[1:]):
        gap = float(right - left)
        if gap < min_gap_deg or gap > max_gap_deg:
            continue
        mid = 0.5 * (left + right)
        if len(observed_peak_tts) and np.min(np.abs(observed_peak_tts - mid)) <= half_width_deg:
            continue
        local = np.abs(two_theta - mid) <= half_width_deg
        if not np.any(local):
            continue
        local_max = float(np.max(spectrum[local]))
        if max_intensity > 0 and local_max > low_intensity_frac * max_intensity:
            continue
        centers.append(mid)
        mask |= local
    return mask, centers


def group_mean(rows, key):
    vals = [row[key] for row in rows]
    return float(np.mean(vals)) if vals else None


def main():
    args = parse_args()
    summary_path = Path(args.summary_json)
    dif_root = Path(args.dif_root)
    with open(summary_path) as f:
        summary = json.load(f)

    with h5py.File(args.eval_data_path, "r") as f:
        x_test = f["X_test"]
        two_theta_all = f["two_theta"]
        case_ids = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in f["case_ids"][:]]

        rows = []
        for row in summary:
            idx = int(row["sample_index"])
            case_id = row["case_id"]
            if case_ids[idx] != case_id:
                raise ValueError(f"Case mismatch at index {idx}: {case_ids[idx]} vs {case_id}")

            spectrum = np.asarray(x_test[idx], dtype=float)
            two_theta = np.asarray(two_theta_all[idx], dtype=float)
            attn_path = Path(row["attention_npy"])
            if not attn_path.exists():
                attn_path = summary_path.parent / attn_path.name
            attn = np.load(attn_path).astype(float)

            prominence = max(float(np.max(spectrum)) * args.peak_prominence_frac, 1.0)
            peak_idx, _ = find_peaks(spectrum, prominence=prominence)
            observed_peak_tts = two_theta[peak_idx] if len(peak_idx) else np.asarray([], dtype=float)
            peak_mask = window_mask(two_theta, observed_peak_tts, args.peak_window_deg)
            inter_peak_mask = ~peak_mask

            hkls = get_hkls_from_dif(
                case_id=case_id,
                dif_root=dif_root,
                two_theta_min=float(two_theta[0]),
                two_theta_max=float(two_theta[-1]),
                max_hkl_index=args.max_hkl_index,
            )
            allowed_tts = np.asarray(sorted({float(tt) for tt, _ in hkls}), dtype=float)
            allowed_mask = window_mask(two_theta, allowed_tts, args.allowed_window_deg) if len(allowed_tts) else np.zeros_like(two_theta, dtype=bool)
            absence_mask, absence_centers = midpoint_absence_windows(
                two_theta=two_theta,
                spectrum=spectrum,
                allowed_tts=allowed_tts,
                observed_peak_tts=observed_peak_tts,
                half_width_deg=args.absence_window_deg,
                min_gap_deg=args.absence_min_gap_deg,
                max_gap_deg=args.absence_max_gap_deg,
                low_intensity_frac=args.absence_low_intensity_frac,
            )

            pred = int(row["topk_pred_egs"][0])
            label = "correct" if pred == int(row["true_eg"]) else "descendant_error"
            out = {
                "sample_index": idx,
                "case_id": case_id,
                "label": label,
                "true_eg": int(row["true_eg"]),
                "top1_pred_eg": pred,
                "n_observed_peaks": int(len(observed_peak_tts)),
                "n_allowed_reflections": int(len(allowed_tts)),
                "n_absence_windows": int(len(absence_centers)),
                "peak_window_mass": attention_mass(attn, peak_mask),
                "inter_peak_mass": attention_mass(attn, inter_peak_mask),
                "allowed_window_mass": attention_mass(attn, allowed_mask),
                "absence_window_mass": attention_mass(attn, absence_mask),
                "allowed_minus_absence_mass": attention_mass(attn, allowed_mask) - attention_mass(attn, absence_mask),
            }
            rows.append(out)

    groups = {"correct": [r for r in rows if r["label"] == "correct"], "descendant_error": [r for r in rows if r["label"] == "descendant_error"]}
    group_summary = {}
    for label, items in groups.items():
        group_summary[label] = {
            "count": len(items),
            "mean_peak_window_mass": group_mean(items, "peak_window_mass"),
            "mean_inter_peak_mass": group_mean(items, "inter_peak_mass"),
            "mean_allowed_window_mass": group_mean(items, "allowed_window_mass"),
            "mean_absence_window_mass": group_mean(items, "absence_window_mass"),
            "mean_allowed_minus_absence_mass": group_mean(items, "allowed_minus_absence_mass"),
            "mean_n_observed_peaks": group_mean(items, "n_observed_peaks"),
            "mean_n_absence_windows": group_mean(items, "n_absence_windows"),
        }

    output = {"cases": rows, "group_summary": group_summary}
    Path(args.output_json).write_text(json.dumps(output, indent=2))

    if args.output_md:
        lines = [
            "# Peak-vs-Absence Attention Summary",
            "",
            "This diagnostic compares attention near observed peaks against inter-peak and absence-sensitive windows.",
            "",
        ]
        for label in ["correct", "descendant_error"]:
            g = group_summary[label]
            lines.extend(
                [
                    f"## {label.replace('_', ' ').title()} ({g['count']} cases)",
                    "",
                    f"- Mean peak-window mass: {g['mean_peak_window_mass']:.3f}",
                    f"- Mean inter-peak mass: {g['mean_inter_peak_mass']:.3f}",
                    f"- Mean allowed-reflection mass: {g['mean_allowed_window_mass']:.3f}",
                    f"- Mean absence-window mass: {g['mean_absence_window_mass']:.3f}",
                    f"- Mean allowed-minus-absence mass: {g['mean_allowed_minus_absence_mass']:.3f}",
                    f"- Mean observed peaks: {g['mean_n_observed_peaks']:.1f}",
                    f"- Mean absence windows: {g['mean_n_absence_windows']:.1f}",
                    "",
                ]
            )
        Path(args.output_md).write_text("\n".join(lines))


if __name__ == "__main__":
    main()
