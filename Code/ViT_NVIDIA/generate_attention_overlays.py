#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import find_peaks

from extinction_multilabel import (
    build_template_bank,
    template_mask_from_ext_groups,
    topk_decoded_split_head_ext_groups,
)
from model import VIT_model, adapt_patch_embed_input_channels


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate attention overlays for multilabel ViT checkpoints on real benchmark spectra."
    )
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path")
    parser.add_argument("--config", required=True, help="Config JSON path")
    parser.add_argument("--eval-data-path", required=True, help="Evaluation HDF5 with X_test/y_test")
    parser.add_argument("--output-dir", required=True, help="Directory for PNGs and summary JSON")
    parser.add_argument(
        "--decoder",
        default="aux_bayes",
        choices=["aux_bayes", "aux", "split"],
        help="Which decoder to summarize in the output metadata.",
    )
    parser.add_argument(
        "--prior-data-path",
        default=None,
        help="Optional HDF5 supplying y_train for seen-mask and Bayesian priors.",
    )
    parser.add_argument(
        "--aux-temperature",
        type=float,
        default=1.0,
        help="Temperature applied to aux logits before decoding.",
    )
    parser.add_argument("--topk", type=int, default=5, help="Number of decoded EGs to record")
    parser.add_argument("--sample-indices", nargs="*", type=int, default=[], help="0-based X_test indices")
    parser.add_argument("--case-ids", nargs="*", default=[], help="Optional case_ids from the HDF5")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--dif-root",
        default="/home1/ALLOC3/UserE/codex/rruff-fit/data/dif",
        help="Directory containing RRUFF DIF files for HKL annotation.",
    )
    parser.add_argument(
        "--annotate-hkl",
        action="store_true",
        help="Annotate top measured peaks with nearest HKLs from the corresponding DIF cell.",
    )
    parser.add_argument(
        "--max-hkl-index",
        type=int,
        default=3,
        help="Maximum absolute HKL index to generate for annotation.",
    )
    parser.add_argument(
        "--hkl-label-limit",
        type=int,
        default=12,
        help="Maximum number of HKL labels to place inside high-attention windows.",
    )
    parser.add_argument(
        "--hkl-min-separation-deg",
        type=float,
        default=0.3,
        help="Minimum 2theta separation between labeled peaks.",
    )
    return parser.parse_args()


def get_model_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def load_config(config_path, checkpoint, eval_data_path):
    config = dict(checkpoint.get("config", {})) if isinstance(checkpoint, dict) else {}
    with open(config_path, "r") as handle:
        config.update(json.load(handle))
    config["data_path"] = eval_data_path
    return config


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
    adapted = adapt_patch_embed_input_channels(get_model_state_dict(checkpoint), model)
    model.load_state_dict(adapted, strict=False)
    return model.to(device).eval()


def load_log_priors(h5_path, ext_group_order, smoothing=1.0):
    with h5py.File(h5_path, "r") as handle:
        y_train = handle["y_train"][:]
    counts = {int(ext): smoothing for ext in ext_group_order}
    for value in y_train.tolist():
        counts[int(value)] = counts.get(int(value), smoothing) + 1.0
    probs = torch.tensor([counts[int(ext)] for ext in ext_group_order], dtype=torch.float32)
    probs = probs / probs.sum()
    return torch.log(probs)


def load_seen_ext_groups(h5_path):
    with h5py.File(h5_path, "r") as handle:
        return sorted(set(int(v) for v in handle["y_train"][:].tolist()))


def decode_string_array(values):
    out = []
    for value in values.tolist():
        if isinstance(value, bytes):
            out.append(value.decode("utf-8"))
        else:
            out.append(str(value))
    return out


def project_attention_to_2theta(attn_map, spec_length, two_theta_min, two_theta_max):
    attn_cls = attn_map.mean(0)[0, 1:]
    attn_cls = attn_cls / attn_cls.sum()
    attn_profile = np.interp(
        np.arange(spec_length),
        np.linspace(0, spec_length - 1, len(attn_cls)),
        attn_cls.detach().cpu().numpy(),
    )
    two_theta = np.linspace(two_theta_min, two_theta_max, spec_length)
    return two_theta, attn_profile


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


def save_overlay(
    spectrum,
    attention,
    two_theta,
    title,
    output_path,
    hkls=None,
    num_bins=64,
    hkl_label_limit=12,
    hkl_min_separation_deg=0.3,
):
    spectrum = np.asarray(spectrum, dtype=np.float32)
    attention = np.asarray(attention, dtype=np.float32)
    baseline = float(np.quantile(attention, 0.75))
    hi = float(np.quantile(attention, 0.995))
    if hi <= baseline:
        baseline = float(np.quantile(attention, 0.5))
        hi = float(attention.max())
    attention = np.clip((attention - baseline) / (hi - baseline + 1e-8), 0.0, 1.0)
    attention = np.power(attention, 1.1)
    attention[attention < 0.22] = 0.0

    peak_max = float(spectrum.max())
    y_max = peak_max * 1.18 if peak_max > 0 else 1.0
    bin_edges = np.linspace(two_theta[0], two_theta[-1], num_bins + 1)
    bin_indices = np.digitize(two_theta, bin_edges) - 1
    bin_scores = np.zeros(num_bins, dtype=np.float32)
    for i in range(num_bins):
        mask = bin_indices == i
        if np.any(mask):
            bin_scores[i] = float(attention[mask].mean())

    top_bin_idx = np.argsort(bin_scores)[-18:]
    top_bin_idx = sorted(int(i) for i in top_bin_idx if bin_scores[i] > 0)
    selected_bins = set()
    for i in top_bin_idx:
        selected_bins.add(i)
        for j in (i - 3, i - 2, i - 1, i + 1, i + 2, i + 3):
            if 0 <= j < num_bins:
                selected_bins.add(j)

    moderate_bins = [int(i) for i in range(num_bins) if bin_scores[i] > 0.03]
    for i in moderate_bins:
        selected_bins.add(i)
        for j in (i - 2, i - 1, i + 1, i + 2):
            if 0 <= j < num_bins:
                selected_bins.add(j)

    # Ensure low-angle windows are not dropped when they carry non-trivial attention.
    low_angle_mask = two_theta < 30.0
    if np.any(low_angle_mask):
        low_bins = np.unique(bin_indices[low_angle_mask])
        positive_low_bins = [int(i) for i in low_bins if 0 <= i < num_bins and bin_scores[int(i)] > 0.05]
        selected_bins.update(positive_low_bins)

    # Fallback: keep bins containing strong observed peaks if they are adjacent to
    # already-retained attention bins. This preserves labels on peaks that sit on
    # the edge of a highlighted window even when the bin-average attention is low.
    peak_indices_for_bins, peak_props_for_bins = find_peaks(spectrum, prominence=max(float(spectrum.max()) * 0.02, 0.5))
    if len(peak_indices_for_bins) > 0:
        peak_bin_map: dict[int, float] = {}
        prominences = peak_props_for_bins.get("prominences", np.zeros(len(peak_indices_for_bins)))
        for p, prom in zip(peak_indices_for_bins.tolist(), prominences.tolist()):
            b = int(bin_indices[p])
            if 0 <= b < num_bins:
                peak_bin_map[b] = max(float(prom), peak_bin_map.get(b, 0.0))
        if peak_bin_map:
            max_prom = max(peak_bin_map.values())
            strong_peak_bins = {
                b for b, prom in peak_bin_map.items()
                if prom >= 0.10 * max_prom and any(abs(b - s) <= 1 for s in selected_bins)
            }
            selected_bins.update(strong_peak_bins)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    for i in range(num_bins):
        mask = bin_indices == i
        if np.any(mask) and i in selected_bins and bin_scores[i] > 0:
            ax.fill_between(
                two_theta[mask],
                0,
                y_max,
                color=(1.0, 0.0, 0.0, min(0.65, float(attention[mask].mean()) * 0.75)),
                step="mid",
            )
    ax.plot(two_theta, spectrum, color="black", linewidth=1.0)
    ax.set_xlim(float(two_theta[0]), float(two_theta[-1]))
    ax.set_ylim(0, y_max)
    ax.set_xlabel(r"2$\theta$")
    ax.set_ylabel("Intensity")
    ax.set_title(title, fontsize=10)

    if hkls:
        peak_indices_all, peak_props = find_peaks(spectrum, prominence=max(peak_max * 0.02, 0.5))
        if len(peak_indices_all) > 0:
            prominence_by_idx = {
                int(p): float(prom)
                for p, prom in zip(peak_indices_all.tolist(), peak_props.get("prominences", np.zeros(len(peak_indices_all))))
            }
            peak_indices = np.asarray(
                [p for p in peak_indices_all if int(bin_indices[p]) in selected_bins],
                dtype=int,
            )
            if len(peak_indices) == 0:
                peak_indices_all, peak_props = find_peaks(spectrum, prominence=max(peak_max * 0.01, 0.25))
                prominence_by_idx = {
                    int(p): float(prom)
                    for p, prom in zip(peak_indices_all.tolist(), peak_props.get("prominences", np.zeros(len(peak_indices_all))))
                }
                peak_indices = np.asarray(
                    [p for p in peak_indices_all if int(bin_indices[p]) in selected_bins],
                    dtype=int,
                )
            peak_scores = []
            for p in peak_indices.tolist():
                prom = prominence_by_idx.get(int(p), float(spectrum[int(p)]))
                score = prom / max(peak_max, 1e-6)
                peak_scores.append(score)
            order = np.argsort(np.asarray(peak_scores))[::-1]
            chosen = []
            for oi in order:
                p = int(peak_indices[oi])
                tt_peak = float(two_theta[p])
                if any(abs(tt_peak - float(two_theta[q])) < hkl_min_separation_deg for q in chosen):
                    continue
                chosen.append(p)
                if len(chosen) >= hkl_label_limit:
                    break
            top_peaks = np.asarray(chosen, dtype=int)
            for p in top_peaks:
                tt_peak = two_theta[p]
                y_peak = spectrum[p]
                _, hkl = min(hkls, key=lambda x: abs(x[0] - tt_peak))
                ax.text(
                    tt_peak,
                    y_peak + 0.03 * peak_max,
                    f"({hkl[0]}{hkl[1]}{hkl[2]})",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="navy",
                    rotation=90,
                    rotation_mode="anchor",
                    zorder=4,
                )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def resolve_indices(case_ids, requested_case_ids, requested_indices):
    resolved = []
    seen = set()
    for idx in requested_indices:
        if idx not in seen:
            resolved.append(idx)
            seen.add(idx)
    if requested_case_ids:
        lookup = {case_id: idx for idx, case_id in enumerate(case_ids)}
        for case_id in requested_case_ids:
            if case_id not in lookup:
                raise KeyError(f"Case id '{case_id}' not found in evaluation HDF5")
            idx = lookup[case_id]
            if idx not in seen:
                resolved.append(idx)
                seen.add(idx)
    return resolved


def main():
    args = parse_args()
    device = torch.device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = load_config(args.config, checkpoint, args.eval_data_path)
    model = build_model(config, checkpoint, device)

    template_bank, ext_group_order, _ = build_template_bank(
        canonical_table_path=config.get("canonical_table_path"),
        final_table_path=config.get("final_table_path"),
        sg_lookup_path=config.get("sg_lookup_path"),
    )
    template_bank = template_bank.to(device)
    seen_mask = None
    log_priors = None
    if args.prior_data_path:
        seen_mask = template_mask_from_ext_groups(ext_group_order, load_seen_ext_groups(args.prior_data_path)).to(device)
        log_priors = load_log_priors(args.prior_data_path, ext_group_order).to(device)

    with h5py.File(args.eval_data_path, "r") as handle:
        spectra = handle["X_test"][:]
        y_test = handle["y_test"][:]
        case_ids = decode_string_array(handle["case_ids"][:]) if "case_ids" in handle else [str(i) for i in range(len(spectra))]

    indices = resolve_indices(case_ids, args.case_ids, args.sample_indices)
    if not indices:
        raise SystemExit("No sample indices selected. Pass --sample-indices and/or --case-ids.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dif_root = Path(args.dif_root)
    summary = []

    for idx in indices:
        spectrum = torch.tensor(spectra[idx], dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            cls_embedding, attn_maps = model.forward_with_attn(spectrum, return_cls_embedding=True)

        split_logits = None
        if config.get("use_split_head", False):
            split_logits = torch.cat(
                [
                    model.head_sys(cls_embedding),
                    model.head_lat(cls_embedding),
                    model.head_ops(cls_embedding),
                ],
                dim=1,
            )
        aux_logits = model.aux_ext_head(cls_embedding) if hasattr(model, "aux_ext_head") else None

        if args.decoder == "split":
            if split_logits is None:
                raise ValueError("Requested split decoder but checkpoint has no split head")
            decoded = topk_decoded_split_head_ext_groups(
                split_logits,
                template_bank,
                ext_group_order,
                k=args.topk,
            )
        elif args.decoder == "aux":
            if aux_logits is None:
                raise ValueError("Requested aux decoder but checkpoint has no aux head")
            aux_topk = torch.topk(aux_logits / args.aux_temperature, k=args.topk, dim=1).indices
            decoded = aux_topk + 1
        else:
            if aux_logits is None:
                raise ValueError("Requested aux_bayes decoder but checkpoint has no aux head")
            bayes_aux_logits = aux_logits / args.aux_temperature
            if log_priors is not None:
                bayes_aux_logits = bayes_aux_logits + log_priors.to(
                    device=aux_logits.device, dtype=aux_logits.dtype
                ).unsqueeze(0)
            if seen_mask is not None:
                allowed = seen_mask.to(device=aux_logits.device)
                bayes_aux_logits = bayes_aux_logits.masked_fill(~allowed.unsqueeze(0), float("-inf"))
            aux_topk = torch.topk(bayes_aux_logits, k=args.topk, dim=1).indices
            decoded = aux_topk + 1

        decoded_list = [int(v) for v in decoded[0].detach().cpu().tolist()]
        last_attn = attn_maps[-1][0] if attn_maps[-1].ndim == 4 else attn_maps[-1]
        two_theta, attn_profile = project_attention_to_2theta(
            last_attn,
            spec_length=spectrum.shape[-1],
            two_theta_min=float(config.get("two_theta_min", 5.0)),
            two_theta_max=float(config.get("two_theta_max", 90.0)),
        )
        case_id = case_ids[idx]
        true_eg = int(y_test[idx])
        png_path = output_dir / f"attention_{idx:04d}_{case_id.replace('/', '_')}.png"
        npy_path = output_dir / f"attention_{idx:04d}_{case_id.replace('/', '_')}.npy"
        hkls = None
        hkl_error = None
        if args.annotate_hkl:
            try:
                hkls = get_hkls_from_dif(
                    case_id=case_id,
                    dif_root=dif_root,
                    two_theta_min=float(config.get("two_theta_min", 5.0)),
                    two_theta_max=float(config.get("two_theta_max", 90.0)),
                    max_hkl_index=args.max_hkl_index,
                )
            except Exception as exc:
                hkls = None
                hkl_error = f"{type(exc).__name__}: {exc}"
        title = (
            f"{case_id} | true EG {true_eg} | "
            f"{args.decoder} top-{min(args.topk, len(decoded_list))}: {decoded_list[:min(args.topk, len(decoded_list))]}"
        )
        save_overlay(
            spectrum=spectra[idx],
            attention=attn_profile,
            two_theta=two_theta,
            title=title,
            output_path=png_path,
            hkls=hkls,
            hkl_label_limit=args.hkl_label_limit,
            hkl_min_separation_deg=args.hkl_min_separation_deg,
        )
        np.save(npy_path, attn_profile)
        summary.append(
            {
                "sample_index": int(idx),
                "case_id": case_id,
                "true_eg": true_eg,
                "decoder": args.decoder,
                "aux_temperature": float(args.aux_temperature),
                "topk_pred_egs": decoded_list,
                "png": str(png_path),
                "attention_npy": str(npy_path),
                "hkl_annotated": hkls is not None,
                "hkl_error": hkl_error,
            }
        )

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote {len(summary)} attention overlays to {output_dir}")


if __name__ == "__main__":
    main()
