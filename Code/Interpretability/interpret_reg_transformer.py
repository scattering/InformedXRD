"""
Step 2: Load pre-generated spectrum + HKL data from gen_spectrum.py,
run through Reg_Transformer_FlashAttn, and plot attention overlay.

Run gen_spectrum.py first (in the cctbx_env) to populate gen_output/.
Run this script in the nvidia env.

Key differences from the ViT notebook:
  - Model loaded with use_flash_attn=False so attention weights are materialised.
    (Flash Attention never computes the full attention matrix.)
  - No CLS token → attention projected via column mean instead of CLS row.
"""

import sys
import os
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import OrderedDict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Reg_Transformer_FlashAttn"))
from model import transformer_model

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
CHECKPOINT  = "/scratch/ALLOC1/UserA/models/xrd_model_hwixtnv7.pth"
GEN_DIR     = os.path.join(os.path.dirname(__file__), "gen_output")
SAVE_DIR    = "./attention_plots"

SPEC_LENGTH = 8501

# Model hyperparameters — must match the checkpoint
NUM_CLASSES = 99
EMBED_DIM   = 256
FF_DIM      = 128
DEPTH       = 6
NUM_HEADS   = 4
MLP_UNITS   = 256

SAMPLE_INDICES = [16, 26]
# ═══════════════════════════════════════════════════════════════════════════


def load_model(checkpoint_path, device):
    """
    Load with use_flash_attn=False so MultiHeadAttentionWithRoPE is used,
    which stores last_attn_weights. Flash Attention never materialises the
    attention matrix so it cannot be used for interpretability.
    """
    model = transformer_model(
        spec_length=SPEC_LENGTH,
        num_output=NUM_CLASSES,
        embed_dim=EMBED_DIM,
        ff_dim=FF_DIM,
        depth=DEPTH,
        num_heads=NUM_HEADS,
        mlp_units=MLP_UNITS,
        dropout=0.0,
        pos_encoding="rope",
        use_flash_attn=False,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model", checkpoint)
    clean = OrderedDict((k.replace("module.", ""), v) for k, v in state_dict.items())
    model.load_state_dict(clean, strict=False)
    model.to(device).eval()
    return model


def load_generated(gen_dir, sample_index):
    """Load pre-generated spectrum + HKL data from gen_output/."""
    npz  = np.load(os.path.join(gen_dir, f"sample_{sample_index}.npz"))
    with open(os.path.join(gen_dir, f"sample_{sample_index}.json")) as f:
        meta = json.load(f)
    return npz["twotheta"], npz["intensity"], npz["twotheta_peaks"], meta


def get_attention(model, spectrum_np, device):
    """Run spectrum through model, return (logits, attn_weights)."""
    x = torch.tensor(spectrum_np, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
    attn = model.transformer_blocks[-1].attention.last_attn_weights.squeeze(0).cpu()
    return logits.squeeze(0).cpu(), attn


def project_attention(attn_weights):
    """
    Project (num_heads, seq_len, seq_len) to a 1-D attention profile.

    attn_weights[h, i, j] = how much position i attends to position j (head h).
    Column mean: for each position j, how much is it attended to by all others.
    This is the right signal for average-pooling models (no CLS token).
    """
    attn_avg = attn_weights.mean(0)          # (seq_len, seq_len)
    attn_profile = attn_avg.mean(0).numpy()  # (seq_len,) — column mean
    attn_profile /= attn_profile.sum()
    return attn_profile


def plot_pred_vs_actual(logits, extinction_group, sample_index, save_dir,
                        class_names=None):
    """
    Horizontal bar chart of top-5 predicted probabilities vs actual label.
    Mirrors plot_predicted_vs_actual_single from the ViT notebook.
    extinction_group is 1-indexed; logits are 0-indexed.
    """
    os.makedirs(save_dir, exist_ok=True)

    probs = torch.softmax(logits, dim=-1)
    top5_probs, top5_preds = torch.topk(probs, 5)

    actual_idx = extinction_group - 1
    pred = top5_preds[0].item()
    correct = pred == actual_idx

    if class_names is None:
        class_names = [str(i) for i in range(1, NUM_CLASSES + 1)]

    pred_name   = class_names[pred]
    actual_name = class_names[actual_idx]
    labels  = [class_names[p.item()] for p in top5_preds]
    colors  = ["green" if top5_preds[i].item() == actual_idx else "steelblue" for i in range(5)]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(labels[::-1], top5_probs.numpy()[::-1], color=colors[::-1])
    ax.set_xlabel("Probability")
    ax.set_xlim(0, 1)
    ax.set_title(
        f"Sample {sample_index} — Actual: {actual_name} | Pred: {pred_name} "
        + ("✓" if correct else "✗"),
        color="green" if correct else "red",
    )
    for i, (prob, label) in enumerate(zip(top5_probs.numpy()[::-1], labels[::-1])):
        ax.text(prob + 0.01, i, f"{prob:.3f}", va="center", fontsize=9)

    fig.tight_layout()
    path = os.path.join(save_dir, f"pred_vs_actual_{sample_index}.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    print(f"  Actual: {actual_name} | Predicted: {pred_name} | {'✓ Correct' if correct else '✗ Incorrect'}")


def plot(twotheta, intensity, twotheta_peaks, hkls, attn_profile,
         extinction_group, crystal_system, pred_class, sample_index,
         save_dir, n_labels=10, num_bins=50, min_x_sep=0.8, y_step=0.06):

    os.makedirs(save_dir, exist_ok=True)

    peak_max = intensity.max()
    y_max = peak_max * (1.20 if n_labels > 0 else 1.10)

    fig, ax = plt.subplots(figsize=(14, 5))

    # Attention overlay
    attn_norm = (attn_profile - attn_profile.min()) / (attn_profile.max() - attn_profile.min() + 1e-8)
    bin_edges = np.linspace(twotheta[0], twotheta[-1], num_bins + 1)
    bin_indices = np.digitize(twotheta, bin_edges) - 1
    for b in range(num_bins):
        mask = bin_indices == b
        if np.any(mask):
            ax.fill_between(twotheta[mask], 0, y_max,
                            color=(1.0, 0.0, 0.0, float(attn_norm[mask].mean())),
                            step="mid")

    # XRD pattern
    ax.plot(twotheta, intensity, "k-", linewidth=0.8, zorder=3)

    # HKL labels with stagger for close peaks
    if n_labels > 0 and len(twotheta_peaks) > 0:
        peak_intensities = np.interp(twotheta_peaks, twotheta, intensity)
        top_idx = np.argsort(peak_intensities)[-n_labels:]
        placed = []
        for i in top_idx:
            h, k, l = hkls[i]
            x = twotheta_peaks[i]
            y = peak_intensities[i] + 0.02 * peak_max
            for px, py in placed:
                if abs(x - px) < min_x_sep:
                    y = max(y, py + y_step * peak_max)
            placed.append((x, y))
            ax.annotate(f"({h}{k}{l})",
                        xy=(x, peak_intensities[i]),
                        xytext=(x, y),
                        xycoords="data", textcoords="data",
                        ha="center", va="bottom",
                        fontsize=7, rotation=45,
                        arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))

    ax.set_xlabel("2θ (degrees)")
    ax.set_ylabel("Intensity")
    ax.set_xlim(twotheta[0], twotheta[-1])
    ax.set_ylim(0, y_max)
    correct = (pred_class + 1) == extinction_group
    ax.set_title(
        f"Index {sample_index} | {crystal_system} | "
        f"Extinction group: {extinction_group} | Pred: {pred_class + 1} "
        + ("✓" if correct else "✗"),
        color="green" if correct else "red",
    )

    path = os.path.join(save_dir, f"attn_{sample_index}_eg{extinction_group}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading model ...")
    model = load_model(CHECKPOINT, device)

    for sample_index in SAMPLE_INDICES:
        print(f"\n── Sample {sample_index} ──")

        twotheta, intensity, twotheta_peaks, meta = load_generated(GEN_DIR, sample_index)
        hkls = meta["hkls"]
        extinction_group  = meta["extinction_group"]
        crystal_system    = meta["crystal_system"]
        print(f"  SG {meta['spacegroup_number']} | {crystal_system} | extinction group {extinction_group}")

        logits, attn_weights = get_attention(model, intensity, device)
        pred_class = int(logits.argmax().item())
        print(f"  Predicted: {pred_class + 1} | True: {extinction_group}")

        attn_profile = project_attention(attn_weights)

        plot_pred_vs_actual(logits, extinction_group, sample_index, SAVE_DIR)
        plot(twotheta, intensity, twotheta_peaks, hkls, attn_profile,
             extinction_group, crystal_system, pred_class, sample_index, SAVE_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()
