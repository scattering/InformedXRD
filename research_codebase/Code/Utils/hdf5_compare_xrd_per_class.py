import h5py
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import pearsonr, spearmanr
from scipy.signal import correlate
from scipy.stats import wasserstein_distance
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path

# =====================================================
# CONFIG
# =====================================================
RRUFF_H5 = "/scratch/10486/UserB/Data/RRUFF_low_bkg_full_intensity_cleaned.hdf5"
SYNTH_H5 = "/scratch/10486/UserB/Data/10Million_pyxtal_RRUFFF-type_repacked_sampled_01.hdf5"

OUT_DIR  = Path("/scratch/10486/UserB/Data/xrd_comparison_outputs")
PLOT_DIR = OUT_DIR / "per_class_plots"
CSV_PATH = OUT_DIR / "per_class_metrics.csv"

USE_CLS  = False
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

OUT_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)

# =====================================================
# LOAD ViT MODEL (OPTIONAL)
# =====================================================
if USE_CLS:
    from vit_model import ViTXRD  # your model

    model = ViTXRD(...)
    model.load_state_dict(torch.load("model.pt", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    @torch.no_grad()
    def extract_cls(x_np):
        x = torch.tensor(x_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        emb = model(x, return_embedding=True)
        emb = F.normalize(emb, dim=1)
        return emb.squeeze(0).cpu().numpy()
else:
    extract_cls = None

# =====================================================
# HELPERS
# =====================================================
def normalize(x):
    x = x - np.min(x)
    s = np.sum(x)
    return x / s if s > 0 else x

def cosine_sim(x, y):
    return float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))

def cross_corr(x, y):
    cc = correlate(x, y, mode="full")
    shift = int(cc.argmax() - (len(x) - 1))
    return float(cc.max()), shift

# =====================================================
# LOAD DATA
# =====================================================
with h5py.File(RRUFF_H5, "r") as f:
    X_rr = f["X_test"][:]
    y_rr = f["y_test"][:]

with h5py.File(SYNTH_H5, "r") as f:
    X_sy = f["X_train"][:]
    y_sy = f["y_train"][:]

# =====================================================
# GROUP BY CLASS
# =====================================================
rr_by_class = defaultdict(list)
sy_by_class = defaultdict(list)

for x, y in zip(X_rr, y_rr):
    rr_by_class[int(y)].append(x)

for x, y in zip(X_sy, y_sy):
    sy_by_class[int(y)].append(x)

classes = sorted(set(rr_by_class) & set(sy_by_class))

# =====================================================
# MAIN LOOP
# =====================================================
rows = []

for cls in tqdm(classes, desc="Processing classes"):
    rr = np.array(rr_by_class[cls])
    sy = np.array(sy_by_class[cls])

    rr_mean = normalize(rr.mean(axis=0))
    sy_mean = normalize(sy.mean(axis=0))

    rr_std = rr.std(axis=0)
    sy_std = sy.std(axis=0)

    pearson = pearsonr(rr_mean, sy_mean)[0]
    spearman = spearmanr(rr_mean, sy_mean)[0]
    cosine = cosine_sim(rr_mean, sy_mean)
    wasser = wasserstein_distance(rr_mean, sy_mean)
    xcorr_max, xcorr_shift = cross_corr(rr_mean, sy_mean)

    if USE_CLS:
        rr_emb = extract_cls(rr_mean)
        sy_emb = extract_cls(sy_mean)
        cls_cos = cosine_sim(rr_emb, sy_emb)
    else:
        cls_cos = 0.0

    # -------------------------------------------------
    # Synthetic realism score (bounded, interpretable)
    # -------------------------------------------------
    realism_score = (
        0.35 * cls_cos +
        0.25 * cosine +
        0.20 * pearson -
        0.20 * wasser
    )

    rows.append({
        "class": cls,
        "pearson": pearson,
        "spearman": spearman,
        "cosine": cosine,
        "wasserstein": wasser,
        "xcorr_max": xcorr_max,
        "xcorr_shift": xcorr_shift,
        "cls_cosine": cls_cos,
        "realism_score": realism_score,
        "n_rruff": len(rr),
        "n_synth_train": len(sy),
    })

    # -------------------------------------------------
    # Overlay plot
    # -------------------------------------------------
    x_axis = np.arange(len(rr_mean))

    plt.figure(figsize=(14, 5))
    plt.plot(x_axis, rr_mean, label="RRUFF (mean)", linewidth=2)
    plt.fill_between(
        x_axis,
        rr_mean - rr_std,
        rr_mean + rr_std,
        alpha=0.3,
    )

    plt.plot(x_axis, sy_mean, label="Synthetic Train (mean)", linewidth=2)
    plt.fill_between(
        x_axis,
        sy_mean - sy_std,
        sy_mean + sy_std,
        alpha=0.3,
    )

    plt.title(
        f"Class {cls} | Realism Score = {realism_score:.3f}"
    )
    plt.xlabel("2θ / Q index")
    plt.ylabel("Normalized Intensity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"class_{cls}.png", dpi=200)
    plt.close()

# =====================================================
# SAVE CSV (RANKED)
# =====================================================
df = pd.DataFrame(rows)
df = df.sort_values("realism_score", ascending=False)
df.to_csv(CSV_PATH, index=False)

# =====================================================
# SUMMARY
# =====================================================
print("\n=== TOP 10 MOST REALISTIC CLASSES ===\n")
print(df.head(10)[["class", "realism_score"]])

print("\n=== BOTTOM 10 LEAST REALISTIC CLASSES ===\n")
print(df.tail(10)[["class", "realism_score"]])

print(f"\nCSV written to: {CSV_PATH}")
print(f"Plots written to: {PLOT_DIR}")
