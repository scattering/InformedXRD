import torch
import wandb
import argparse
import numpy as np
from collections import OrderedDict

from model import transformer_model
from dataset import get_dataloaders  

def parse_args():
    parser = argparse.ArgumentParser(description="ViT Inference")

    # File paths
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data_path", type=str, required=True, help="Path to dataset (HDF5)")
    parser.add_argument("--run_id", type=str, required=True, help="Run identifier")
    parser.add_argument("--project", type=str, required=True, help="Project name")

    # Spectra settings
    parser.add_argument("--two_theta_min", type=float, default=10.0)
    parser.add_argument("--two_theta_max", type=float, default=90.0)

    # Model hyperparameters
    parser.add_argument("--num_classes", type=int, required=True)
    parser.add_argument("--spec_length", type=int, required=True)
    parser.add_argument("--ff_dim", type=int, required=True)
    parser.add_argument("--embed_dim", type=int, required=True)
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--num_heads", type=int, required=True)
    parser.add_argument("--mlp_units", type=int, required=True)

    return parser.parse_args()

def project_attention_to_2theta(attn_map, spec_length, two_theta_range):
    """
    Project attention weights back to 2θ domain.
    Args:
        attn_map: torch.Tensor (num_heads, seq_len, seq_len)
        spec_length: int, length of spectrum (e.g. 1000)
        two_theta_range: tuple (min_2θ, max_2θ)
    Returns:
        two_theta: np.array of shape (spec_length,)
        attn_profile: np.array of shape (spec_length,)
    """
    # Average attention across heads and across the sequence dimension
    # Take diagonal or mean across attention matrix
    attn_avg = attn_map.mean(0)  # Average across heads: (seq_len, seq_len)
    
    # Option 1: Take diagonal (self-attention weights)
    attn_profile = torch.diag(attn_avg)  # (seq_len,)
    
    # Option 2: Take mean across the attention dimension
    # attn_profile = attn_avg.mean(1)  # (seq_len,)
    
    attn_profile = attn_profile / attn_profile.sum()  # Normalize
    
    # Convert to numpy
    attn_profile = attn_profile.cpu().numpy()
    
    # If the attention profile length doesn't match spec_length, interpolate
    if len(attn_profile) != spec_length:
        attn_profile = np.interp(
            np.arange(spec_length),
            np.linspace(0, spec_length-1, len(attn_profile)),
            attn_profile
        )
    
    # 2θ mapping
    two_theta = np.linspace(two_theta_range[0], two_theta_range[1], spec_length)
    return two_theta, attn_profile

def plot_attention_overlay(xrd, attn_map, two_theta, sample_id, save_dir="./", num_bins=50, y_max=1000):
    """
    Overlay attention map as red-gradient bins along full Y-axis.
    - xrd: 1D numpy array of intensity
    - attn_map: 1D numpy array aligned with two_theta
    - two_theta: 1D numpy array of 2θ values
    - sample_id: str/int label for saving
    - save_dir: directory to save figure
    - num_bins: number of bins for color gradient
    - y_max: maximum Y value for overlay (full spectrum range)
    """
    import os
    import matplotlib.pyplot as plt
    import numpy as np

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"attention_overlay_{sample_id}.png")

    fig, ax = plt.subplots(figsize=(12, 5))

    # Plot XRD spectrum
    ax.plot(two_theta, xrd, color="black", lw=1)
    ax.set_xlabel("2θ")
    ax.set_ylabel("Intensity")
    ax.set_ylim(0, y_max)

    # Normalize attention to [0, 1]
    attn_norm = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

    # Bin the spectrum for gradient overlay
    bin_edges = np.linspace(two_theta[0], two_theta[-1], num_bins + 1)
    bin_indices = np.digitize(two_theta, bin_edges) - 1  # which bin each point belongs to

    # Plot bins with red gradient along full Y-axis
    for i in range(num_bins):
        mask = bin_indices == i
        if np.any(mask):
            ax.fill_between(two_theta[mask], 0, y_max,
                            color=(1.0, 0, 0, attn_norm[mask].mean()),  # RGBA
                            step='mid')

    plt.title(f"Sample {sample_id} Attention Map")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    return save_path


def main(args):
    # --- Init W&B run ---
    run = wandb.init(project=args.project, id=args.run_id)

    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load model ---
    model = transformer_model(
        spec_length=args.spec_length,
        num_output=args.num_classes,
        embed_dim=args.embed_dim,
        ff_dim=args.ff_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_units=args.mlp_units,
        dropout=0.0,
        pos_encoding="rope"
    )
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Strip "module." prefix
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        new_state_dict[k.replace("module.", "")] = v

    # Load weights
    model.load_state_dict(new_state_dict, strict=False)

    # Send to device
    model = model.to(device).eval()
    print(f"DEBUG: spec_length={args.spec_length}")
    print(f"DEBUG: num_classes={args.num_classes}")
    print(f"DEBUG: embed_dim={args.embed_dim}")
    print(f"DEBUG: ff_dim={args.ff_dim}")
    print(f"DEBUG: depth={args.depth}")
    print(f"DEBUG: num_heads={args.num_heads}")
    print(f"DEBUG: mlp_units={args.mlp_units}")
    # --- Load data ---
    train_loader, val_loader, test_loader, num_classes, intensity_points = get_dataloaders(
        args.data_path,
        batch_size=1,
        num_classes=args.num_classes
    )

    print(f"[DEBUG] num_classes={num_classes}, intensity_points={intensity_points}")
    dataloader = val_loader
    spectrum, label = next(iter(dataloader))
    
    with torch.no_grad():
        spectrum = spectrum.to(device)
        outputs = model(spectrum)
        last_layer_attention = model.transformer_blocks[-1].attention.last_attn_weights
        last_layer_attention = last_layer_attention.squeeze(0)  # Now (num_heads, seq_len, seq_len)

    # --- Project attention to 2θ ---
    two_theta, attn_profile = project_attention_to_2theta(
        last_layer_attention,  # last layer attention
        spec_length=args.spec_length,
        two_theta_range=(args.two_theta_min, args.two_theta_max)
    )

    # --- Plot and log ---
    save_path = plot_attention_overlay(
        xrd=spectrum.cpu().squeeze(0).numpy(),
        attn_map=attn_profile,
        two_theta=two_theta,
        sample_id=int(label.item()),  # convert label tensor to int/string
        save_dir="./"  # optional, defaults to current dir
    )
    wandb.log({"attention_overlay": wandb.Image(save_path)})

    print("✅ Logged attention overlay to W&B")

if __name__ == "__main__":
    args = parse_args()
    main(args)