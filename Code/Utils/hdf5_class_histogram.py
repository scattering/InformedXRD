"""
Class distribution histograms for HDF5 datasets.

- CPU-only
- Reads labels only (y_*)
- Outputs class counts and percentages per split
"""

import h5py
import numpy as np

# -----------------------------
# Configuration
# -----------------------------
INPUT_FILE = "data_maxnorm_sampled.h5"
Y_DATASETS = ["y_train", "y_val", "y_test"]
OUTPUT_TXT = "class_distribution.txt"

# -----------------------------
# Helpers
# -----------------------------
def compute_class_histogram(y):
    """
    Returns:
        classes: np.ndarray
        counts: np.ndarray
        percentages: np.ndarray
    """
    classes, counts = np.unique(y, return_counts=True)
    percentages = counts / counts.sum() * 100.0
    return classes, counts, percentages


def write_histogram(fh, split_name, classes, counts, percentages):
    fh.write("\n" + "=" * 60 + "\n")
    fh.write(f"Split: {split_name}\n")
    fh.write(f"Total samples: {counts.sum()}\n\n")
    fh.write(f"{'Class':>10} | {'Count':>10} | {'Percent':>8}\n")
    fh.write("-" * 40 + "\n")

    for c, n, p in zip(classes, counts, percentages):
        fh.write(f"{int(c):>10} | {int(n):>10} | {p:7.2f}%\n")


# -----------------------------
# Main
# -----------------------------
def main():
    with h5py.File(INPUT_FILE, "r") as f, open(OUTPUT_TXT, "w") as fh:
        fh.write("HDF5 Class Distribution Summary\n")
        fh.write(f"Input file: {INPUT_FILE}\n")

        for y_name in Y_DATASETS:
            if y_name not in f:
                print(f"[WARN] {y_name} not found, skipping")
                continue

            y = f[y_name][:]
            classes, counts, percentages = compute_class_histogram(y)
            write_histogram(fh, y_name, classes, counts, percentages)

            print(f"✔ Class histogram written for {y_name}")

    print(f"\n✅ Class distribution saved to: {OUTPUT_TXT}")


if __name__ == "__main__":
    main()
