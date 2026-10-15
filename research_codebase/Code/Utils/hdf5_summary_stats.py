import h5py
import numpy as np

FILE_PATH = "data.h5"          # <-- change this
OUTPUT_TXT = "summary_stats.txt"
DATASETS = ["X_train", "X_val", "X_test"]
CHUNK_SIZE = 1024


def summarize_dataset(dset):
    n_samples = dset.shape[0]

    total_sum = 0.0
    total_sq_sum = 0.0
    total_count = 0
    min_val = np.inf
    max_val = -np.inf
    nan_count = 0
    inf_count = 0

    for i in range(0, n_samples, CHUNK_SIZE):
        chunk = dset[i:i + CHUNK_SIZE]

        nan_count += np.isnan(chunk).sum()
        inf_count += np.isinf(chunk).sum()

        chunk = np.nan_to_num(chunk, nan=0.0, posinf=0.0, neginf=0.0)

        total_sum += chunk.sum()
        total_sq_sum += np.square(chunk).sum()
        total_count += chunk.size

        min_val = min(min_val, chunk.min())
        max_val = max(max_val, chunk.max())

    mean = total_sum / total_count
    var = (total_sq_sum / total_count) - mean ** 2
    std = np.sqrt(var)

    return {
        "shape": dset.shape,
        "dtype": dset.dtype,
        "min": min_val,
        "max": max_val,
        "mean": mean,
        "std": std,
        "nan_count": int(nan_count),
        "inf_count": int(inf_count),
    }


def main():
    with h5py.File(FILE_PATH, "r") as f, open(OUTPUT_TXT, "w") as out:
        for name in DATASETS:
            out.write(f"\n===== {name} =====\n")

            if name not in f:
                out.write("[WARN] Dataset not found\n")
                continue

            stats = summarize_dataset(f[name])

            for k, v in stats.items():
                out.write(f"{k:12}: {v}\n")

    print(f"Summary statistics written to {OUTPUT_TXT}")


if __name__ == "__main__":
    main()
