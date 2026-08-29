import h5py
import numpy as np

INPUT_FILE  = "/scratch/10486/UserB/Data/10Million_pyxtal_RRUFFF-type_repacked.hdf5"
OUTPUT_FILE = "/scratch/10486/UserB/Data/10Million_pyxtal_RRUFFF-type_repacked_sampled_01.hdf5"
X_DATASETS  = ["X_train", "X_val", "X_test"]
Y_DATASETS  = ["y_train", "y_val", "y_test"]

SAMPLING_RATE = 0.01
CHUNK_SIZE    = 128

# -------------------------------
# Scaling options: "none", "log", "zscore", "l2", "max"
# -------------------------------
SCALING = "none"

# -------------------------------
# Uniform sampling (SAFE)
# -------------------------------
UNIFORM_SAMPLE = True        # <<< toggle here
N_UNIFORM_POINTS = 1000      # <<< target length

np.random.seed(42)

# -------------------------------
# Helpers
# -------------------------------

def uniform_indices(n_points, n_target):
    """Generate monotonic, non-duplicating uniform indices"""
    step = n_points / n_target
    idx = (step * np.arange(n_target)).astype(int)
    idx[idx >= n_points] = n_points - 1
    return idx


def scale_chunk(chunk, mode="none"):
    """Per-spectrum scaling"""
    if mode == "log":
        return np.log1p(chunk)

    elif mode == "zscore":
        mean = chunk.mean(axis=1, keepdims=True)
        std  = chunk.std(axis=1, keepdims=True)
        std[std == 0] = 1.0
        return (chunk - mean) / std

    elif mode == "l2":
        norm = np.linalg.norm(chunk, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        return chunk / norm

    elif mode == "max":
        max_vals = np.max(chunk, axis=1, keepdims=True)
        max_vals[max_vals == 0] = 1.0
        return chunk / max_vals

    return chunk  # "none"


# ===============================
# Main processing
# ===============================

with h5py.File(INPUT_FILE, "r") as f_in, h5py.File(OUTPUT_FILE, "w") as f_out:
    for x_name, y_name in zip(X_DATASETS, Y_DATASETS):
        if x_name not in f_in or y_name not in f_in:
            print(f"[WARN] {x_name} or {y_name} not found, skipping")
            continue

        X_in = f_in[x_name]
        y_in = f_in[y_name]

        n_samples, n_points = X_in.shape
        n_sample = max(1, int(n_samples * SAMPLING_RATE))

        # random sample indices (sorted for sequential reads)
        indices = np.sort(
            np.random.choice(n_samples, size=n_sample, replace=False)
        )

        # uniform sampling setup
        if UNIFORM_SAMPLE:
            idx_uniform = uniform_indices(n_points, N_UNIFORM_POINTS)
            out_points = N_UNIFORM_POINTS
        else:
            idx_uniform = None
            out_points = n_points

        # output datasets
        X_out = f_out.create_dataset(
            x_name,
            shape=(n_sample, out_points),
            dtype=np.float32,
            chunks=(min(CHUNK_SIZE, n_sample), out_points)
        )
        y_out = f_out.create_dataset(
            y_name,
            shape=(n_sample,),
            dtype=y_in.dtype
        )

        # copy attributes
        for k, v in X_in.attrs.items():
            X_out.attrs[k] = v
        for k, v in y_in.attrs.items():
            y_out.attrs[k] = v

        written = 0
        for i in range(0, n_sample, CHUNK_SIZE):
            batch_idx = indices[i:i + CHUNK_SIZE]

            # load chunk
            X_chunk = np.empty((len(batch_idx), n_points), dtype=np.float32)
            for j, idx in enumerate(batch_idx):
                X_chunk[j] = X_in[idx]

            # clean NaNs / infs
            X_chunk = np.nan_to_num(
                X_chunk, nan=0.0, posinf=0.0, neginf=0.0
            )

            # uniform sampling (SAFE)
            if UNIFORM_SAMPLE:
                X_chunk = X_chunk[:, idx_uniform]

            # scaling
            X_chunk = scale_chunk(X_chunk, SCALING)

            # write
            X_out[written:written + len(batch_idx)] = X_chunk
            y_out[written:written + len(batch_idx)] = y_in[batch_idx]
            written += len(batch_idx)

        # summary
        min_val = np.inf
        max_val = -np.inf
        for i in range(0, n_sample, CHUNK_SIZE):
            chunk = X_out[i:i + CHUNK_SIZE]
            min_val = min(min_val, chunk.min())
            max_val = max(max_val, chunk.max())

        print(f"✔ {x_name} / {y_name}: {n_sample}/{n_samples} samples saved")
        print(f"   Shape X: {X_out.shape}, Shape y: {y_out.shape}")
        print(f"   Min/Max X: {min_val:.6f} / {max_val:.6f}\n")

    # global metadata
    f_out.attrs["scaling"] = SCALING
    f_out.attrs["sampling_rate"] = SAMPLING_RATE
    f_out.attrs["uniform_sampling"] = UNIFORM_SAMPLE
    f_out.attrs["n_uniform_points"] = (
        N_UNIFORM_POINTS if UNIFORM_SAMPLE else n_points
    )
    f_out.attrs["note"] = (
        f"Scaling={SCALING}, "
        f"{SAMPLING_RATE*100:.1f}% sampling, "
        f"{'uniform downsampled' if UNIFORM_SAMPLE else 'full resolution'}"
    )

print(f"\n✅ Sampled datasets written to: {OUTPUT_FILE}")
