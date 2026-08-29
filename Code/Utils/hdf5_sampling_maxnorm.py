import h5py
import numpy as np

INPUT_FILE  = "/scratch/10486/UserB/Data/10Million_pyxtal_RRUFFF-type_repacked.hdf5"
OUTPUT_FILE = "/scratch/10486/UserB/Data/10Million_pyxtal_RRUFFF-type_repacked_sampled_01.hdf5"
X_DATASETS  = ["X_train", "X_val", "X_test"]
Y_DATASETS  = ["y_train", "y_val", "y_test"]
SAMPLING_RATE = 0.01   # fraction of samples to keep per dataset
CHUNK_SIZE    = 128    # for memory-efficient chunk processing
USE_MAXNORM   = False   # <<<< flag to toggle max-normalization

def max_normalize_chunk(chunk):
    """Per-spectrum max normalization"""
    max_vals = np.max(chunk, axis=1, keepdims=True)
    max_vals[max_vals == 0] = 1.0
    return chunk / max_vals

np.random.seed(42)  # reproducible sampling

with h5py.File(INPUT_FILE, "r") as f_in, h5py.File(OUTPUT_FILE, "w") as f_out:
    for x_name, y_name in zip(X_DATASETS, Y_DATASETS):
        if x_name not in f_in or y_name not in f_in:
            print(f"[WARN] {x_name} or {y_name} not found, skipping")
            continue

        X_in = f_in[x_name]
        y_in = f_in[y_name]
        n_samples, n_points = X_in.shape
        n_sample = max(1, int(n_samples * SAMPLING_RATE))

        # randomly select indices (sorted for sequential access)
        indices = np.sort(np.random.choice(n_samples, size=n_sample, replace=False))

        # create output datasets
        X_out = f_out.create_dataset(
            x_name,
            shape=(n_sample, n_points),
            dtype=np.float32,
            chunks=(min(CHUNK_SIZE, n_sample), n_points)
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

        # --- process X and y in chunks ---
        written = 0
        for i in range(0, n_sample, CHUNK_SIZE):
            batch_idx = indices[i:i + CHUNK_SIZE]

            # --- load X chunk ---
            X_chunk = np.empty((len(batch_idx), n_points), dtype=np.float32)
            for j, idx in enumerate(batch_idx):
                X_chunk[j] = X_in[idx]

            # clean NaNs / infs
            X_chunk = np.nan_to_num(X_chunk, nan=0.0, posinf=0.0, neginf=0.0)

            # apply max-normalization if flag is True
            if USE_MAXNORM:
                X_chunk = max_normalize_chunk(X_chunk)

            # write to output
            X_out[written:written + len(batch_idx)] = X_chunk
            y_out[written:written + len(batch_idx)] = y_in[batch_idx]
            written += len(batch_idx)

        # --- small summary ---
        min_val = np.inf
        max_val = -np.inf
        for i in range(0, n_sample, CHUNK_SIZE):
            chunk = X_out[i:i + CHUNK_SIZE]
            min_val = min(min_val, chunk.min())
            max_val = max(max_val, chunk.max())

        print(f"✔ {x_name} / {y_name}: {n_sample}/{n_samples} samples saved")
        print(f"   Shape X: {X_out.shape}, Shape y: {y_out.shape}")
        print(f"   Min/Max X: {min_val:.6f} / {max_val:.6f}\n")

    f_out.attrs["note"] = (
        f"{'Max-normalized' if USE_MAXNORM else 'Raw'} + "
        f"{SAMPLING_RATE*100:.1f}% sampling per dataset, chunked processing"
    )

print(f"\n✅ Sampled datasets written to: {OUTPUT_FILE}")
