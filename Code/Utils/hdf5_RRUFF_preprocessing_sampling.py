import h5py
import numpy as np

# ============================
# Configuration
# ============================

INPUT_FILE = "/content/drive/MyDrive/Research/OrgA/OrgA Team Shared Folder/Data - HDF5/RRUFF_raw_full.hdf5"
OUTPUT_FILE = (
    "/content/drive/MyDrive/Research/OrgA/OrgA Team Shared Folder/Data - HDF5/"
    "RRUFF_low_bkg_uniform_sampled_cleaned.hdf5"
)

TARGET_LENGTH = 8501
INTENSITY_MAX = 1000

UNIFORM_SAMPLE = True       # 🔵 toggle here
N_UNIFORM_POINTS = 1000     # 🔵 downsample length

# ============================
# Helper: uniform sampling
# ============================

def uniform_sample(intensity, n_points):
    """
    Uniformly downsample a 1D spectrum without duplicate indices.
    Assumes fixed 2θ grid across all samples.
    """
    N = len(intensity)
    step = N / n_points
    idx = (step * np.arange(n_points)).astype(int)
    idx[idx >= N] = N - 1
    return intensity[idx]

# ============================
# Load + filter + sample
# ============================

X_list = []
y_list = []

kept = 0
dropped = 0

with h5py.File(INPUT_FILE, "r") as f:
    for key in f.keys():
        group = f[key]
        if not isinstance(group, h5py.Group):
            continue

        # must have intensity
        if "intensity" not in group:
            dropped += 1
            continue

        intensity = group["intensity"][:]

        # exact length filter
        if len(intensity) != TARGET_LENGTH:
            dropped += 1
            continue

        # intensity sanity check
        if np.any(intensity > INTENSITY_MAX):
            dropped += 1
            continue

        # optional uniform sampling
        if UNIFORM_SAMPLE:
            intensity = uniform_sample(intensity, N_UNIFORM_POINTS)

        # label extraction
        if "eg" in group.attrs:
            y_value = group.attrs["eg"]
        else:
            attrs = list(group.attrs.items())
            if len(attrs) == 1:
                y_value = attrs[0][1]
            else:
                dropped += 1
                continue

        X_list.append(intensity.astype(np.float32))
        y_list.append(y_value)
        kept += 1

# ============================
# Stack + Save
# ============================

X_test = np.stack(X_list, axis=0)
y_test = np.array(y_list)

print(f"Kept spectra   : {kept}")
print(f"Dropped spectra: {dropped}")
print(f"Final shape    : X_test={X_test.shape}, y_test={y_test.shape}")

with h5py.File(OUTPUT_FILE, "w") as f_out:
    f_out.create_dataset("X_test", data=X_test, compression="gzip")
    f_out.create_dataset("y_test", data=y_test)

    # metadata
    f_out.attrs["uniform_sampling"] = UNIFORM_SAMPLE
    f_out.attrs["n_uniform_points"] = (
        N_UNIFORM_POINTS if UNIFORM_SAMPLE else TARGET_LENGTH
    )
    f_out.attrs["intensity_max_threshold"] = INTENSITY_MAX
    f_out.attrs["note"] = (
        "RRUFF spectra filtered by length and intensity threshold. "
        "Uniform sampling applied after filtering."
    )

print(f"\n✅ Saved cleaned dataset to:\n{OUTPUT_FILE}")
