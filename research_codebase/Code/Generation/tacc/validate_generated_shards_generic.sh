#!/bin/bash
#SBATCH -J validate-shards
#SBATCH -A CDA24014
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 00:30:00

set -euo pipefail

VENV_PYTHON="${VENV_PYTHON:-/scratch/ALLOC3/UserE/pycrysfml_spr/venv/bin/python}"
OUT_DIR="${OUT_DIR:?OUT_DIR is required}"
NUM_SHARDS="${NUM_SHARDS:?NUM_SHARDS is required}"
LABEL_KEY="${LABEL_KEY:-extinction_group}"
LABEL_MIN="${LABEL_MIN:-1}"
LABEL_MAX="${LABEL_MAX:-99}"
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"

"$VENV_PYTHON" - <<'PY'
import glob
import os
import h5py

base = os.environ["OUT_DIR"]
expected_shards = int(os.environ["NUM_SHARDS"])
label_key = os.environ.get("LABEL_KEY", "extinction_group")
label_min = int(os.environ.get("LABEL_MIN", "1"))
label_max = int(os.environ.get("LABEL_MAX", "99"))

paths = sorted(glob.glob(base + "/*.hdf5"))
assert len(paths) == expected_shards, f"expected {expected_shards} shards, found {len(paths)}"

total = 0
for p in paths:
    with h5py.File(p, "r") as f:
        root = f["/"]
        expected = int(root.attrs["local_samples_target"])
        written = int(root.attrs["successes_written"])
        avail = int(f["intensity"].shape[0])
        assert written == expected == avail, (
            f"{os.path.basename(p)} incomplete: written={written} expected={expected} avail={avail}"
        )
        y = f[label_key][:]
        assert y.min() >= label_min and y.max() <= label_max, f"{os.path.basename(p)} bad label range for {label_key}"
        total += written

print("validated_shards", len(paths))
print("validated_rows", total)
PY
