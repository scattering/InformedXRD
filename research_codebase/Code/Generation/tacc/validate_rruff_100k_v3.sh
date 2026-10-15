#!/bin/bash
#SBATCH -J val-rruff100k
#SBATCH -A CDA24014
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 00:30:00
#SBATCH -o /scratch/ALLOC3/UserE/val-rruff100k.%j.out
#SBATCH -e /scratch/ALLOC3/UserE/val-rruff100k.%j.err

set -euo pipefail

/scratch/ALLOC3/UserE/pycrysfml_spr/venv/bin/python - <<'PY'
import glob
import os

import h5py

base = "/scratch/ALLOC3/UserE/ai_diffraction_generated/rruff_conditioned_100k_v3"
paths = sorted(glob.glob(base + "/*.hdf5"))
assert len(paths) == 16, f"expected 16 shards, found {len(paths)}"

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
        y = f["extinction_group"][:]
        assert y.min() >= 1 and y.max() <= 99, f"{os.path.basename(p)} bad label range"
        total += written

print("validated_shards", len(paths))
print("validated_rows", total)
PY
