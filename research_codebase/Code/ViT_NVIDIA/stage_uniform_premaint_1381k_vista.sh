#!/bin/bash
#SBATCH -J stage-u-1381k
#SBATCH -A CDA24014
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -t 01:00:00
#SBATCH -o /scratch/ALLOC3/UserE/stage_uniform_1381k_%j.out
#SBATCH -e /scratch/ALLOC3/UserE/stage_uniform_1381k_%j.err

set -euo pipefail

module load gcc/13.2.0 python3/3.11.8
source /scratch/ALLOC3/UserE/repack_vista_gg_venv/bin/activate

src=/work2/ALLOC3/UserE/uniform_premaint_1381k_trainready.hdf5
dst=/scratch/ALLOC3/UserE/ai_diffraction_generated/uniform_premaint_1381k_trainready.hdf5
tmp="${dst}.tmp"

mkdir -p "$(dirname "$dst")"
rm -f "$tmp" "$dst"
cp "$src" "$tmp"

SRC_PATH="$src" TMP_PATH="$tmp" python - <<'PY'
import os
import h5py

src = os.environ["SRC_PATH"]
tmp = os.environ["TMP_PATH"]

if os.path.getsize(src) != os.path.getsize(tmp):
    raise SystemExit("staged file size mismatch")

with h5py.File(tmp, "r") as f:
    for key in ("X_train", "y_train", "X_val", "y_val", "X_test", "y_test"):
        if key not in f:
            raise SystemExit(f"missing dataset: {key}")
    print(
        "validated",
        int(f["X_train"].shape[0]),
        int(f["X_val"].shape[0]),
        int(f["X_test"].shape[0]),
    )
PY

mv "$tmp" "$dst"
