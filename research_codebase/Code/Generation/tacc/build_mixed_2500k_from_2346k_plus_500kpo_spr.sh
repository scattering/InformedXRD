#!/bin/bash
#SBATCH -J mix-2500k-spr
#SBATCH -A CDA24014
#SBATCH -p spr
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 06:00:00
#SBATCH -o /scratch/ALLOC3/UserE/mix-2500k-spr.%j.out
#SBATCH -e /scratch/ALLOC3/UserE/mix-2500k-spr.%j.err

set -euo pipefail

module purge
module load python/3.12.11

source /scratch/ALLOC3/UserE/pycrysfml_spr/venv/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE

OUT="/scratch/ALLOC3/UserE/ai_diffraction_generated/rruff_conditioned_mixed_2500k_v1_trainready.hdf5"
TMP="${OUT}.tmp"

set +e
python - <<'PY'
import os
import sys
import h5py

path = "/scratch/ALLOC3/UserE/ai_diffraction_generated/rruff_conditioned_mixed_2500k_v1_trainready.hdf5"
expected = {
    "X_train": (2000000, 8501),
    "X_val": (250000, 8501),
    "X_test": (250000, 8501),
}

if not os.path.exists(path):
    raise SystemExit(1)

try:
    with h5py.File(path, "r") as h:
        ok = all(tuple(h[k].shape) == expected[k] for k in expected)
except Exception:
    raise SystemExit(1)

print("[INFO] existing mixed_2500k trainready already complete; skipping rebuild")
raise SystemExit(0)
PY
status=$?
set -e
if [[ $status -eq 0 ]]; then
  exit 0
fi

rm -f "$TMP"

python /scratch/ALLOC3/UserE/project/Code/Generation/tacc/build_mixed_trainready_blocks.py \
  --standard /work2/ALLOC3/UserE/rruff_conditioned_2346k_v1_trainready.hdf5 \
  --po /scratch/ALLOC3/UserE/ai_diffraction_generated/rruff_conditioned_500k_po_v1_trainready.hdf5 \
  --output "$OUT" \
  --std-train 1600000 \
  --po-train 400000 \
  --std-val 200000 \
  --po-val 50000 \
  --std-test 200000 \
  --po-test 50000 \
  --chunk-rows 256 \
  --seed 1337
