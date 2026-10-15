#!/bin/bash
#SBATCH -J vista-mix-2500k-gh
#SBATCH -A CDA24014
#SBATCH -p gh
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 16:00:00
#SBATCH -o /scratch/ALLOC3/UserE/vista_rruff_mixed_2500k_%j.out
#SBATCH -e /scratch/ALLOC3/UserE/vista_rruff_mixed_2500k_%j.err

set -euo pipefail

module load gcc/13.2.0 cuda/12.5 python3/3.11.8
source /scratch/ALLOC3/UserE/project-venv/bin/activate

export WANDB_API_KEY
export WANDB_NAME="rruff-mixed-2500k-gh-${SLURM_JOB_ID}"
WANDB_API_KEY="$(cat /scratch/ALLOC3/UserE/project/.wandb_api_key)"
DATA_PATH="/scratch/ALLOC3/UserE/ai_diffraction_generated/rruff_conditioned_mixed_2500k_v1_trainready.hdf5"
REMOTE_DATA_PATH="/scratch/ALLOC3/UserE/ai_diffraction_generated/rruff_conditioned_mixed_2500k_v1_trainready.hdf5"
REMOTE_HOST="stampede3"
META_PATH="/scratch/ALLOC3/UserE/mixed2500k_train_${SLURM_JOB_ID}.json"
START_TS="$(date +%s)"
export META_PATH START_TS

cd /scratch/ALLOC3/UserE/project/Code/ViT_NVIDIA

while true; do
  if [[ -f "$DATA_PATH" ]]; then
    python - <<'PY'
import h5py
path = "/scratch/ALLOC3/UserE/ai_diffraction_generated/rruff_conditioned_mixed_2500k_v1_trainready.hdf5"
with h5py.File(path, "r") as h:
    ok = (
        h["X_train"].shape == (2000000, 8501)
        and h["X_val"].shape == (250000, 8501)
        and h["X_test"].shape == (250000, 8501)
    )
    print(f"[INFO] dataset_shapes train={h['X_train'].shape} val={h['X_val'].shape} test={h['X_test'].shape}")
    raise SystemExit(0 if ok else 1)
PY
    break
  fi
  if ssh "$REMOTE_HOST" "test -f '$REMOTE_DATA_PATH'"; then
    echo "[INFO] staging mixed dataset from ${REMOTE_HOST}:${REMOTE_DATA_PATH}"
    rsync -a --partial "$REMOTE_HOST:$REMOTE_DATA_PATH" "$DATA_PATH"
    continue
  fi
  echo "[INFO] waiting for mixed dataset at ${REMOTE_HOST}:${REMOTE_DATA_PATH}"
  sleep 120
done

python - <<'PY'
import os
import wandb
wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)
PY

python train.py --config config_rruff_conditioned_mixed_2500k_from_ic6gfmvm_physpe_coord.json

python - <<'PY'
import glob
import json
import os

model_dir = "/scratch/ALLOC3/UserE/ai_diffraction_models"
cfg_path = "/scratch/ALLOC3/UserE/project/Code/ViT_NVIDIA/config_rruff_conditioned_mixed_2500k_from_ic6gfmvm_physpe_coord.json"
prior_path = "/scratch/ALLOC3/UserE/ai_diffraction_generated/rruff_conditioned_mixed_2500k_v1_trainready.hdf5"
meta_path = os.environ["META_PATH"]
start_ts = int(os.environ["START_TS"])

candidates = []
for path in glob.glob(os.path.join(model_dir, "xrd_model_*_best.pth")):
    try:
        mtime = int(os.path.getmtime(path))
    except OSError:
        continue
    if mtime >= start_ts - 60:
        candidates.append((mtime, path))

if not candidates:
    raise SystemExit("No best checkpoint produced by mixed2500k training job")

candidates.sort()
best_path = candidates[-1][1]
payload = {
    "checkpoint": best_path,
    "config": cfg_path,
    "prior_data_path": prior_path,
    "train_job_id": os.environ["SLURM_JOB_ID"],
}
with open(meta_path, "w") as fh:
    json.dump(payload, fh, indent=2)
print(f"[INFO] wrote metadata to {meta_path}")
print(f"[INFO] best checkpoint {best_path}")
PY
