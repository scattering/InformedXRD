#!/bin/bash
#SBATCH -J vista-rruff-200k-po
#SBATCH -A CDA24014
#SBATCH -p gh
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 04:00:00
#SBATCH -o /scratch/ALLOC3/UserE/vista_rruff_conditioned_200k_po_%j.out
#SBATCH -e /scratch/ALLOC3/UserE/vista_rruff_conditioned_200k_po_%j.err

set -euo pipefail

module load gcc/13.2.0 cuda/12.5 python3/3.11.8
source /scratch/ALLOC3/UserE/project-venv/bin/activate

export WANDB_API_KEY
WANDB_API_KEY="$(cat /scratch/ALLOC3/UserE/project/.wandb_api_key)"
DATA_PATH="/scratch/ALLOC3/UserE/ai_diffraction_generated/rruff_conditioned_200k_po_v1_trainready.hdf5"
EXPECTED_BYTES=5889557080

cd /scratch/ALLOC3/UserE/project/Code/ViT_NVIDIA
while true; do
  if [[ -f "$DATA_PATH" ]]; then
    CURRENT_BYTES="$(stat -c%s "$DATA_PATH")"
    echo "[INFO] data_path=$DATA_PATH current_bytes=$CURRENT_BYTES expected_bytes=$EXPECTED_BYTES"
    if [[ "$CURRENT_BYTES" -eq "$EXPECTED_BYTES" ]]; then
      break
    fi
  else
    echo "[INFO] waiting for dataset at $DATA_PATH"
  fi
  sleep 30
done

python - <<'PY'
import os
import wandb
wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)
PY
python train.py --config config_rruff_conditioned_200k_po_from_ic6gfmvm_physpe_coord.json
