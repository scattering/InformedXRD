#!/bin/bash
#SBATCH -J vista-rruff-2346k
#SBATCH -A CDA24014
#SBATCH -p gh
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 08:00:00
#SBATCH -o /scratch/ALLOC3/UserE/vista_rruff_conditioned_2346k_%j.out
#SBATCH -e /scratch/ALLOC3/UserE/vista_rruff_conditioned_2346k_%j.err

set -euo pipefail

module load gcc/13.2.0 cuda/12.5 python3/3.11.8
source /scratch/ALLOC3/UserE/project-venv/bin/activate

export WANDB_API_KEY
WANDB_API_KEY="$(cat /scratch/ALLOC3/UserE/project/.wandb_api_key)"

cd /scratch/ALLOC3/UserE/project/Code/ViT_NVIDIA
python - <<'PY'
import os
import wandb
wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)
PY
python train.py --config config_rruff_conditioned_2346k_from_ic6gfmvm_physpe_coord.json
