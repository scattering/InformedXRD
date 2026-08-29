#!/bin/bash
#SBATCH -J vista-u2m-sg230-ref
#SBATCH -A CDA24014
#SBATCH -p gh
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH -t 06:00:00
#SBATCH -o /scratch/ALLOC3/UserE/vista_uniform_2m_sg230_ref_%j.out
#SBATCH -e /scratch/ALLOC3/UserE/vista_uniform_2m_sg230_ref_%j.err

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

# Referee rerun: match the 2-epoch uniform-2M recipe and start from the same
# pretrained backbone lineage used by the direct-EG comparison. This keeps the
# target-label change as the main experimental difference.
python train.py --config config_uniform_2m_blackbird_sg230_physpe_coord_referee_rerun.json
