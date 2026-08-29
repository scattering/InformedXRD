#!/bin/bash
#SBATCH -J vista-r-v7-ft
#SBATCH -A CDA24014
#SBATCH -p gh
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH -t 03:00:00
#SBATCH -o /scratch/ALLOC3/UserE/vista_rruff_v7_ft_%j.out
#SBATCH -e /scratch/ALLOC3/UserE/vista_rruff_v7_ft_%j.err

set -euo pipefail

module load gcc/13.2.0 cuda/12.5 python3/3.11.8
source /scratch/ALLOC3/UserE/project-venv/bin/activate
cd /scratch/ALLOC3/UserE/project/Code/ViT_NVIDIA

export WANDB_API_KEY
WANDB_API_KEY="$(cat /scratch/ALLOC3/UserE/project/.wandb_api_key)"

python train.py --config config_rruff_v7_finetune_physpe_coord.json
