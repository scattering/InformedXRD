#!/bin/bash
#SBATCH -J vista-u-v7-raw
#SBATCH -A CDA24014
#SBATCH -p gh
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH -t 03:00:00
#SBATCH -o /scratch/ALLOC3/UserE/vista_uniform_v7_raw_%j.out
#SBATCH -e /scratch/ALLOC3/UserE/vista_uniform_v7_raw_%j.err

set -euo pipefail

module load gcc/13.2.0 cuda/12.5 python3/3.11.8
source /scratch/ALLOC3/UserE/project-venv/bin/activate
cd /scratch/ALLOC3/UserE/project/Code/ViT_NVIDIA

export WANDB_API_KEY
WANDB_API_KEY="$(cat /scratch/ALLOC3/UserE/project/.wandb_api_key)"

python train.py --config config_uniform_merged_v7_rawpos_physpe_coord.json
