#!/bin/bash
#SBATCH -J vista-decode-eval
#SBATCH -A CDA24014
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 00:30:00
#SBATCH -o /scratch/ALLOC3/UserE/vista_decode_eval_%j.out
#SBATCH -e /scratch/ALLOC3/UserE/vista_decode_eval_%j.err

set -euo pipefail

module load gcc/13.2.0 cuda/12.5 python3/3.11.8
source /scratch/ALLOC3/UserE/project-venv/bin/activate
cd /scratch/ALLOC3/UserE/project/Code/ViT_NVIDIA

python evaluate_decoder_variants.py \
  --checkpoint /scratch/ALLOC3/UserE/ai_diffraction_models/xrd_model_c6nuopyg_latest.pth \
  --config config_rruff_vista_large.json \
  --device cuda
