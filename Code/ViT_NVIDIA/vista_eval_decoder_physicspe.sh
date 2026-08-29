#!/bin/bash
#SBATCH -J vista-decode-physpe
#SBATCH -A CDA24014
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH -t 00:30:00
#SBATCH -o /scratch/ALLOC3/UserE/vista_decode_physicspe_%j.out
#SBATCH -e /scratch/ALLOC3/UserE/vista_decode_physicspe_%j.err

set -euo pipefail

module load gcc/13.2.0 cuda/12.5 python3/3.11.8
source /scratch/ALLOC3/UserE/project-venv/bin/activate
cd /scratch/ALLOC3/UserE/project/Code/ViT_NVIDIA

python -u evaluate_decoder_variants.py \
  --checkpoint /scratch/ALLOC3/UserE/ai_diffraction_models/xrd_model_yybkvba9_best.pth \
  --config /scratch/ALLOC3/UserE/project/Code/ViT_NVIDIA/config_uniform_partial_vista_pilot_stockyard_physicspe.json \
  --prior-data-path /work/ALLOC3/UserE/stampede3/ai_diffraction_generated/uniform2m_v1_partial_trainready_ordered_v2.hdf5 \
  > /scratch/ALLOC3/UserE/decode_physpe_synth.txt

python -u evaluate_decoder_variants.py \
  --checkpoint /scratch/ALLOC3/UserE/ai_diffraction_models/xrd_model_yybkvba9_best.pth \
  --config /scratch/ALLOC3/UserE/project/Code/ViT_NVIDIA/config_uniform_partial_vista_pilot_stockyard_physicspe.json \
  --eval-data-path /work2/ALLOC2/UserC/shared_dir/RRUFFTesting/RRUFF_low_bkg_full_intensity_cleaned_maxnorm.hdf5 \
  --prior-data-path /work/ALLOC3/UserE/stampede3/ai_diffraction_generated/uniform2m_v1_partial_trainready_ordered_v2.hdf5 \
  > /scratch/ALLOC3/UserE/decode_physpe_real.txt
