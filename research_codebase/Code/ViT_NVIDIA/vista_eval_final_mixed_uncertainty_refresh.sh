#!/bin/bash
#SBATCH -J unc-refresh-655279
#SBATCH -A CDA24014
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -t 02:00:00
#SBATCH -o /scratch/ALLOC3/UserE/uncertainty_refresh_%j.out
#SBATCH -e /scratch/ALLOC3/UserE/uncertainty_refresh_%j.err

set -euo pipefail

module load gcc/13.2.0 cuda/12.5 python3/3.11.8
source /scratch/ALLOC3/UserE/project-venv/bin/activate
cd /scratch/ALLOC3/UserE/project/Code/ViT_NVIDIA

CKPT=/scratch/ALLOC3/UserE/ai_diffraction_models/xrd_model_82ept35h_best.pth
CFG=/scratch/ALLOC3/UserE/project/Code/ViT_NVIDIA/config_rruff_conditioned_dualsource_2346k_500kpo_from_ic6gfmvm_physpe_coord.json
PRIOR=/scratch/ALLOC3/UserE/ai_diffraction_generated/rruff_conditioned_500k_po_v1_trainready.hdf5
BENCH325=/work2/ALLOC3/UserE/rruff-benchmark/RRUFF_usable_plus_recoverable_325_with_labels_maxnorm.hdf5
BENCH473=/work2/ALLOC3/UserE/rruff-benchmark/RRUFF_option1_473_with_buckets_maxnorm.hdf5

python evaluate_split_head_validity.py \
  --checkpoint "$CKPT" \
  --config "$CFG" \
  --eval-data-path "$BENCH325" \
  --bootstrap 2000 \
  --output-json /scratch/ALLOC3/UserE/mixed2500k_split_validity_325_655279_bootstrap.json

python evaluate_split_head_validity.py \
  --checkpoint "$CKPT" \
  --config "$CFG" \
  --eval-data-path "$BENCH473" \
  --bootstrap 2000 \
  --output-json /scratch/ALLOC3/UserE/mixed2500k_split_validity_473_655279_bootstrap.json

python analyze_catastrophic_paradox_control.py \
  --checkpoint "$CKPT" \
  --config "$CFG" \
  --eval-data-path "$BENCH473" \
  --prior-data-path "$PRIOR" \
  --temperature 5.0 \
  --bootstrap 2000 \
  --output-json /scratch/ALLOC3/UserE/catastrophic_paradox_control_655279_bootstrap.json
