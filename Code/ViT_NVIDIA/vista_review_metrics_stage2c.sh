#!/bin/bash
#SBATCH -J vista-review-metrics
#SBATCH -A CDA24014
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 01:00:00
#SBATCH -o /scratch/ALLOC3/UserE/vista_review_metrics_%j.out
#SBATCH -e /scratch/ALLOC3/UserE/vista_review_metrics_%j.err

set -euo pipefail

module load gcc/13.2.0 cuda/12.5 python3/3.11.8
source /scratch/ALLOC3/UserE/project-venv/bin/activate

cd /scratch/ALLOC3/UserE/project/Code/ViT_NVIDIA

ckpt=/scratch/ALLOC3/UserE/ai_diffraction_models/xrd_model_9rwv1qly_best.pth
cfg=/scratch/ALLOC3/UserE/project/Code/ViT_NVIDIA/config_rruff_conditioned_2346k_from_ic6gfmvm_physpe_coord.json
prior=/scratch/ALLOC3/UserE/ai_diffraction_generated/rruff_conditioned_2346k_v1_trainready.hdf5

python evaluate_calibration_metrics.py \
  --checkpoint "$ckpt" \
  --config "$cfg" \
  --eval-data-path /work2/ALLOC3/UserE/rruff-benchmark/RRUFF_usable_plus_recoverable_325_with_labels_maxnorm.hdf5 \
  --prior-data-path "$prior" \
  --aux-temperature 5.0 \
  --bootstrap 2000 \
  --output-json /scratch/ALLOC3/UserE/review_calibration_metrics_325_stage2c.json

python evaluate_split_head_validity.py \
  --checkpoint "$ckpt" \
  --config "$cfg" \
  --eval-data-path /work2/ALLOC3/UserE/rruff-benchmark/RRUFF_usable_plus_recoverable_325_with_labels_maxnorm.hdf5 \
  --output-json /scratch/ALLOC3/UserE/review_split_validity_325_stage2c.json

python evaluate_split_head_validity.py \
  --checkpoint "$ckpt" \
  --config "$cfg" \
  --eval-data-path /work2/ALLOC3/UserE/rruff-benchmark/RRUFF_option1_473_with_buckets_maxnorm.hdf5 \
  --output-json /scratch/ALLOC3/UserE/review_split_validity_473_stage2c.json

python analyze_catastrophic_paradox_control.py \
  --checkpoint "$ckpt" \
  --config "$cfg" \
  --eval-data-path /work2/ALLOC3/UserE/rruff-benchmark/RRUFF_option1_473_with_buckets_maxnorm.hdf5 \
  --prior-data-path "$prior" \
  --temperature 5.0 \
  --bootstrap 2000 \
  --output-json /scratch/ALLOC3/UserE/review_catastrophic_control_473_stage2c.json
