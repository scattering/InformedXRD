#!/bin/bash
#SBATCH -J stage2c-subset
#SBATCH -A CDA24014
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 01:00:00
#SBATCH -o /scratch/ALLOC3/UserE/stage2c_subset_%j.out
#SBATCH -e /scratch/ALLOC3/UserE/stage2c_subset_%j.err

set -euo pipefail

module load gcc/13.2.0 cuda/12.5 python3/3.11.8
source /scratch/ALLOC3/UserE/project-venv/bin/activate

cd /scratch/ALLOC3/UserE/project/Code/ViT_NVIDIA

CKPT=${CKPT:-/scratch/ALLOC3/UserE/ai_diffraction_models/xrd_model_9rwv1qly_best.pth}
CFG=${CFG:-/scratch/ALLOC3/UserE/project/Code/ViT_NVIDIA/config_rruff_conditioned_2346k_from_ic6gfmvm_physpe_coord.json}
PRIOR=${PRIOR:-/scratch/ALLOC3/UserE/ai_diffraction_generated/rruff_conditioned_2346k_v1_trainready.hdf5}
EVAL_H5=${EVAL_H5:-/work2/ALLOC3/UserE/rruff-benchmark/RRUFF_option1_473_with_buckets_maxnorm.hdf5}
OUT_JSON=${OUT_JSON:-/scratch/ALLOC3/UserE/stage2c_subset_metrics.json}
AUX_T=${AUX_T:-5.0}

python evaluate_calibration_metrics.py \
  --checkpoint "$CKPT" \
  --config "$CFG" \
  --eval-data-path "$EVAL_H5" \
  --prior-data-path "$PRIOR" \
  --aux-temperature "$AUX_T" \
  --bootstrap 2000 \
  --output-json "$OUT_JSON"
