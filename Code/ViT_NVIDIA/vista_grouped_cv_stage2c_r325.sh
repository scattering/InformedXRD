#!/bin/bash
#SBATCH -J vista-cv-stage2c
#SBATCH -A CDA24014
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 01:00:00
#SBATCH -o /scratch/ALLOC3/UserE/vista_grouped_cv_stage2c_%j.out
#SBATCH -e /scratch/ALLOC3/UserE/vista_grouped_cv_stage2c_%j.err

set -euo pipefail

module load gcc/13.2.0 cuda/12.5 python3/3.11.8
source /scratch/ALLOC3/UserE/project-venv/bin/activate

cd /scratch/ALLOC3/UserE/project/Code/ViT_NVIDIA

ckpt=/scratch/ALLOC3/UserE/ai_diffraction_models/xrd_model_9rwv1qly_best.pth
cfg=/scratch/ALLOC3/UserE/project/Code/ViT_NVIDIA/config_rruff_conditioned_2346k_from_ic6gfmvm_physpe_coord.json
prior=/scratch/ALLOC3/UserE/ai_diffraction_generated/rruff_conditioned_2346k_v1_trainready.hdf5
eval_h5=/work2/ALLOC3/UserE/rruff-benchmark/RRUFF_usable_plus_recoverable_325_with_labels_maxnorm.hdf5

python evaluate_grouped_temperature_cv.py \
  --checkpoint "$ckpt" \
  --config "$cfg" \
  --eval-data-path "$eval_h5" \
  --prior-data-path "$prior" \
  --group-key minerals \
  --n-folds 5 \
  --selection-metric nll \
  --temperature-grid \
    1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0 5.5 6.0 6.5 7.0 7.5 8.0 8.5 9.0 9.5 \
    10.0 10.5 11.0 11.5 12.0 12.5 13.0 13.5 14.0 14.5 15.0 15.5 16.0 16.5 17.0 17.5 \
    18.0 18.5 19.0 19.5 20.0 22.5 25.0 27.5 30.0 35.0 40.0 45.0 50.0 75.0 100.0 150.0 200.0 \
  --output-json /scratch/ALLOC3/UserE/grouped_cv_stage2c_r325_by_mineral.json
