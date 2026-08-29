#!/bin/bash
#SBATCH -J eval-mixed200k-r325473
#SBATCH -A CDA24014
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 02:00:00
#SBATCH -o /scratch/ALLOC3/UserE/eval-mixed200k-r325473.%j.out
#SBATCH -e /scratch/ALLOC3/UserE/eval-mixed200k-r325473.%j.err

set -euo pipefail
module purge
module load gcc/13.2.0 cuda/12.5 python3/3.11.8
source /scratch/ALLOC3/UserE/project-venv/bin/activate
cd /scratch/ALLOC3/UserE/project/Code/ViT_NVIDIA

ckpt=/scratch/ALLOC3/UserE/ai_diffraction_models/xrd_model_eeru8svx_best.pth
cfg=/scratch/ALLOC3/UserE/project/Code/ViT_NVIDIA/config_rruff_conditioned_mixed_200k_from_ic6gfmvm_physpe_coord.json
prior=/scratch/ALLOC3/UserE/ai_diffraction_generated/rruff_conditioned_mixed_200k_v1_trainready.hdf5

python evaluate_calibration_metrics.py --checkpoint "$ckpt" --config "$cfg" --eval-data-path /work2/ALLOC3/UserE/rruff-benchmark/RRUFF_usable_plus_recoverable_325_with_labels_maxnorm.hdf5 --prior-data-path "$prior" --aux-temperature 5.0 --bootstrap 1000 --output-json /scratch/ALLOC3/UserE/mixed200k_calibration_metrics_325_eeru8svx.json

python evaluate_calibration_metrics.py --checkpoint "$ckpt" --config "$cfg" --eval-data-path /work2/ALLOC3/UserE/rruff-benchmark/RRUFF_option1_473_with_buckets_maxnorm.hdf5 --prior-data-path "$prior" --aux-temperature 5.0 --bootstrap 1000 --output-json /scratch/ALLOC3/UserE/mixed200k_calibration_metrics_473_eeru8svx.json

python evaluate_split_head_validity.py --checkpoint "$ckpt" --config "$cfg" --eval-data-path /work2/ALLOC3/UserE/rruff-benchmark/RRUFF_usable_plus_recoverable_325_with_labels_maxnorm.hdf5 --output-json /scratch/ALLOC3/UserE/mixed200k_split_validity_325_eeru8svx.json

python evaluate_split_head_validity.py --checkpoint "$ckpt" --config "$cfg" --eval-data-path /work2/ALLOC3/UserE/rruff-benchmark/RRUFF_option1_473_with_buckets_maxnorm.hdf5 --output-json /scratch/ALLOC3/UserE/mixed200k_split_validity_473_eeru8svx.json
