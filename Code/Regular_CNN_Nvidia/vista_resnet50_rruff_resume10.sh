#!/bin/bash
#SBATCH -J resnet50-rruff-r10
#SBATCH -A PHY26002
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 01:30:00
#SBATCH -o /scratch/ALLOC3/UserE/resnet50_rruff_resume10.%j.out
#SBATCH -e /scratch/ALLOC3/UserE/resnet50_rruff_resume10.%j.err

set -euo pipefail
module purge
module load gcc/13.2.0 cuda/12.5 python3/3.11.8
source /scratch/ALLOC3/UserE/project-venv/bin/activate
cd /scratch/ALLOC3/UserE/project/Code/Regular_CNN_Nvidia

python resnet50_rruff_train.py \
  --train-h5 /scratch/ALLOC3/UserE/ai_diffraction_generated/rruff_conditioned_2346k_v1_trainready.hdf5 \
  --eval-h5-473 /work2/ALLOC3/UserE/rruff-benchmark/RRUFF_option1_473_with_buckets_maxnorm.hdf5 \
  --eval-h5-325 /work2/ALLOC3/UserE/rruff-benchmark/RRUFF_usable_plus_recoverable_325_with_labels_maxnorm.hdf5 \
  --output-dir /scratch/ALLOC3/UserE/resnet50_rruff_runs \
  --run-name pilot3_bs64_lr3e4 \
  --epochs 10 \
  --batch-size 64 \
  --num-workers 4 \
  --prefetch-factor 4 \
  --learning-rate 3e-4 \
  --min-learning-rate 1e-5 \
  --weight-decay 1e-4 \
  --warmup-epochs 1 \
  --label-smoothing 0.05 \
  --log-interval 100 \
  --save-every-epoch \
  --resume-checkpoint /scratch/ALLOC3/UserE/resnet50_rruff_runs/pilot3_bs64_lr3e4/checkpoint_latest.pt
