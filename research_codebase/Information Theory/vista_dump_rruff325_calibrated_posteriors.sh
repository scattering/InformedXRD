#!/bin/bash
#SBATCH -J it_rruff325_post
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00
#SBATCH -A PHY26002
#SBATCH -o /scratch/ALLOC3/UserE/information_theory_rruff325_post_%j.out
#SBATCH -e /scratch/ALLOC3/UserE/information_theory_rruff325_post_%j.err

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/scratch/ALLOC3/UserE/project}
VENV=${VENV:-/scratch/ALLOC3/UserE/project-venv}
SCRIPT_DIR="$REPO_ROOT/Information Theory"
OUT_DIR=${OUT_DIR:-/scratch/ALLOC3/UserE/information_theory_outputs}

CHECKPOINT=${CHECKPOINT:-/scratch/ALLOC3/UserE/ai_diffraction_models/xrd_model_9rwv1qly_best.pth}
EVAL_DATA_PATH=${EVAL_DATA_PATH:-/work2/ALLOC3/UserE/rruff-benchmark/RRUFF_usable_plus_recoverable_325_with_labels_maxnorm.hdf5}
PRIOR_DATA_PATH=${PRIOR_DATA_PATH:-/scratch/ALLOC3/UserE/ai_diffraction_generated/rruff_conditioned_2346k_v1_trainready.hdf5}
AUX_TEMPERATURE=${AUX_TEMPERATURE:-5.0}
NUM_WORKERS=${NUM_WORKERS:-4}

mkdir -p "$OUT_DIR"

source "$VENV/bin/activate"
python "$SCRIPT_DIR/dump_rruff325_calibrated_posteriors.py" \
  --checkpoint "$CHECKPOINT" \
  --eval-data-path "$EVAL_DATA_PATH" \
  --prior-data-path "$PRIOR_DATA_PATH" \
  --aux-temperature "$AUX_TEMPERATURE" \
  --num-workers "$NUM_WORKERS" \
  --output-npz "$OUT_DIR/rruff325_stage2c_9rwv1qly_t${AUX_TEMPERATURE}_posteriors.npz" \
  --output-json "$OUT_DIR/rruff325_stage2c_9rwv1qly_t${AUX_TEMPERATURE}_posteriors.json"
