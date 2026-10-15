#!/bin/bash
#SBATCH -J s2c-209-sub
#SBATCH -A CDA24014
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 00:30:00
#SBATCH -o /scratch/ALLOC3/UserE/stage2c_209_subset_ghdev.%j.out
#SBATCH -e /scratch/ALLOC3/UserE/stage2c_209_subset_ghdev.%j.err

set -euo pipefail

ROOT=/scratch/ALLOC3/UserE
REPO=$ROOT/project
IT_DIR="$REPO/Information Theory"
CHECKPOINT=${CHECKPOINT:-$ROOT/ai_diffraction_models/xrd_model_9rwv1qly_best.pth}
CONFIG=${CONFIG:-$REPO/Code/ViT_NVIDIA/config_rruff_conditioned_2346k_from_ic6gfmvm_physpe_coord.json}
EVAL_H5=${EVAL_H5:-/work2/ALLOC3/UserE/stage2c_stampede/RRUFF_option1_473_xy_processed_overlap_209_maxnorm.hdf5}
SUBSET_JSON=${SUBSET_JSON:-}
OUT_DIR=${OUT_DIR:-/scratch/ALLOC3/UserE/stage2c_rruff209_subset}
TAG=${TAG:-stage2c_rruff209_full_209}
AUX_TEMPERATURE=${AUX_TEMPERATURE:-5.0}

module purge
module load gcc/13.2.0 python3/3.11.8 cuda/12.4
source "$ROOT/project-venv/bin/activate"

mkdir -p "$OUT_DIR"

ARGS=(
  --checkpoint "$CHECKPOINT"
  --config "$CONFIG"
  --eval-h5 "$EVAL_H5"
  --output-json "$OUT_DIR/${TAG}_summary.json"
  --output-npz "$OUT_DIR/${TAG}_posteriors.npz"
  --aux-temperature "$AUX_TEMPERATURE"
  --device cuda
  --label-indexing auto
)

if [[ -n "$SUBSET_JSON" ]]; then
  ARGS+=(--subset-json "$SUBSET_JSON")
fi

python "$IT_DIR/minimal_vit_synthetic_eval.py" "${ARGS[@]}"
