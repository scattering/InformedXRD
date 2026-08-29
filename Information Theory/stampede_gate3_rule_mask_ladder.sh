#!/bin/bash
#SBATCH -J it_rulemask
#SBATCH -p spr
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -t 02:00:00
#SBATCH -o /work2/ALLOC3/UserE/information_theory_outputs/it_rulemask_%j.out
#SBATCH -e /work2/ALLOC3/UserE/information_theory_outputs/it_rulemask_%j.err

set -euo pipefail

ROOT=/scratch/ALLOC3/UserE/project
IT_DIR="$ROOT/Information Theory"
OUT_DIR=/work2/ALLOC3/UserE/information_theory_outputs
ENV_ROOT=/work2/ALLOC3/UserE/shared/pycrysfml_spr
VENV="$ENV_ROOT/venv"

mkdir -p "$OUT_DIR"
cd "$ROOT"

source "$VENV/bin/activate"
export PYTHONPATH="$ENV_ROOT/src:$ROOT/Information Theory:$ROOT/Code/ViT_Streaming:$PYTHONPATH"
export PYCRYSFML_LIB="$ENV_ROOT/lib/libcrysfml.so"
export PYCRYSFML_MODS="$ENV_ROOT/src/pycrysfml"

SAMPLES_PER_EG=${SAMPLES_PER_EG:-100}
DIRICHLET_ALPHA=${DIRICHLET_ALPHA:-0.0}
EG_INDEX_START=${EG_INDEX_START:-0}
EG_INDEX_STOP=${EG_INDEX_STOP:-0}
SEED=${SEED:-123}
OUT_TAG=${OUT_TAG:-gate3_rule_mask_ladder}
SAVE_STRUCTURES=${SAVE_STRUCTURES:-0}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-150}

SAVE_STRUCTURES_FLAG=()
if [[ "$SAVE_STRUCTURES" == "1" ]]; then
  SAVE_STRUCTURES_FLAG+=(--save-structures)
fi

python3 "$IT_DIR/gate3_rule_mask_ladder.py" \
  --mapping-csv "$ROOT/Code/Post_Processing/canonical_extinction_to_space_group.csv" \
  --samples-per-eg "$SAMPLES_PER_EG" \
  --seed "$SEED" \
  --max-attempts "$MAX_ATTEMPTS" \
  --dirichlet-alpha "$DIRICHLET_ALPHA" \
  --eg-index-start "$EG_INDEX_START" \
  --eg-index-stop "$EG_INDEX_STOP" \
  --num-processes "${SLURM_CPUS_PER_TASK:-32}" \
  --out-json "$OUT_DIR/${OUT_TAG}.json" \
  --out-npz "$OUT_DIR/${OUT_TAG}.npz" \
  "${SAVE_STRUCTURES_FLAG[@]}"
