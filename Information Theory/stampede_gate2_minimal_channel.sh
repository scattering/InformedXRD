#!/bin/bash
#SBATCH -J it_gate2_min
#SBATCH -p spr
#SBATCH -A PHY26002
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -t 00:45:00
#SBATCH -o /work2/ALLOC3/UserE/information_theory_outputs/it_gate2_min_%j.out
#SBATCH -e /work2/ALLOC3/UserE/information_theory_outputs/it_gate2_min_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/scratch/ALLOC3/UserE/project}"
IT_DIR="$REPO_ROOT/Information Theory"
VENV="${VENV:-/work2/ALLOC3/UserE/shared/pycrysfml_spr/venv}"
OUT_JSON="${OUT_JSON:-/work2/ALLOC3/UserE/information_theory_outputs/gate2_minimal_channel.json}"
OUT_NPZ="${OUT_NPZ:-}"
SAMPLES_PER_EG="${SAMPLES_PER_EG:-2}"
SEED="${SEED:-123}"
DIRICHLET_ALPHA="${DIRICHLET_ALPHA:-0.5}"
EG_INDEX_START="${EG_INDEX_START:-0}"
EG_INDEX_STOP="${EG_INDEX_STOP:-0}"
SKIP_SUMMARY="${SKIP_SUMMARY:-0}"

mkdir -p "$(dirname "$OUT_JSON")"
cd "$REPO_ROOT"

source "$VENV/bin/activate"

cmd=(
  python "$IT_DIR/gate2_minimal_channel.py"
  --mapping-csv "$REPO_ROOT/Code/Post_Processing/canonical_extinction_to_space_group.csv"
  --samples-per-eg "$SAMPLES_PER_EG"
  --seed "$SEED"
  --eg-index-start "$EG_INDEX_START"
  --eg-index-stop "$EG_INDEX_STOP"
  --dirichlet-alpha "$DIRICHLET_ALPHA"
  --out-json "$OUT_JSON"
)
if [[ -n "$OUT_NPZ" ]]; then
  cmd+=(--out-npz "$OUT_NPZ")
fi
if [[ "$SKIP_SUMMARY" == "1" ]]; then
  cmd+=(--skip-summary)
fi
"${cmd[@]}"
