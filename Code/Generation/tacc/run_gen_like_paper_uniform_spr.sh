#!/bin/bash
#SBATCH -J gen-paper-u99
#SBATCH -p spr
#SBATCH -A PHY26002
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -t 04:00:00
#SBATCH -o /home1/ALLOC3/UserE/gen-paper-u99.%j.out
#SBATCH -e /home1/ALLOC3/UserE/gen-paper-u99.%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/project}"
WORKROOT="${WORKROOT:-$SCRATCH/pycrysfml_spr}"
VENV_DIR="${VENV_DIR:-$WORKROOT/venv}"
FULLPROF_DIR="${FULLPROF_DIR:-$WORKROOT/fullprof}"
OUT_DIR="${OUT_DIR:-$SCRATCH/ai_diffraction_generated}"

TOTAL_SAMPLES="${TOTAL_SAMPLES:-4096}"
GENMAP_NUM_SHARDS="${GENMAP_NUM_SHARDS:-1}"
GENMAP_SHARD_ID="${GENMAP_SHARD_ID:-0}"
BATCH_ID="${BATCH_ID:-0}"
BASE_OUTPUT_HDF5="${BASE_OUTPUT_HDF5:-$OUT_DIR/uniform_eg_impurity}"
EXT_MAP_CSV="${EXT_MAP_CSV:-$REPO_DIR/Code/Post_Processing/FINAL_SPG_ExtG_CrysS_Table.csv}"

module purge
module load gcc/13.2.0
module load python/3.12.11

if [ ! -d "$VENV_DIR" ]; then
  echo "Missing VENV_DIR=$VENV_DIR"
  exit 2
fi

if [ ! -d "$FULLPROF_DIR" ]; then
  echo "Missing FULLPROF_DIR=$FULLPROF_DIR"
  exit 3
fi

source "$VENV_DIR/bin/activate"
mkdir -p "$OUT_DIR"

if ! python - <<'PY'
try:
    import pycrysfml  # noqa: F401
    ok = True
except Exception:
    try:
        import CFML_api  # noqa: F401
        ok = True
    except Exception:
        ok = False
raise SystemExit(0 if ok else 1)
PY
then
  WHEEL_PATH="$WORKROOT/crysfml/cfml-0.0.3-py3-none-linux_x86_64.whl"
  if [ ! -f "$WHEEL_PATH" ]; then
    echo "Missing wheel $WHEEL_PATH"
    exit 4
  fi
  python -m pip install "$WHEEL_PATH"
fi

export FULLPROF="$FULLPROF_DIR"
export EG_SAMPLING_MODE="uniform"
export TOTAL_SAMPLES
export GENMAP_NUM_SHARDS
export GENMAP_SHARD_ID
export BATCH_ID
export BASE_OUTPUT_HDF5
export EXT_MAP_CSV
cd "$REPO_DIR/Code/Generation"
NUM_PROCESSES_OVERRIDE="${NUM_PROCESSES_OVERRIDE:-1}" python gen_like_paper.py
