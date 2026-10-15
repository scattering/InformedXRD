#!/bin/bash
#SBATCH -J prof-rruff-cond
#SBATCH -p skx-dev
#SBATCH -A CDA24014
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 00:30:00
#SBATCH -o /scratch/ALLOC3/UserE/prof-rruff-cond.%j.out
#SBATCH -e /scratch/ALLOC3/UserE/prof-rruff-cond.%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$SCRATCH/project}"
WORKROOT="${WORKROOT:-$SCRATCH/pycrysfml_spr}"
VENV_DIR="${VENV_DIR:-$WORKROOT/venv}"
FULLPROF_DIR="${FULLPROF_DIR:-$WORKROOT/fullprof}"
EXT_MAP_CSV="${EXT_MAP_CSV:-$REPO_DIR/Code/Post_Processing/FINAL_SPG_ExtG_CrysS_Table.csv}"
RRUFF_PRIOR_CSV="${RRUFF_PRIOR_CSV:-/work2/ALLOC3/UserE/rruff-benchmark/option1_metadata_qc_full.csv}"
ICSD_CHEMISTRY_CSV="${ICSD_CHEMISTRY_CSV:-/work2/ALLOC3/UserE/reference_data/ICSD_With_Bravais.csv}"
PRIOR_MODE="${PRIOR_MODE:-robust}"
EG_SAMPLING_MODE="${EG_SAMPLING_MODE:-rruff}"
PROFILE_SAMPLES="${PROFILE_SAMPLES:-10}"
PROFILE_SEED="${PROFILE_SEED:-12345}"

module purge
module load gcc/13.2.0
module load python/3.12.11

source "$VENV_DIR/bin/activate"

export FULLPROF="$FULLPROF_DIR"
export EXT_MAP_CSV
export RRUFF_PRIOR_CSV
export ICSD_CHEMISTRY_CSV
export PRIOR_MODE
export EG_SAMPLING_MODE

cd "$REPO_DIR/Code/Generation"
python tacc/profile_rruff_conditioned.py --samples "$PROFILE_SAMPLES" --seed "$PROFILE_SEED"
