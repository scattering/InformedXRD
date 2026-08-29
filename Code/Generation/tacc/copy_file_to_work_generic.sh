#!/bin/bash
#SBATCH -J copy-trainready
#SBATCH -A CDA24014
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH -t 00:30:00

set -euo pipefail

SRC_PATH="${SRC_PATH:?SRC_PATH is required}"
DST_PATH="${DST_PATH:?DST_PATH is required}"

cp "$SRC_PATH" "$DST_PATH"
ls -lh "$DST_PATH"
