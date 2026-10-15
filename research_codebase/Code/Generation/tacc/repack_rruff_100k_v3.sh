#!/bin/bash
#SBATCH -J repack-rruff100k
#SBATCH -A CDA24014
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 02:00:00
#SBATCH -o /scratch/ALLOC3/UserE/repack-rruff100k.%j.out
#SBATCH -e /scratch/ALLOC3/UserE/repack-rruff100k.%j.err

set -euo pipefail

module load python/3.12.11
source /scratch/ALLOC3/UserE/pycrysfml_spr/venv/bin/activate

python /scratch/ALLOC3/UserE/project/Code/Generation/tacc/repack_shards_trainready_with_label.py \
  --input-dir /scratch/ALLOC3/UserE/ai_diffraction_generated/rruff_conditioned_100k_v3 \
  --output-h5 /scratch/ALLOC3/UserE/ai_diffraction_generated/rruff_conditioned_100k_v3_trainready.hdf5 \
  --label-key extinction_group \
  --split-mode order
