#!/bin/bash
#SBATCH -J mix-2500k
#SBATCH -A CDA24014
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 02:00:00
#SBATCH -o /scratch/ALLOC3/UserE/mix-2500k.%j.out
#SBATCH -e /scratch/ALLOC3/UserE/mix-2500k.%j.err

set -euo pipefail

module purge
module load python/3.12.11

source /scratch/ALLOC3/UserE/pycrysfml_spr/venv/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE

python /scratch/ALLOC3/UserE/project/Code/Generation/tacc/build_mixed_trainready_blocks.py \
  --standard /work2/ALLOC3/UserE/rruff_conditioned_2346k_v1_trainready.hdf5 \
  --po /scratch/ALLOC3/UserE/ai_diffraction_generated/rruff_conditioned_500k_po_v1_trainready.hdf5 \
  --output /scratch/ALLOC3/UserE/ai_diffraction_generated/rruff_conditioned_mixed_2500k_v1_trainready.hdf5 \
  --std-train 1600000 \
  --po-train 400000 \
  --std-val 200000 \
  --po-val 50000 \
  --std-test 200000 \
  --po-test 50000 \
  --chunk-rows 256 \
  --seed 1337
