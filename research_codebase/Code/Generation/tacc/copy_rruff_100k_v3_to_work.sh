#!/bin/bash
#SBATCH -J copy-rruff100k
#SBATCH -A CDA24014
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH -t 00:30:00
#SBATCH -o /scratch/ALLOC3/UserE/copy-rruff100k.%j.out
#SBATCH -e /scratch/ALLOC3/UserE/copy-rruff100k.%j.err

set -euo pipefail

cp /scratch/ALLOC3/UserE/ai_diffraction_generated/rruff_conditioned_100k_v3_trainready.hdf5 \
   /work2/ALLOC3/UserE/rruff_conditioned_100k_v3_trainready.hdf5

ls -lh /work2/ALLOC3/UserE/rruff_conditioned_100k_v3_trainready.hdf5
