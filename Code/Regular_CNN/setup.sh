#!/bin/bash
cd /scratch/ALLOC1/UserA
source oneapi/bin/activate
cd /home1/ALLOC1/UserA/crystals/
export OCL_ICD_VENDORS=/etc/OpenCL/vendors
export CCL_ROOT=${CONDA_PREFIX}
source /scratch/projects/compilers/intel25.0/setvars.sh --force
# Start MySQL in the background
#/home1/ALLOC1/UserA/mysql/bin/mysqld_safe --datadir=/home1/ALLOC1/UserA/mysql/data --socket=/home1/ALLOC1/UserA/mysql/mysql.sock --port=3307 &