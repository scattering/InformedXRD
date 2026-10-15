#!/bin/bash
#SBATCH -J xrd-paired
#SBATCH -A PHY26002
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH -t 00:05:00
#SBATCH -o /scratch/ALLOC3/UserE/xrdbench_pairwise_mcnemar_20260727/slurm.%j.out
#SBATCH -e /scratch/ALLOC3/UserE/xrdbench_pairwise_mcnemar_20260727/slurm.%j.err

set -euo pipefail

source_root=/scratch/ALLOC3/UserE/xrdbench_clean_second_set_20260726/runs/job_867777
run_root=/scratch/ALLOC3/UserE/xrdbench_pairwise_mcnemar_20260727
script=${run_root}/paired_mcnemar_from_failure_json.py

mkdir -p "${run_root}"
module purge
module load gcc/13.2.0 python3/3.11.8

for benchmark in rruff473 rruff325; do
  python "${script}" \
    --benchmark "${benchmark}" \
    --model "CNN10=${source_root}/cnn10/evaluation/${benchmark}/failure.json" \
    --model "PatchTST=${source_root}/PatchTST/evaluation/${benchmark}/failure.json" \
    --model "CNN11=${source_root}/original_models_common_grid/cnn11/${benchmark}/failure.json" \
    --top-k 1 3 5 \
    --bootstrap 10000 \
    --output-json "${run_root}/${benchmark}_pairwise_mcnemar.json"
done
