#!/bin/bash
#SBATCH -J cnn-r473-dag
#SBATCH -A PHY26002
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 00:10:00
#SBATCH -o /scratch/ALLOC3/UserE/cnn_rruff473_dag.%j.out
#SBATCH -e /scratch/ALLOC3/UserE/cnn_rruff473_dag.%j.err

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Refusing to run outside a Slurm compute allocation. Submit this file with sbatch." >&2
  exit 2
fi

code_root=${NEURIPS_REBUTTAL_CODE_ROOT:-/scratch/ALLOC3/UserE/neurips_rebuttal_20260726/repo}
output_root=${NEURIPS_REBUTTAL_OUTPUT_ROOT:-/scratch/ALLOC3/UserE/neurips_rebuttal_20260726}
converged_checkpoint=${1:-/scratch/ALLOC3/UserE/resnet_rruff_runs/submitted_resnet18_convergence/checkpoint_best.pt}
submitted_checkpoint=${2:-/scratch/ALLOC3/UserE/resnet_rruff_runs/resume6_from_epoch3_bs128/checkpoint_epoch_06_frozen.pt}
benchmark473=/work2/ALLOC3/UserE/rruff-benchmark/RRUFF_option1_473_with_buckets_maxnorm.hdf5
benchmark325=/work2/ALLOC3/UserE/rruff-benchmark/RRUFF_usable_plus_recoverable_325_with_labels_maxnorm.hdf5
graph="${code_root}/artifacts/topology_graphs/extinction_group_adjacency.json"

module purge
module load gcc/13.2.0 cuda/12.5 python3/3.11.8
source /scratch/ALLOC3/UserE/project-venv/bin/activate

python "${code_root}/Code/Regular_CNN_Nvidia/eval_resnet_rruff_dag.py" \
  --checkpoint "${converged_checkpoint}" \
  --checkpoint-label converged_best_validation \
  --eval-h5 "${benchmark473}" \
  --graph-json "${graph}" \
  --output-json "${output_root}/cnn_converged_rruff473_dag.json"

python "${code_root}/Code/Regular_CNN_Nvidia/eval_resnet_rruff_dag.py" \
  --checkpoint "${submitted_checkpoint}" \
  --checkpoint-label submitted_epoch6 \
  --eval-h5 "${benchmark473}" \
  --graph-json "${graph}" \
  --output-json "${output_root}/cnn_submitted_epoch6_rruff473_dag.json"

python "${code_root}/Code/Regular_CNN_Nvidia/eval_resnet_rruff_dag.py" \
  --checkpoint "${converged_checkpoint}" \
  --checkpoint-label converged_best_validation \
  --eval-h5 "${benchmark325}" \
  --graph-json "${graph}" \
  --output-json "${output_root}/cnn_converged_rruff325_dag.json"

python "${code_root}/Code/Regular_CNN_Nvidia/eval_resnet_rruff_dag.py" \
  --checkpoint "${submitted_checkpoint}" \
  --checkpoint-label submitted_epoch6 \
  --eval-h5 "${benchmark325}" \
  --graph-json "${graph}" \
  --output-json "${output_root}/cnn_submitted_epoch6_rruff325_dag.json"
