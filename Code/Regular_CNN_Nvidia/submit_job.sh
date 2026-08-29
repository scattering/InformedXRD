#!/bin/bash
#SBATCH --job-name=ResNetTrainer
#SBATCH -o SlurmOutputs/%j.log
#SBATCH -e SlurmOutputs/%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=userc@example.org
#SBATCH --partition=cuda+
#SBATCH -N 1
#SBATCH --ntasks-per-node=4    # 4 tasks per node, matching 4 GPUs
#SBATCH --gres=gpu:4           # Request 4 GPUs
#SBATCH --time=48:00:00

bash train.sh
