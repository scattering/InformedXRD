#!/bin/bash
#SBATCH -J supervised_trainer_dcs
#SBATCH -o supervised_trainer_dcs.o%j
#SBATCH -e supervised_trainer_dcs.e%j
#SBATCH -p h100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 48:00:00
#SBATCH --mail-user=usera@example.org
#SBATCH --mail-type=all

echo "=== GPU state before training ==="
nvidia-smi

bash train.sh

echo "=== GPU state after training ==="
nvidia-smi
