#!/bin/bash
#SBATCH -J supervised_trainer_dcs
#SBATCH -o supervised_trainer_dcs.o%j
#SBATCH -e supervised_trainer_dcs.e%j
#SBATCH -p pvc
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH --mail-user=userb@example.org
#SBATCH --mail-type=all

./train.sh