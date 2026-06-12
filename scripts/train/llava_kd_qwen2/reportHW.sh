#!/bin/bash

#SBATCH --job-name="RPRT-HW"
#SBATCH --nodelist=iti-1290
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus=2
#SBATCH --time=10:00:00
#SBATCH --partition=highmem
#SBATCH --output=%x-%j.out

# script_name=$0
# echo "script_name: $script_name"

source ./venv/bin/activate

nvidia-smi
python -V
nvcc -V