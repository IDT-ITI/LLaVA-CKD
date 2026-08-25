#!/bin/bash

#SBATCH --job-name="Safetensor2Components"
#SBATCH --nodelist=iti-1290
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus=2
#SBATCH --time=1-00:00:00
#SBATCH --partition=highmem
#SBATCH --output=%x-%j.out           # output file name

# script_name=$0
# echo "script_name: $script_name"

source ./venv/bin/activate


SAFE_TENSOR_PATH=/scratch/checkpoints/qwen25_distill_llava_factory/tiny-llava-Qwen2.5-0.5B-siglip-so400m-patch14-384-qwen2-3b_0_5b_base-finetune # safetensor path of the model

python llavakd/model/save_from_pretrained.py --model-path  $SAFE_TENSOR_PATH
