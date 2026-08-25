#!/bin/bash

#SBATCH --job-name="CKD-FT-Q25-3B-15B-05B"
#SBATCH --nodelist=iti-1290
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
## #SBATCH --gres=gpu:1
#SBATCH --gpus=2
#SBATCH --time=2-00:00:00
#SBATCH --partition=highmem
#SBATCH --output=%x-%j.out

# script_name=$0
# echo "script_name: $script_name"

source /home/public/sgkalelis/prj/CKD/venv/bin/activate


DATA_PATH=/shared_v2/datasets/LlavaImg/text_files/blip_laion_cc_sbu_558k.json #pretrain annotation file path
FINETUNE_DATA_PATH=/shared_v2/datasets/LlavaImg/text_files/llava_v1_5_mix665k.json #finetune annotation file path
IMAGE_PATH=/shared_v2/datasets/LlavaImg/llava/llava_pretrain/images #pretrain image dir
FINETUNE_IMAGE_PATH=/shared_v2/datasets/LlavaImg #finetune image dir

LLM_VERSION=Qwen/Qwen2.5-0.5B # llm path in huggingface
VT_VERSION=google/siglip-so400m-patch14-384 #vision tower path in huggingface
VT_VERSION2="" #if you are not using mof vision tower, keep it empty
CN_VERSION=mlp2x_gelu #connector type, other options are: qformer, resampler, etc
CONV_VERSION=qwen2_base #chat template, other options are: phi, llama, gemmma, etc
VERSION=qwen25-15b_05b_3b_05b_base #experiment name for recording different runnings
TRAIN_RECIPE=common #training recipes, other options are: lora, qlora
MODEL_MAX_LENGTH=2048 #max model length for llm


bash scripts/train/qwen2/finetune_qwen2_bu.sh "$FINETUNE_DATA_PATH" "$FINETUNE_IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$CONV_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH"
