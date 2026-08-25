#!/bin/bash

#SBATCH --job-name="CLVKD-BU-PT"
#SBATCH --nodelist=iti-1290
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
## #SBATCH --gpu-bind=single:0
#SBATCH --time=2-00:00:00
#SBATCH --partition=highmem
#SBATCH --output=%x-%j.out

source ./venv/bin/activate


DATA_PATH=/shared_v2/datasets/LlavaImg/text_files/blip_laion_cc_sbu_558k.json #pretrain annotation file path
FINETUNE_DATA_PATH=/shared_v2/datasets/LlavaImg/text_files/llava_v1_5_mix665k.json #finetune annotation file path
IMAGE_PATH=/shared_v2/datasets/LlavaImg/llava/llava_pretrain/images #pretrain image dir
FINETUNE_IMAGE_PATH=/shared_v2/datasets/LlavaImg #finetune image dir

LLM_VERSION=Qwen/Qwen2.5-0.5B # llm path or huggingface repository path (Qwen/Qwen2.5-0.5B)
VT_VERSION=google/siglip-so400m-patch14-384 #vision tower path in huggingface
VT_VERSION2="" #if you are not using mof vision tower, keep it empty
CN_VERSION=mlp2x_gelu #connector type, other options are: qformer, resampler, etc
CONV_VERSION=qwen2_base #chat template, other options are: phi, llama, gemmma, etc
VERSION=qwen2-3b_05b_lbu_base #experiment name for recording different runnings
TRAIN_RECIPE=common #training recipes, other options are: lora, qlora
MODEL_MAX_LENGTH=2048 #max model length for llm

bash scripts/train/llava_kd_qwen2/pretrain_qwen2_distill_bu.sh "$DATA_PATH" "$IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH"
