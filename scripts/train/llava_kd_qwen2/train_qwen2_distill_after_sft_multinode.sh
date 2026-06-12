#!/bin/bash

#SBATCH --job-name="LLaVA-KD"        # name
#SBATCH --nodes=2                    # nodes
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --gres=gpu:1                 # number of gpus
#SBATCH --time 20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out           # output file name
#SBATCH --partition=gpu

export GPUS_PER_NODE=1
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

echo $SLURM_JOB_NODELIST
echo $MASTER_ADDR

# script_name=$0
# echo "script_name: $script_name"

source ./venv/bin/activate

# nvidia-smi
# python -V
# nvcc -V
# python -c "import torch; print(torch.__version__)"


DATA_PATH=/shared/LlavaImg/text_files/llava_v1_5_mix665k.json #finetune annotation file path
IMAGE_PATH=/shared/LlavaImg #finetune image dir
LLM_VERSION=Qwen/Qwen2.5-0.5B # llm path in huggingface
VT_VERSION=google/siglip-so400m-patch14-384 #vision tower path in huggingface
VT_VERSION2="" #if you are not using mof vision tower, keep it empty
CN_VERSION=mlp2x_gelu #connector type, other options are: qformer, resampler, etc
CONV_VERSION=qwen2_base #chat template, other options are: phi, llama, gemmma, etc
VERSION=qwen2-0_5b_base #experiment name for recording different runnings
# TRAIN_RECIPE="common" #training recipes, other options are: lora, qlora
# MODEL_MAX_LENGTH=2048 #max model length for llm
TEACHER_PATH=/scratch/checkpoints/TinyLlava_ckpts/TinyLLaVA-Qwen2.5-3B-SigLIP
TEACHER_NAME=tiny-llava-Qwen2.5-3B-siglip-so400m-patch14-384-base-finetune
VT_VARIANT=siglip-so400m-patch14-384
LLM_VARIANT=Qwen2.5-0.5B

srun --jobid $SLURM_JOBID bash -c 'python -m torch.distributed.run \
 --nproc_per_node $GPUS_PER_NODE --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID \
 --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
 llavakd/train/train_distill_after_qwen2_sft.py \
 --deepspeed ./scripts/zero3_offload.json \
 --data_path /shared/LlavaImg/text_files/llava_v1_5_mix665k.json \
 --image_folder /shared/LlavaImg \
 --teacher_pretrained_dir /scratch/checkpoints/TinyLlava_ckpts/TinyLLaVA-Qwen2.5-3B-SigLIP \
 --teacher_pretrained_name tiny-llava-Qwen2.5-3B-siglip-so400m-patch14-384-base-finetune \
 --is_multimodal True \
 --conv_version qwen2_base \
 --model_name_or_path Qwen/Qwen2.5-0.5B \
 --vision_tower google/siglip-so400m-patch14-384 \
 --vision_tower2 "" \
 --connector_type mlp2x_gelu \
 --mm_vision_select_layer -2 \
 --image_aspect_ratio square \
 --attn_implementation flash_attention_2 \
 --bf16 True \
 --training_recipe common \
 --tune_type_llm full \
 --tune_type_vision_tower frozen\
 --tune_vision_tower_from_layer 0 \
 --tune_type_connector full \
 --group_by_modality_length True \
 --pretrained_model_path "/scratch/checkpoints/qwen2_distill_llava_factory/tiny-llava-Qwen1.5-0.5B-siglip-so400m-patch14-384-base-finetune" \
 --output_dir "/scratch/checkpoints/qwen25_distill_llava_factory/tiny-llava-${LLM_VARIANT}-${VT_VARIANT}-${VERSION}-3B_to_05B-DFT" \
 --num_train_epochs 1 \
 --per_device_train_batch_size 1 \
 --per_device_eval_batch_size 4 \
 --gradient_accumulation_steps 128 \
 --save_strategy "steps" \
 --save_steps 500 \
 --save_total_limit 1 \
 --learning_rate 2e-5 \
 --weight_decay 0. \
 --warmup_ratio 0.03 \
 --lr_scheduler_type "cosine" \
 --logging_steps 1 \
 --tf32 False \
 --model_max_length 2048 \
 --gradient_checkpointing True \
 --dataloader_num_workers 2 \
 --lazy_preprocess True \
 --report_to tensorboard \
 --tokenizer_use_fast False \
 --run_name "tiny-llava-Qwen2.5-0.5B-VT_VARIANT-qwen2-0_5b_base-finetune"'