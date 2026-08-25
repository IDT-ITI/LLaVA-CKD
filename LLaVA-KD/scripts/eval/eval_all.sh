#!/bin/bash


#MODEL_PATH=/scratch/checkpoints/qwen25_distill_llava_factory/llava_factory/tiny-llava-Qwen2.5-1.5B-siglip-so400m-patch14-384-qwen2-3b_1_5b_7b_1_5b_bu_base-distill-finetune
MODEL_PATH=/home/gkalelis/Documents/py/checkpoints/LMMKD/qwen25_distill_llava_factory/tiny-llava-Qwen2.5-1.5B-siglip-so400m-patch14-384-qwen2-3b_1_5b_7b_1_5b_bu_base-distill-finetune/checkpoint-2550

MODEL_NAME="3b_1_5b_7b_1_5b_bu"
EVAL_DIR="/home/gkalelis/Documents/py/datasets/LlavaImgEval"

echo "Running gqa"
CUDA_VISIBLE_DEVICES=0 bash scripts/eval/gqa.sh "$MODEL_PATH" "$MODEL_NAME" "$EVAL_DIR"

echo "Running sqa"
CUDA_VISIBLE_DEVICES=0 bash scripts/eval/sqa.sh "$MODEL_PATH" "$MODEL_NAME" "$EVAL_DIR"

echo "Running textvqa"
CUDA_VISIBLE_DEVICES=0 bash scripts/eval/textvqa.sh "$MODEL_PATH" "$MODEL_NAME" "$EVAL_DIR"

echo "Running pope"
CUDA_VISIBLE_DEVICES=0 bash scripts/eval/pope.sh "$MODEL_PATH" "$MODEL_NAME" "$EVAL_DIR"

echo "Running mme"
#CUDA_VISIBLE_DEVICES=0 bash scripts/eval/mme.sh "$MODEL_PATH" "$MODEL_NAME" "$EVAL_DIR"

echo "Running mmmu"
CUDA_VISIBLE_DEVICES=0 bash scripts/eval/mmmu.sh "$MODEL_PATH" "$MODEL_NAME" "$EVAL_DIR"

echo "Running vqav2"
#CUDA_VISIBLE_DEVICES=0,1 bash scripts/eval/vqav2.sh "$MODEL_PATH" "$MODEL_NAME" "$EVAL_DIR"

echo "Running vizwiz"
#CUDA_VISIBLE_DEVICES=0 bash scripts/eval/vizwiz.sh "$MODEL_PATH" "$MODEL_NAME" "$EVAL_DIR"

echo "Running mmbench"
#CUDA_VISIBLE_DEVICES=0 bash scripts/eval/mmbench.sh "$MODEL_PATH" "$MODEL_NAME" "$EVAL_DIR"

echo "Running mmbench_cn"
#CUDA_VISIBLE_DEVICES=0 bash scripts/eval/mmbench_cn.sh "$MODEL_PATH" "$MODEL_NAME" "$EVAL_DIR"
