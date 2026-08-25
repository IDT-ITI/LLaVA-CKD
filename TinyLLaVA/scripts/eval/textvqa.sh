#!/bin/bash

#MODEL_PATH="/data/LMMKD/checkpoints/llava_factory/tiny-llava-Qwen2-0.5B-siglip-so400m-patch14-384-qwen2-0_5b_base-finetune/tinyllava_model/"
#MODEL_PATH="/data/LMMKD/checkpoints/llava_factory/tiny-llava-Qwen2-1.5B-siglip-so400m-patch14-384-qwen2-1_5b_base-finetune/checkpoint-4000/"
#MODEL_PATH="/data/LMMKD/checkpoints/llava_factory/tiny-llava-Qwen2-3B-siglip-so400m-patch14-384-qwen2-3b_base-finetune/tinyllava_model/"
#MODEL_PATH="/data/LMMKD/checkpoints/llava_factory/tiny-llava-Qwen2-1.5B-siglip-so400m-patch14-384-qwen2-1_5b_base-finetune/checkpoint-5198/"
#MODEL_PATH="/data/LMMKD/checkpoints/llava_factory/tiny-llava-Qwen2.5-0.5B-siglip-so400m-patch14-384-qwen2-0_5b_base-3B_to_05B-DFT/checkpoint-2598/"
MODEL_PATH="/data/LMMKD/checkpoints/llava_factory/tiny-llava-Qwen3-0.6B-siglip-so400m-patch14-384-qwen3-06b_base-finetune/checkpoint-5198/"

# tiny-llava-Qwen2-0.5B-siglip-so400m-patch14-384-qwen2-0_5b_base-finetune
# tiny-llava-Qwen2-3B-siglip-so400m-patch14-384-qwen2-3b_base-finetune
# tiny-llava-Qwen2-1.5B-siglip-so400m-patch14-384-qwen2-1_5b_base-finetune
# tiny-llava-Qwen2.5-0.5B-siglip-so400m-patch14-384-qwen2-0_5b_base-3B_to_05B-DFT
MODEL_NAME="qwen3_06b_FT"
EVAL_DIR="/home/gkalelis/Documents/py/datasets/tinyLlavaEval"

python -m tinyllava.eval.model_vqa_loader \
    --model-path $MODEL_PATH \
    --question-file $EVAL_DIR/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder $EVAL_DIR/textvqa/train_images \
    --answers-file $EVAL_DIR/textvqa/answers/$MODEL_NAME.jsonl \
    --temperature 0 \
    --conv-mode qwen2_base # qwen2_base, phi

python -m tinyllava.eval.eval_textvqa \
    --annotation-file $EVAL_DIR/textvqa/TextVQA_0.5.1_val.json \
    --result-file $EVAL_DIR/textvqa/answers/$MODEL_NAME.jsonl

