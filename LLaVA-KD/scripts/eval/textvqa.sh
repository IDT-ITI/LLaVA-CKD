#!/bin/bash

MODEL_PATH=$1
MODEL_NAME=$2
EVAL_DIR=$3

#MODEL_PATH="/scratch/checkpoints/qwen25_distill_llava_factory/tiny-llava-Qwen2.5-1.5B-siglip-so400m-patch14-384-qwen2-7b_1_5b_base-distill-finetune"
#MODEL_NAME="Qwen25-1_5b_7b_DFT"
#EVAL_DIR="/shared_v2/datasets/LlavaImgEval"

python3 -m llavakd.eval.model_vqa_loader \
    --model-path $MODEL_PATH \
    --question-file $EVAL_DIR/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder $EVAL_DIR/textvqa/train_images \
    --answers-file $EVAL_DIR/textvqa/answers/$MODEL_NAME.jsonl \
    --temperature 0 \
    --conv-mode phi

python3 -m llavakd.eval.eval_textvqa \
    --annotation-file $EVAL_DIR/textvqa/TextVQA_0.5.1_val.json \
    --result-file $EVAL_DIR/textvqa/answers/$MODEL_NAME.jsonl

