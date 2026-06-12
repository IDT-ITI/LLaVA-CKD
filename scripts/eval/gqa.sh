#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

SPLIT="llava_gqa_testdev_balanced"
#GQADIR="./eval_dataset/eval/gqa"

MODEL_PATH=$1
MODEL_NAME=$2
EVAL_DIR=$3

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python3 -m llavakd.eval.model_vqa_loader \
        --model-path $MODEL_PATH \
        --question-file $EVAL_DIR/gqa/$SPLIT.jsonl \
        --image-folder $EVAL_DIR/gqa/images \
        --answers-file $EVAL_DIR/gqa/answers/$SPLIT/$MODEL_NAME/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode phi &
done

wait

output_file=$EVAL_DIR/gqa/answers/$SPLIT/$MODEL_NAME/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $EVAL_DIR/gqa/answers/$SPLIT/$MODEL_NAME/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python3 scripts/convert_gqa_for_eval.py --src $output_file --dst $EVAL_DIR/gqa/testdev_balanced_predictions.json

cd $EVAL_DIR/gqa
python3 eval/eval.py --tier testdev_balanced
