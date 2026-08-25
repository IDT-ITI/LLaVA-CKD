#!/bin/bash

MODEL_PATH=$1
MODEL_NAME=$2
EVAL_DIR=$3

python3 -m llavakd.eval.model_vqa_mmmu \
    --model-path $MODEL_PATH \
    --question-file $EVAL_DIR/MMMU/anns_for_eval.json \
    --image-folder $EVAL_DIR/MMMU/all_images \
    --answers-file $EVAL_DIR/MMMU/answers/$MODEL_NAME.jsonl \
    --temperature 0 \
    --conv-mode phi

python3 scripts/convert_answer_to_mmmu.py \
    --answers-file $EVAL_DIR/MMMU/answers/$MODEL_NAME.jsonl \
    --answers-output $EVAL_DIR/MMMU/answers/${MODEL_NAME}_output.json

cd $EVAL_DIR/MMMU/eval

#<<<<<<< HEAD
python3 main_eval_only.py --output_path $EVAL_DIR/MMMU/answers/${MODEL_NAME}_output.json
#=======
python3 main_eval_only.py --output_path $EVAL_DIR/MMMU/answers/${MODEL_NAME}_output.json
#>>>>>>> origin/main
