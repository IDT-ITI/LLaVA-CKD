#!/bin/bash

#SBATCH --job-name="EV-MME"
## #SBATCH --nodelist=iti-930
#SBATCH --nodelist=iti-1290
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
## #SBATCH --gres=gpu:1
#SBATCH --gpus=1
#SBATCH --time=2-00:00:00
#SBATCH --partition=highmem
#SBATCH --output=%x-%j.out

MODEL_PATH="/scratch/checkpoints/llava_factory/tiny-llava-Qwen2.5-1.5B-siglip-so400m-patch14-384-qwen25-15b_base-finetune/tinyllava_model"
MODEL_NAME="tiny-llava-Qwen2.5-1.5B-siglip-so400m-patch14-384-qwen25-15b_base-finetune"
EVAL_DIR="/shared_v2/datasets/LlavaImgEval"

source /home/public/sgkalelis/prj/CKD/venv/bin/activate

python3 -m tinyllava.eval.model_vqa_loader \
    --model-path $MODEL_PATH \
    --question-file $EVAL_DIR/MME/llava_mme.jsonl \
    --image-folder $EVAL_DIR/MME/MME_Benchmark_release_version \
    --answers-file $EVAL_DIR/MME/answers/$MODEL_NAME.jsonl \
    --temperature 0 \
   --conv-mode phi

cd $EVAL_DIR/MME

python convert_answer_to_mme.py --experiment $MODEL_NAME

cd eval_tool

python calculation.py --results_dir answers/$MODEL_NAME

