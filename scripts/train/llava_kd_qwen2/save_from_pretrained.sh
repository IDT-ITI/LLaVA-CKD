#SAFETENSOR_PATH=/data/LMMKD/checkpoints/qwen25_distill_llava_factory/tiny-llava-Qwen3-0.6B-siglip-so400m-patch14-384-qwen3-1_5b_06b_base-distill-finetune
SAFETENSOR_PATH=/data/LMMKD/checkpoints/llava_factory/tiny-llava-Qwen3-0.6B-siglip-so400m-patch14-384-qwen3-06b_base-finetune

python3 llavakd/model/save_from_pretrained.py  --model-path $SAFETENSOR_PATH
