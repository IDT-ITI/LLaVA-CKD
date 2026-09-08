# Train


- Replace data paths with yours in `scripts/train/train_.sh`
- Replace `output_dir` with yours in `scripts/train/pretrain_.sh`
- Replace `pretrained_model_path` and `output_dir` with yours in `scripts/train/finetune.sh`
- Adjust your GPU ids (localhost) and `per_device_train_batch_size` in `scripts/train/pretrain.sh` and `scripts/train/finetune.sh`

* Stage 1

```
bash scripts/train/llava_kd_qwen2/train_qwen2_distill_bu.sh
bash LLaVA-KD/scripts/train/llava_kd_qwen2/train_qwen2_distill_bu.sh
bash TinyLLaVA/scripts/train/qwen2/train_qwen2_base_bu.sh
```

* Stage 2

```
bash scripts/train/llava_kd_qwen2/train_qwen2_distill_bu.sh
bash LLaVA-KD/scripts/train/llava_kd_qwen2/train_qwen2_distill_bu.sh
bash TinyLLaVA/scripts/train/qwen2/train_qwen2_base_bu.sh
```
