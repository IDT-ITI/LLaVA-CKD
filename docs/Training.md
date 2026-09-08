# Train

In all involved steps below the user has to replace data paths, and where applicable `TEACHER_PATH`, in `scripts/train/train_`***`.sh`, with the ones used in her/his experiment.

## Stage 1
In this stage we apply the 3-steps process of LLaVA-KD.

### Pretraining (PT)

In the file `train_distill_qwen2.py`:

Set the `teacher_dir` variable with the teacher model path. E.g., in case the TinyLLaVA Qwen2.5-1.5B model is used as Teacher, set:

`teacher_dir = '/path/to/tiny-llava-Qwen2.5-1.5B-siglip-so400m-patch14-384-qwen25-15b_base-finetune'`.

Run the following script:
```
bash LLaVA-KD/scripts/train/llava_kd_qwen2/train_qwen2_distill.sh
```

### Supervised finetuning (FT)

In the file `train_sft.py`:

Set the `distill_pretrained_model_path` variable with the path of the model derived in the previous step. E.g., in case the TinyLLaVA Qwen2.5-0.5B model is used as Student, set:

`distill_pretrained_model_path = '/path/to/tiny-llava-Qwen2.5-0.5B-siglip-so400m-patch14-384-qwen25-3b_05b_ewt_base-distill-pretrain'`

Run the following script:
```
bash LLaVA-KD/scripts/train/llava_kd_qwen2/train_qwen2_sft.sh
```

### Distillation SFT (DFT)

```
bash LLaVA-KD/scripts/train/llava_kd_qwen2/train_qwen2_distill_after_sft.sh
```

## Stage 2

```
bash LLaVA-KD/scripts/train/llava_kd_qwen2/train_qwen2_distill_bu.sh
bash TinyLLaVA/scripts/train/qwen2/train_qwen2_base_bu.sh
bash LLaVA-KD/scripts/train/llava_kd_qwen2/train_qwen2_distill_after_sft.sh

```
