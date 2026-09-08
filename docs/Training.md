# Train

In all involved steps below the user has to replace data paths, and where applicable `TEACHER_PATH`, in `scripts/train/train_`***`.sh`, with the ones used in her/his experiment for training.
Above, the text replacing `***` in the name of the shell script depends on the stage step (see the scripts corresponding to the different steps below).  

Also ensure that the variable `VERSION` is the same along all steps within the same stage.

To facilitate the user, we assume the following setting for Student, Teacher Assistant (TA) and Teacher:
- Student: TinyLLaVA Qwen2.5-0.5B (`tiny-llava-Qwen2.5-1.5B-siglip-so400m-patch14-384-qwen25-05b_base-finetune`)
- TA: TinyLLaVA Qwen2.5-1.5B (`tiny-llava-Qwen2.5-1.5B-siglip-so400m-patch14-384-qwen25-1_5b_base-finetune`)
- Teacher: TinyLLaVA Qwen2.5-3B (`tiny-llava-Qwen2.5-1.5B-siglip-so400m-patch14-384-qwen25-3b_base-finetune`)

## Stage 1

In this stage we apply the 3-steps process of LLaVA-KD.

To reflect the settings of the above Student, TA and Teacher, the version variable (determining the name of the overall experiment in stage 1) along all shell scripts in this stage is set as following:

`VERSION=qwen25-1_5b_05b_base`

### Pretraining (PT)

In the file `train_distill_qwen2.py`:

Set the `teacher_dir` variable with the Teacher Assistant (TA) model path.

In our example (i.e., TinyLLaVA Qwen2.5-1.5B model is used as TA), set:

`teacher_dir = '/path/to/tiny-llava-Qwen2.5-1.5B-siglip-so400m-patch14-384-qwen25-1_5b_base-finetune'`.

Run the following script:
```
bash LLaVA-KD/scripts/train/llava_kd_qwen2/train_qwen2_distill.sh
```

For our example, this will produce the following pretrained Student model:

```
tiny-llava-Qwen2.5-0.5B-siglip-so400m-patch14-384-qwen25-1_5b_05b_base-distill-pretrain
```

### Supervised finetuning (FT)

In the file `train_sft.py`:

Set the `distill_pretrained_model_path` variable with the path of the model derived in the previous step.

That is, for our example, set:

`distill_pretrained_model_path = '/path/to/tiny-llava-Qwen2.5-0.5B-siglip-so400m-patch14-384-qwen25-1_5b_05b_base-distill-pretrain'`

Run the following script:
```
bash LLaVA-KD/scripts/train/llava_kd_qwen2/train_qwen2_sft.sh
```

For our example, this will produce the following finetuned Student model:

```
tiny-llava-Qwen2.5-0.5B-siglip-so400m-patch14-384-qwen25-1_5b_05b_base-finetune
```


### Distillation SFT (DFT)

In the file `train_distill_after_qwen2_sft.py`:

Set the `teacher_dir` variable with the path of the TA model.

In case the TinyLLaVA Qwen2.5-1.5B model is used as TA, as our example, set:

`teacher_dir = '/path/to/tiny-llava-Qwen2.5-1.5B-siglip-so400m-patch14-384-qwen25-1_5b_base-finetune'`

Set the `student_sft_path` variable with the path of the model derived in the previous step.
E.g., in case the TinyLLaVA Qwen2.5-0.5B model is used as Student, set:

`student_sft_path = '/path/to/tiny-llava-Qwen2.5-0.5B-siglip-so400m-patch14-384-qwen25-1_5b_05b_base-finetune'`

Run the following script:
```
bash LLaVA-KD/scripts/train/llava_kd_qwen2/train_qwen2_distill_after_sft.sh
```

For our example, this will produce the following finetuned Student model:

```
tiny-llava-Qwen2.5-0.5B-siglip-so400m-patch14-384-qwen25-1_5b_05b_base-distill-finetune
```

## Stage 2

In this stage we apply the 3-steps process of LLaVA-KD with the necessary modifications to enforce the cascaded bottom up (BU) distillation strategy.
To this end, in the second step (FT), we run the TinyLLaVA script `train_qwen2_base_bu.sh`.

To reflect the settings of the above Student, TA and Teacher, the version variable (determining the name of the overall experiment in stage 2) along all shell scripts in this stage is set as following:

`VERSION=qwen25-1_5b_05b_3b_05b_base`

### PT

In the file `train_distill_bu_qwen2.py`:

Set the `teacher_dir` variable with the Teacher model path.

For our example, where TinyLLaVA Qwen2.5-3B model is used as Teacher, set:

`teacher_dir = '/path/to/tiny-llava-Qwen2.5-1.5B-siglip-so400m-patch14-384-qwen25-3b_base-finetune'`.

Set the `student_dir` variable with the path of the model derived in the previous stage.

That is, for our example, set:

`student_dir = '/path/to/tiny-llava-Qwen2.5-0.5B-siglip-so400m-patch14-384-qwen25-1_5b_05b_base-distill-finetune'`

Run the following script:
```
bash LLaVA-KD/scripts/train/llava_kd_qwen2/train_qwen2_distill_bu.sh
```

For our example, this will produce the following finetuned Student model:

```
tiny-llava-Qwen2.5-0.5B-siglip-so400m-patch14-384-qwen25-1_5b_05b_3b_05b_base-distill-pretrain
```

### FT

In the script `train_qwen2_base_bu.sh` of TinyLLaVA set the correct version in the variable `VERSION` in order to ensure that the distilled Student model from the previous step is loaded.  

That is, for our example you may set: 
`VERSION=qwen25-1_5b_05b_3b_05b_base`

Run the following script:
```
bash TinyLLaVA/scripts/train/qwen2/train_qwen2_base_bu.sh
```

For our example, this will produce the following finetuned Student model:

```
tiny-llava-Qwen2.5-0.5B-siglip-so400m-patch14-384-qwen25-1_5b_05b_3b_05b_base-finetune
```

### DFT

In the file `train_distill_after_qwen2_sft.py`:

Set the `teacher_dir` variable with the path of the Teacher model.

For our example (where TinyLLaVA Qwen2.5-3B model is used as Teacher) set:

`teacher_dir = '/path/to/tiny-llava-Qwen2.5-1.5B-siglip-so400m-patch14-384-qwen25-3b_base-finetune'`

Set the `student_sft_path` variable with the path of the model derived in the previous step.

That is, for our example set:

`distill_pretrained_model_path = '/path/to/tiny-llava-Qwen2.5-0.5B-siglip-so400m-patch14-384-qwen25-1_5b_05b_3b_05b_base-distill-finetune'`

Run the following script:
```
bash LLaVA-KD/scripts/train/llava_kd_qwen2/train_qwen2_distill_after_sft.sh
```

For our example, this will produce the final Student model:

```
tiny-llava-Qwen2.5-0.5B-siglip-so400m-patch14-384-qwen25-1_5b_05b_3b_05b_base-distill_finetune
```

