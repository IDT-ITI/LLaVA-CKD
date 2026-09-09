# Setup LLaVA-CKD

After downloading github repository of LLaVA-CKD, follow the steps below.

## 1. Move to LLaVA-CKD home directory and create python virtual environment

```shell
cd LLaVA-CKD
python3 -m venv venv
source venv/bin/activate
```

## 2. Install PyTorch and flash-attn version according to the installed python, torch and cuda version

```shell
pip3 install torch torchvision
pip3 install flash-attn --no-build-isolation
```

## 3. Install the following libraries

```shell
pip3 install transformers==4.57.6
pip3 install deepspeed
pip3 install peft
pip3 install einops-exts
pip3 install open-clip-torch
pip3 install tensorboard
pip3 install scikit-learn
pip3 install shortuuid
```
