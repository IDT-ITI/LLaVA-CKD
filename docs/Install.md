# Install LLaVA-CKD

```shell
# move to LLaVA-CKD home directory
cd LLaVA-CKD
# create a python virtual environment
python3 -m venv venv
source venv/bin/activate
# install PyTorch version according to your cuda version
pip3 install torch torchvision
# install flash-attn appropriate version according to python, torch and cuda version 
pip3 install flash-attn --no-build-isolation
pip3 install transformers==4.57.6
pip3 install deepspeed
pip3 install peft
pip3 install einops-exts
pip3 install open-clip-torch
pip3 install tensorboard
pip3 install scikit-learn
pip3 install shortuuid
```