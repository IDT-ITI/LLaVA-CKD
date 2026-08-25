import argparse
import os

import torch

import sys
sys.path.append('/home/gkalelis/Documents/py/LLaVA-KD')

from llavakd.model.modeling_LLaVA_KD import LLaVAKD

from llavakd.utils import disable_torch_init
from llavakd.model.load_model import load_pretrained_model
#from tinyllava.data.dataset import make_supervised_data_module
#from tinyllava.train.tinyllava_trainer import LLaVATrainer

from llavakd.utils import get_state_maybe_zero_3
from transformers import PreTrainedModel


def save_from_pretrained(args):

    disable_torch_init()

    model_path = os.path.expanduser(args.model_path)
    #model, tokenizer, image_processor, context_len = load_pretrained_model(model_path)
    #model, _, _, _ = load_pretrained_model(os.path.join(model_path, 'hf_model'))
    #model, _, _, _ = load_pretrained_model(os.path.join(model_path))
    model = LLaVAKD.from_pretrained(model_path, low_cpu_mem_usage=True, dtype=torch.float16) #, dtype=torch.float16
    #model.to(device='cuda')

    #training_args_path = os.path.join(model_path, 'training_args.bin')
    #training_arguments = torch.load(training_args_path, map_location='cpu')
    #data_arguments = model.config
    #data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_arguments)
    #trainer = LLaVATrainer(model=model, tokenizer=tokenizer, args=training_arguments, **data_module)
    #training_recipe.save(model, trainer)

    model.config.use_cache = True
    # save tokenizer
    model.tokenizer.save_pretrained(model_path)
    # save entire model config
    model.config.save_pretrained(model_path, from_pt=True)

    # the followings are for pretrain stage
    # save language model
    language_model_state_dict = get_state_maybe_zero_3(model.language_model.named_parameters(), [''], False)
    language_model_output_dir = os.path.join(model_path, 'language_model')
    os.makedirs(language_model_output_dir, exist_ok=True)
    language_model_output_path = os.path.join(model_path, 'language_model/pytorch_model.bin')
    torch.save(language_model_state_dict, language_model_output_path)
    model.config.text_config.save_pretrained(language_model_output_dir, from_pt=True)

    # save vision tower
    vision_tower_state_dict = get_state_maybe_zero_3(model.vision_tower._vision_tower.named_parameters(), [''], False)
    vision_tower_output_dir = os.path.join(model_path, 'vision_tower')
    os.makedirs(vision_tower_output_dir, exist_ok=True)
    vision_tower_output_path = os.path.join(model_path, 'vision_tower/pytorch_model.bin')
    torch.save(vision_tower_state_dict, vision_tower_output_path)
    if isinstance(model.vision_tower._vision_tower, PreTrainedModel):
        model.vision_tower._vision_tower.config.save_pretrained(vision_tower_output_dir, from_pt=True)

    # save connector
    connector_state_dict = get_state_maybe_zero_3(model.connector.named_parameters(), [''], False)
    connector_output_dir = os.path.join(model_path, 'connector')
    os.makedirs(connector_output_dir, exist_ok=True)
    connector_output_path = os.path.join(model_path, 'connector/pytorch_model.bin')
    torch.save(connector_state_dict, connector_output_path)

    print("save_from_pretrained>> finished!")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--model-path", type=str, default='/data/LMMKD/checkpoints/llava_factory/tiny-llava-Qwen2-3B-siglip-so400m-patch14-384-qwen2-3b_base-finetune/tinyllava_model')
    parser.add_argument("--model-path", type=str, required=True)

    args = parser.parse_args()

    save_from_pretrained(args)