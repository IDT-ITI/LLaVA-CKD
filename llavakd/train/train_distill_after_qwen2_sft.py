from packaging import version
import pathlib
import safetensors

import tokenizers
import transformers

import sys
sys.path.append('/home/gkalelis/Documents/py/LLaVA-KD')
from llavakd.train.tinyllava_distill_trainer import DistillLLaVATrainer
from llavakd.training_recipe import TrainingRecipeFactory
from llavakd.utils import *
from llavakd.model import *
from llavakd.data.dataset import make_supervised_data_module


IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')

import wandb
os.environ["WANDB_API_KEY"] = 'YOUR API WANDB KEY'
os.environ["WANDB_MODE"] = "offline"

wandb.disabled = True # nikgkal

def load_settings(model_arguments, data_arguments, training_arguments):
    model_arguments.tune_type_connector = training_arguments.tune_type_connector
    model_arguments.tune_type_llm = training_arguments.tune_type_llm
    model_arguments.tune_type_vision_tower = training_arguments.tune_type_vision_tower
    model_arguments.image_aspect_ratio = data_arguments.image_aspect_ratio

    model_args = {}
    model_args['llm'] = _load_llm_settings(model_arguments)
    model_args['vision_tower'] = _load_vision_settings(model_arguments)
    model_args['connector'] = _load_connector_settings(model_arguments) 
    return model_args

def _load_llm_settings(model_arguments):
    llm_args = {}
    llm_args['model_name_or_path'] = model_arguments.model_name_or_path
    llm_args['cache_dir'] = model_arguments.cache_dir
    llm_args['attn_implementation'] = model_arguments.attn_implementation # flash_attention_2 only supports torch.float16 and torch.bfloat16 dtypes
    return llm_args

def _load_vision_settings(model_arguments):
    vision_args = {}
    vision_args['model_name_or_path'] = model_arguments.vision_tower.split(':')[-1]
    if model_arguments.vision_tower2 != '':
        vision_args['model_name_or_path2'] = model_arguments.vision_tower2.split(':')[-1]
    return vision_args

def _load_connector_settings(model_arguments):
    connector_args = {}
    connector_args['connector_type'] = model_arguments.connector_type
    return connector_args


def wandb_log(model_arguments):

    wandb_dir = "./wandb/"
    os.makedirs(wandb_dir, exist_ok=True)

    config = {
        "model_name": f"{model_arguments.model_name_or_path.split('/')[-1]}-{model_arguments.vision_tower.split('/')[-1]}",
        'total_epochs': 1,
    }
    if int(os.getenv('LOCAL_RANK', '0')) == 0:
        wandb.init(project='DistillTinyLLaVA',
                   name=f"{model_arguments.model_name_or_path.split('/')[-1]}-{model_arguments.vision_tower.split('/')[-1]}",
                   config=config,
                   dir=wandb_dir)
                

def get_conncet_w(weights, keyword):
    return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

def train():
    
    teacher_dir = '/data/LMMKD/checkpoints/llava_factory/tiny-llava-Qwen2-1.5B-siglip-so400m-patch14-384-qwen2-1_5b_base-finetune'
    pre_vision_tower_path = os.path.join(teacher_dir, 'vision_tower/pytorch_model.bin')
    pre_vision = torch.load(pre_vision_tower_path, map_location='cpu')
    pre_connector_path = os.path.join(teacher_dir, 'connector/pytorch_model.bin')
    pre_connector = torch.load(pre_connector_path, map_location='cpu')
        
    teacher_model_config = TinyLlavaConfig.from_pretrained(teacher_dir)
    teacher_model = LLaVAKD(teacher_model_config)

    teacher_model.language_model = teacher_model.language_model.from_pretrained(os.path.join(teacher_dir, 'language_model'))

    # load from vision pretrained
    teacher_model.vision_tower._vision_tower.load_state_dict(pre_vision)
    print(f"Teacher Loading vision from {pre_vision_tower_path}")

    # load from connector pretrained
    teacher_model.connector._connector.load_state_dict(get_conncet_w(pre_connector, '_connector'))
    print(f"Teacher Loading connector from {pre_connector_path}")
    
    teacher_model.eval()
        
    # load argument
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_arguments, data_arguments, training_arguments = parser.parse_args_into_dataclasses()

    print(f"Model name or path is {model_arguments.model_name_or_path}")

    logger_setting(getattr(training_arguments, 'output_dir', None))
    # wandb_log(model_arguments) # nikgkal

    training_recipe = TrainingRecipeFactory(training_arguments.training_recipe)(training_arguments) 
    # model_args contain arguements for huggingface model .from_pretrained function
    model_args = load_settings(model_arguments, data_arguments, training_arguments)
    model_args = training_recipe.add_args(model_args)
    model_config = TinyLlavaConfig()
    model_config.load_from_config(model_arguments)




    # load pretrained checkpoint
    student_sft_path = os.path.join('/data/LMMKD/checkpoints/qwen25_distill_llava_factory/tiny-llava-Qwen3-0.6B-siglip-so400m-patch14-384-qwen3-1_5b_06b_base-finetune', 'model.safetensors')
    with safetensors.safe_open(student_sft_path, framework="pt") as f:
        student_sft_weight = {k: f.get_tensor(k).detach().clone() for k in f.keys()} # nikgkal
        # student_sft_weight = {k: torch.tensor(f.get_tensor(k)) for k in f.keys()}

    model = LLaVAKD(model_config)
    model.load_state_dict(student_sft_weight, strict=False)
    
    # model.vision_tower._vision_tower.load_state_dict(pre_vision)  
    # print(f"Student Loading vision from {pre_vision_tower_path}")
    # model.connector._connector.load_state_dict(get_conncet_w(pre_connector, '_connector'))
    # print(f"Student Loading connector from {pre_connector_path}")
    # model.language_model = model.language_model.from_pretrained(os.path.join(student_sft_path, 'language_model'))

    model = training_recipe(model)
    model.config.use_cache = False
    model.config.image_aspect_ratio = data_arguments.image_aspect_ratio
    tokenizer = model.tokenizer
    data_arguments.image_processor = model.vision_tower._image_processor
    data_arguments.is_multimodal = True
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_arguments)

    log_trainable_params(model)  # not work well with zero3
    trainer = DistillLLaVATrainer(
        teacher_model=teacher_model,
        model=model, #does not require model.to(device), huggingface/deepspeed does it for you?
        tokenizer=tokenizer,
        args=training_arguments,
        **data_module)
    
    if eval(training_arguments.resume_from_checkpoint) is True: # nikgkal: added branch to resume from checkpoint
        print('Resuming from checkpoint...')
        # training_arguments.pretrained_model_path = None # nikgkal: model will resume from checkpoint in output_dir calling
        trainer.train(resume_from_checkpoint=True)
    elif eval(training_arguments.resume_from_checkpoint) is not None:
        # training_arguments.pretrained_model_path = None # nikgkal: model will resume from specific checkpoint
        print(f"Resuming from checkpoint: {training_arguments.resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=training_arguments.resume_from_checkpoint)
    else: # resume_from_checkpoint is None
        print(f"Normal training; resuming from checkpoint is: {training_arguments.resume_from_checkpoint}")
        trainer.train()
    
    training_recipe.save(model, trainer)

    wandb.finish()

if __name__ == "__main__":
    transformers.set_seed(1234)
    train()
