from packaging import version
import pathlib
import safetensors

import tokenizers
import transformers

import sys
sys.path.append('/home/gkalelis/Documents/py/KD-LLaVA') # nikgkal: amend path

from llavakd.train.tinyllava_distill_trainer import DistillLLaVATrainer
from llavakd.training_recipe import TrainingRecipeFactory
from llavakd.utils import *
from llavakd.model import *
from llavakd.data.dataset import make_supervised_data_module

from llavakd.data.dataset import LazySupervisedDataset, DataCollatorForSupervisedDataset
from llavakd.train.tinyllava_distill_trainer import LengthGroupedSampler

from torch.utils.data import DataLoader

import tqdm
import numpy as np
import time
import datetime

from ml_dtypes import bfloat16, float8_e3m4, float8_e5m2 # bfloat for numpy

IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')

import wandb
os.environ["WANDB_API_KEY"] = 'YOUR API WANDB KEY'
os.environ["WANDB_MODE"] = "offline"

wandb.disabled = True # nikgkal

import warnings

import gc

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

def save_teacher_logis():

    start_time = time.monotonic()

    # load teacher model
    teacher_dir = './pretrained_checkpoints/LLaVA_KD_ckpts/TinyLLaVA-Qwen2.5-3B-SigLIP'
    teacher_model_config = TinyLlavaConfig.from_pretrained(teacher_dir)
    #if teacher_model_config.output_hidden_states is False:
    #    print(f"Setting teacher output_hidden_states to True")
    #    teacher_model_config.output_hidden_states = True

    teacher_model = LLaVAKD(teacher_model_config)

    pre_vision_tower_path = os.path.join(teacher_dir, 'vision_tower/pytorch_model.bin')
    print(f"Teacher Loading vision from {pre_vision_tower_path}")
    pre_vision = torch.load(pre_vision_tower_path, map_location='cpu')
    teacher_model.vision_tower._vision_tower.load_state_dict(pre_vision)

    pre_connector_path = os.path.join(teacher_dir, 'connector/pytorch_model.bin')
    print(f"Teacher Loading connector from {pre_connector_path}")
    pre_connector = torch.load(pre_connector_path, map_location='cpu')
    teacher_model.connector._connector.load_state_dict(get_conncet_w(pre_connector, '_connector'))

    pre_llm_path = os.path.join(teacher_dir, 'language_model')
    print(f"Teacher Loading llm from {pre_llm_path}")
    teacher_model.language_model = teacher_model.language_model.from_pretrained(pre_llm_path, device_map="cpu")

    context_len = getattr(teacher_model.config, 'max_sequence_length', 2048)

    teacher_model.eval()
    teacher_model.to(device='cuda')
    teacher_model.requires_grad_(False)

    #data_arguments = teacher_model.config

    # prepare the evauation dataset
    data_arguments = DataArguments()
    data_arguments.data_path =  '/home/gkalelis/Documents/py/datasets/LlavaPtSft/text_files/llava_v1_5_mix665k_20sample_noim.json'
    data_arguments.lazy_preprocess = True
    data_arguments.is_multimodal = True
    data_arguments.image_folder = '/home/gkalelis/Documents/py/datasets/LlavaPtSft'
    data_arguments.image_aspect_ratio = 'square'
    data_arguments.conv_version = 'qwen2_base'
    data_arguments.image_size_resize =768

    data_arguments.dataloader_num_workers = 2
    data_arguments.pin_memory = True

    #lazy_preprocess = True
    per_device_eval_batch_size = 1
    world_size = 1 # https://stackoverflow.com/questions/58271635/in-distributed-computing-what-are-world-size-and-rank

    # get image processor and tokenizer from student
    #teacher_model_image_processor = teacher_model.vision_tower._image_processor # this is not in the data arguments
    #teacher_model_tokenizer = teacher_model.tokenizer

    from transformers import  AutoImageProcessor, AutoTokenizer #Qwen2Tokenizer,  SiglipImageProcessor

    datasets_root = '/home/gkalelis/Documents/py/datasets/'
    teacher_name = 'tiny-llava-Qwen2.5-3B-siglip-so400m-patch14-384-qwen2_base-finetune'
    teacher_lm_head_name = 'lm_head_weights'
    dataset_name = 'LlavaPtSft'
    subdataset_name = 'llava_v1_5_mix665k'
    teacher_hidden_states_name = 'TeacherHiddenStates'  # 'TeachersLogits'
    output_data_path = os.path.join(datasets_root, teacher_hidden_states_name, teacher_name, dataset_name, subdataset_name)
    output_lm_head_path = os.path.join(datasets_root, teacher_hidden_states_name, teacher_name, teacher_lm_head_name)

    student_model_image_processor = AutoImageProcessor.from_pretrained('studentImageProcessor')
    student_model_tokenizer = AutoTokenizer.from_pretrained("studentTokenizer")

    data_arguments.image_processor = student_model_image_processor

    data_arguments.teacher_hidden_states_path = output_data_path
    eval_dataset = LazySupervisedDataset(tokenizer=student_model_tokenizer,
                                          data_path=data_arguments.data_path,
                                          data_args=data_arguments)

    eval_collator = DataCollatorForSupervisedDataset(tokenizer=student_model_tokenizer)

    eval_sampler = LengthGroupedSampler(
        # self.args.train_batch_size * self.args.gradient_accumulation_steps, # TODO: seems that we should not have gradient_accumulation_steps
        per_device_eval_batch_size,
        world_size= world_size,
        lengths=eval_dataset.modality_lengths,
        group_by_modality=True,
    )

    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, collate_fn=eval_collator,
                                 num_workers=data_arguments.dataloader_num_workers, pin_memory= data_arguments.pin_memory) # , persistent_workers=True


    if not os.path.exists(output_lm_head_path):
        print(f"Creating directory: {output_lm_head_path}")
        os.makedirs(output_lm_head_path)

    # get and save LM head
    teacher_model_lm_head = teacher_model.language_model.get_output_embeddings()
    torch.save(teacher_model_lm_head.state_dict(), os.path.join(output_lm_head_path, 'lm_head_state_dict.pth'))

    # (hidden_size, vocab_size) = (2048, 151936)
    #teacher_model_lm_head = nn.Linear(teacher_model.language_model.config.hidden_size, teacher_model.language_model.config.vocab_size, bias=False)
    #teacher_model_lm_head.load_state_dict(torch.load(output_lm_head_path))


    #for step, batch in enumerate(tqdm(eval_dataloader)):
    for step, batch in enumerate(eval_dataloader):
        #batch = {key: val.cuda() for key, val in batch.items()}
        #batch = {key: batch[key].cuda() for key in ("input_ids", "labels", "attention_mask", "images")}

        #print(f"Current step: {step}")

        sample_ids = batch.pop("ids")
        batch = {key: val.cuda() for key, val in batch.items()}
        sample_id_str = sample_ids[0]
        sample_id_type = type(sample_id_str)
        if type(sample_id_str) is int:
            sample_id_str = str(sample_id_str)
            #print(f"Converting sample from integer to string: {sample_id_str}")

        if type(sample_id_str) is not str:
            sys.exit(f"Exiting; sample id is of type: {sample_id_type}")

        filepath = os.path.join(output_data_path, sample_id_str)
        if os.path.isfile(filepath):
            print(f"File already exists: {filepath}; skipping...")
            continue

        dirname = os.path.dirname(filepath) # discard filename
        if not os.path.exists(dirname):
            print(f"Creating directory: {dirname}")
            os.makedirs(dirname)

        #print(f"Processing sample id: {sample_id_str}")

        batch.update({'output_hidden_states': True})

        with torch.no_grad():
            teacher_outputs, multimodal_teacher_labels, _, _, _, _, _ = teacher_model(**batch)

            # process logits
            _, _, voc_size = teacher_outputs['logits'].shape
            shift_teacher_logits = teacher_outputs['logits'][..., :-1, :].contiguous() # drop last element of sequence
            shift_teacher_labels = multimodal_teacher_labels[..., 1:].contiguous() # drop label of first element in sequence
            shift_teacher_labels = shift_teacher_labels.view(-1) # flatten the labels: # put batch sample sequence labels next to each other
            teacher_mask = torch.ne(shift_teacher_labels, -100).int()

            masked_teacher_logits = shift_teacher_logits.view(-1, voc_size)[teacher_mask.bool()] # put batch sample sequences next to each other
            masked_teacher_logits_np = masked_teacher_logits.detach().cpu().numpy()
            #masked_teacher_logits_npbf16 = masked_teacher_logits_np.astype(bfloat16)

            # process penultimate stage hidden states
            teacher_penultimate_stage_hidden_states = teacher_outputs.hidden_states[-1]
            _, _, embed_size = teacher_penultimate_stage_hidden_states.shape
            with torch.no_grad():
                teacher_logits_from_head = teacher_model.language_model.lm_head(teacher_penultimate_stage_hidden_states)
            teacher_logits_from_head= teacher_logits_from_head.float()
            if torch.equal(teacher_outputs['logits'], teacher_logits_from_head) is False:
                print(f"teacher logits from output and head are NOT equal ")


            shift_teacher_penultimate_stage_hidden_states = teacher_penultimate_stage_hidden_states[..., :-1, :].contiguous() # drop last element of sequence
            masked_shift_teacher_penultimate_stage_hidden_states = shift_teacher_penultimate_stage_hidden_states.view(-1, embed_size)[teacher_mask.bool()]
            masked_shift_teacher_logits_from_head = teacher_model.language_model.lm_head(masked_shift_teacher_penultimate_stage_hidden_states)
            masked_shift_teacher_logits_from_head = masked_shift_teacher_logits_from_head.float()
            if torch.equal(masked_teacher_logits, masked_shift_teacher_logits_from_head) is False:
                numNotEqElms = torch.sum(~torch.isclose(masked_teacher_logits, masked_shift_teacher_logits_from_head, atol=1e-3))
                #masked_teacher_logits_neq = masked_teacher_logits[~torch.isclose(masked_teacher_logits, masked_shift_teacher_logits_from_head)]
                #masked_shift_teacher_logits_from_head_neq = masked_shift_teacher_logits_from_head[~torch.isclose(masked_teacher_logits, masked_shift_teacher_logits_from_head)]
                #torch.equal(masked_teacher_logits_neq, masked_shift_teacher_logits_from_head_neq)
                if numNotEqElms > 10:
                    warnings.warn(f"Masked teacher logits from output and head are NOT equal; number of not equal elements are: {numNotEqElms}")
                #print(f"Masked teacher logits from output and head are NOT equal but close; number of not equal elements are: {numNotEqElms}")

            masked_shift_teacher_penultimate_stage_hidden_states_np = masked_shift_teacher_penultimate_stage_hidden_states.detach().cpu().numpy()
            #masked_teacher_logits_npbf16 = masked_teacher_logits_np.astype(bfloat16)

            #print(f"Saving file: {filepath}")
            #np.savez_compressed(filepath, masked_shift_teacher_penultimate_stage_hidden_states_np)
            #np.savez_compressed(filepath, masked_teacher_logits_npbf16)
            np.save(filepath, masked_shift_teacher_penultimate_stage_hidden_states_np)

            if step % 10 == 0:
                curr_time = time.monotonic()
                curr_ellapsed_time = curr_time - start_time
                curr_ellapsed_timeformatted_time = str(datetime.timedelta(seconds=curr_ellapsed_time))
                print(f"Current step:{step}; elapsed time:{curr_ellapsed_timeformatted_time}")
                #torch.cuda.memory_summary()
                gc.collect()
                torch.cuda.empty_cache()


            #teacher_outputs["loss"].detach() # trying releasing memory from cuda, but un successfull
            #multimodal_teacher_labels.detach()
            # see this: https://stackoverflow.com/questions/70602796/pytorch-gpu-memory-keeps-increasing-with-every-batch
            # del teacher_outputs["loss"]
            # del multimodal_teacher_labels
            #for key, val in teacher_outputs.items():
            #    del teacher_outputs[key]
            # https://medium.com/@raghadalghonaim/memory-leakage-with-pytorch-23f15203faa4
            # use all under torch.no_grad!!!

    end_time = time.monotonic()
    ellapsed_time = end_time - start_time
    formatted_time = str(datetime.timedelta(seconds=ellapsed_time))
    print("Time in days, hours, minutes, and seconds:", formatted_time)


if __name__ == "__main__":
    transformers.set_seed(1234)
    save_teacher_logis()