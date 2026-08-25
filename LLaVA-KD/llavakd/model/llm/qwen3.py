from transformers import Qwen3ForCausalLM, AutoTokenizer, AutoModelForCausalLM
# from .modeling_qwen2 import Qwen2ForCausalLM
# from transformers import AutoTokenizer

from . import register_llm

@register_llm('qwen3')
def return_phiclass():
    def tokenizer_and_post_load(tokenizer):
        tokenizer.unk_token = tokenizer.pad_token
#        tokenizer.pad_token = tokenizer.unk_token
        # change eos_token ('<|im_end|>') to pad_token ('<|endoftext|>') - qwen has been trained with '<|endoftext|>' as final token
        # see https://github.com/huggingface/sentence-transformers/issues/3563
        # https://kaitchup.substack.com/p/qwen3-when-im_end-suddenly-becomes
        # https://huggingface.co/blog/qgallouedec/gotchas-in-tokenizer-behavior
        # tokenizer.eos_token_id == tokenizer.encode('<|im_end|>') == 151645
        # tokenizer.pad_token_id == tokenizer.encode('<|endoftext|>') == 151643
        tokenizer.eos_token = tokenizer.pad_token
        #tokenizer.eos_token_id = tokenizer.pad_token_id # not necessary, runs the above as dict
        return tokenizer
    return Qwen3ForCausalLM, (AutoTokenizer, tokenizer_and_post_load)
    #return AutoModelForCausalLM, (AutoTokenizer, tokenizer_and_post_load) # gkaln: support for Qwen3 as well