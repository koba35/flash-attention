import math
import json
import re
from pathlib import Path
import os
from collections import OrderedDict

from functools import partial
import torch
import torch.nn.functional as F

from transformers import GPT2Config, LlamaConfig, LlamaForCausalLM


def remap_state_dict_meta_llama(state_dict, config):
    def key_mapping_layers(key):
        key = re.sub(r'^model.', r'transformer.', key)
        return key

    state_dict = OrderedDict((key_mapping_layers(k), v) for k, v in state_dict.items())

    def key_mapping_emb(key):
        return re.sub(r'^transformer.embed_tokens.', 'transformer.embeddings.word_embeddings.', key)

    state_dict = OrderedDict((key_mapping_emb(k), v) for k, v in state_dict.items())
    word_embeddings = state_dict.pop('transformer.embeddings.word_embeddings.weight')
    state_dict['transformer.embeddings.word_embeddings.weight'] = word_embeddings

    if getattr(config, 'tie_word_embeddings'):
        state_dict['lm_head.weight'] = state_dict['transformer.embeddings.word_embeddings.weight']

    # LayerNorm
    def key_mapping_ln(key):
        key = re.sub(r'^transformer.norm.', r'transformer.ln_f.', key)
        key = re.sub(r'^transformer.layers.(\d+).input_layernorm.', r'transformer.layers.\1.norm1.', key)
        key = re.sub(r'^transformer.layers.(\d+).post_attention_layernorm.', r'transformer.layers.\1.norm2.', key)
        return key

    state_dict = OrderedDict((key_mapping_ln(k), v) for k, v in state_dict.items())

    # MLP
    for l in range(config.num_hidden_layers):
        w1 = state_dict.pop(f'transformer.layers.{l}.mlp.gate_proj.weight')
        w3 = state_dict.pop(f'transformer.layers.{l}.mlp.up_proj.weight')
        # Our ordering is different
        state_dict[f'transformer.layers.{l}.mlp.fc1.weight'] = torch.cat([w3, w1], dim=0)

    def key_mapping_mlp(key):
        return re.sub(r'^transformer.layers.(\d+).mlp.down_proj.',
                      r'transformer.layers.\1.mlp.fc2.', key)

    state_dict = OrderedDict((key_mapping_mlp(k), v) for k, v in state_dict.items())
    n_heads = config.num_attention_heads
    dim = config.hidden_size

    # Attention

    def unpermute(w):
        return w.view(n_heads, 2, dim // n_heads // 2, dim).transpose(2, 1).reshape(dim, dim)

    for l in range(config.num_hidden_layers):
        Wq = state_dict.pop(f'transformer.layers.{l}.self_attn.q_proj.weight')
        Wk = state_dict.pop(f'transformer.layers.{l}.self_attn.k_proj.weight')
        Wq = unpermute(Wq)
        Wk = unpermute(Wk)
        Wv = state_dict.pop(f'transformer.layers.{l}.self_attn.v_proj.weight')
        state_dict[f'transformer.layers.{l}.mixer.Wqkv.weight'] = torch.cat([Wq, Wk, Wv], dim=0)
        # We don't store these
        state_dict.pop(f'transformer.layers.{l}.attention.inner_attention.rope.freqs', None)

    def key_mapping_attn(key):
        return re.sub(r'^transformer.layers.(\d+).self_attn.o_proj.',
                      r'transformer.layers.\1.mixer.out_proj.', key)

    state_dict = OrderedDict((key_mapping_attn(k), v) for k, v in state_dict.items())

    def key_mapping_rotary(key):
        return re.sub(r'^transformer.layers.(\d+).self_attn.rotary_emb.inv_freq',
                      r'transformer.layers.\1.mixer.rotary_emb.inv_freq', key)

    state_dict = OrderedDict((key_mapping_rotary(k), v) for k, v in state_dict.items())
    return state_dict

def config_from_checkpoint(checkpoint_path: str) -> LlamaConfig:
    """Load a LlamaConfig from a checkpoint path."""
    with open(Path(checkpoint_path) / 'config.json') as f:
        params = json.load(f)
    params['intermediate_size'] = None
    config = LlamaConfig(**params)
    return config


def load_state_dict_from_hf(destination):
    WEIGHTS_INDEX_NAME = "pytorch_model.bin.index.json"
    index_file = os.path.join(destination, WEIGHTS_INDEX_NAME)
    with open(index_file, "r", encoding="utf-8") as f:
        index = json.load(f)

    shard_files = list(set(index["weight_map"].values()))
    loader = partial(torch.load, map_location="cpu")
    state_dict = {}

    for shard_file in sorted(shard_files):
        print(shard_file)
        tmp_dict = loader(os.path.join(destination, shard_file))
        for k, v in tmp_dict.items():
            state_dict[k] = v
        del tmp_dict

    return state_dict

def llama_config_to_gpt2_config(llama_config: LlamaConfig) -> GPT2Config:
    return GPT2Config(
        vocab_size=llama_config.vocab_size,
        n_positions=0,  # No absolute position embedding
        n_embd=llama_config.hidden_size,
        n_layer=llama_config.num_hidden_layers,
        n_head=llama_config.num_attention_heads,
        n_inner=llama_config.intermediate_size,
        activation_function='swiglu',  # Hardcode since HF calls it 'silu'
        # Llama doesn't have dropout, idk if it's because they only release the inference code
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        layer_norm_epsilon=llama_config.rms_norm_eps,
        initializer_range=llama_config.initializer_range,
        bos_token_id=llama_config.bos_token_id,
        eos_token_id=llama_config.eos_token_id,
        # These are new arguments not in the original GPT2Config
        pad_token_id=llama_config.pad_token_id,  # Idk if this does anything
        rms_norm=True,
        rotary_emb_fraction=1.0,
        rotary_emb_interleaved=True,
        tie_word_embeddings=False,
        qkv_proj_bias=False,
        out_proj_bias=False,
        mlp_fc1_bias=False,
        mlp_fc2_bias=False,
    )
