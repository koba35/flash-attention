# Copyright (c) 2023, Tri Dao.

import logging
from collections.abc import Sequence
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput

from flash_attn.models.llama_custom import FlashLlamaConfig
from flash_attn.modules.block import Block
from flash_attn.modules.mha import MHA, ParallelMHA
from flash_attn.modules.mlp import Mlp, GatedMlp, FusedMLP, ParallelFusedMLP
from flash_attn.ops.activations import sqrelu_fwd

try:
    from flash_attn.ops.fused_dense import ColumnParallelLinear
except ImportError:
    ColumnParallelLinear = None

try:
    from flash_attn.ops.layer_norm import dropout_add_layer_norm
except ImportError:
    dropout_add_layer_norm = None

try:
    from flash_attn.ops.layer_norm import dropout_add_layer_norm_parallel_residual
except ImportError:
    dropout_add_layer_norm_parallel_residual = None

try:
    from flash_attn.ops.rms_norm import RMSNorm, dropout_add_rms_norm
except ImportError:
    RMSNorm, dropout_add_rms_norm = None, None

try:
    from flash_attn.ops.rms_norm import dropout_add_rms_norm_parallel_residual
except ImportError:
    dropout_add_rms_norm_parallel_residual = None

try:
    from flash_attn.ops.triton.mlp import FusedDenseSqreluDense
except ImportError:
    FusedDenseSqreluDense = None

logger = logging.getLogger(__name__)


class MyRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, device=None, dtype=None):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, **factory_kwargs))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


def create_mixer_cls(config, layer_idx=None, process_group=None, device=None, dtype=None):
    factory_kwargs = {'device': device, 'dtype': dtype}
    head_dim = getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads)
    softmax_scale = 1.0 if not config.scale_attn_weights else head_dim ** (-0.5)
    if config.scale_attn_by_inverse_layer_idx:
        assert layer_idx is not None
        softmax_scale /= float(layer_idx + 1)
    dwconv = getattr(config, 'attn_dwconv', False)
    if dwconv:
        assert process_group is None, 'TensorParallel MHA does not support dwconv yet'
    qkv_proj_bias = getattr(config, 'qkv_proj_bias', True)
    out_proj_bias = getattr(config, 'out_proj_bias', True)
    rotary_emb_dim = int(getattr(config, 'rotary_emb_fraction', 0.0) * head_dim)
    rotary_emb_scale_base = getattr(config, 'rotary_emb_scale_base', None)
    rotary_emb_interleaved = getattr(config, 'rotary_emb_interleaved', False)
    use_flash_attn = getattr(config, 'use_flash_attn', False)
    fused_bias_fc = getattr(config, 'fused_bias_fc', False)
    if not fused_bias_fc:
        assert process_group is None, 'TensorParallel MHA requires fused_bias_fc'
    mha_cls = MHA if process_group is None else ParallelMHA
    serial_kwargs = ({'fused_bias_fc': fused_bias_fc, 'dwconv': dwconv}
                     if process_group is None else {})
    parallel_kwargs = ({'process_group': process_group,
                        'sequence_parallel': getattr(config, 'sequence_parallel', True)}
                       if process_group is not None else {})
    mixer_cls = partial(mha_cls, num_heads=config.num_attention_heads,
                        qkv_proj_bias=qkv_proj_bias, out_proj_bias=out_proj_bias,
                        dropout=config.attn_pdrop,
                        softmax_scale=softmax_scale, causal=True, layer_idx=layer_idx,
                        rotary_emb_dim=rotary_emb_dim, rotary_emb_scale_base=rotary_emb_scale_base,
                        rotary_emb_interleaved=rotary_emb_interleaved,
                        use_flash_attn=use_flash_attn,
                        **serial_kwargs, **parallel_kwargs, **factory_kwargs)
    return mixer_cls


def create_mlp_cls(config, layer_idx=None, process_group=None, device=None, dtype=None):
    factory_kwargs = {'device': device, 'dtype': dtype}
    mlp_fc1_bias = getattr(config, 'mlp_fc1_bias', True)
    mlp_fc2_bias = getattr(config, 'mlp_fc2_bias', True)
    fused_mlp = getattr(config, 'fused_mlp', False)
    if fused_mlp:
        assert config.activation_function in ['gelu_new', 'gelu_fast', 'gelu_approx', 'relu', 'sqrelu']
    fused_dense_sqrelu_dense = getattr(config, 'fused_dense_sqrelu_dense', False)
    if fused_dense_sqrelu_dense:
        assert config.activation_function == 'sqrelu', ('fused_dense_sqrelu_dense only '
                                                        'supports approximate activation_function sqrelu')
    assert not (fused_dense_sqrelu_dense and fused_mlp)
    if process_group is not None:
        assert fused_mlp, 'Tensor Parallel is only implemented for FusedMLP'
    if not fused_mlp and not fused_dense_sqrelu_dense:
        assert config.activation_function in ['gelu_new', 'gelu_fast', 'gelu_approx', 'relu',
                                              'sqrelu', 'glu', 'swiglu', 'geglu']
        if config.activation_function in ['glu', 'swiglu', 'geglu']:
            activation = (F.sigmoid if config.activation_function == 'glu'
                          else (F.silu if config.activation_function == 'swiglu'
                                else F.gelu))
            mlp_cls = partial(GatedMlp, hidden_features=config.n_inner, activation=activation,
                              bias1=mlp_fc1_bias, bias2=mlp_fc2_bias, **factory_kwargs)
        else:
            if config.activation_function == 'relu':
                activation = partial(F.relu, inplace=True)
            elif config.activation_function == 'sqrelu':
                activation = sqrelu_fwd
            else:
                approximate = ('tanh' if config.activation_function
                                         in ['gelu_new', 'gelu_fast', 'gelu_approx'] else 'none')
                activation = partial(F.gelu, approximate=approximate)
            mlp_cls = partial(Mlp, hidden_features=config.n_inner, activation=activation,
                              bias1=mlp_fc1_bias, bias2=mlp_fc2_bias, **factory_kwargs)
    else:
        mlp_checkpoint_lvl = getattr(config, 'mlp_checkpoint_lvl', 0)
        # mlp_checkpoint_lvl could be a list, which contains the checkpoint_lvl for each layer
        if isinstance(mlp_checkpoint_lvl, Sequence):
            assert layer_idx is not None
            mlp_checkpoint_lvl = mlp_checkpoint_lvl[layer_idx]
        if fused_mlp:
            if FusedMLP is None:
                raise ImportError('fused_dense is not installed')
            activation = ('gelu_approx' if config.activation_function
                                           in ['gelu_new', 'gelu_fast', 'gelu_approx'] else config.activation_function)
            mlp_cls = FusedMLP if process_group is None else ParallelFusedMLP
            parallel_kwargs = ({'process_group': process_group,
                                'sequence_parallel': getattr(config, 'sequence_parallel', True)}
                               if process_group is not None else {})
            mlp_cls = partial(mlp_cls, hidden_features=config.n_inner, activation=activation,
                              checkpoint_lvl=mlp_checkpoint_lvl,
                              bias1=mlp_fc1_bias, bias2=mlp_fc2_bias,
                              **parallel_kwargs, **factory_kwargs)
        elif fused_dense_sqrelu_dense:
            assert FusedDenseSqreluDense is not None
            mlp_cls = partial(FusedDenseSqreluDense, hidden_features=config.n_inner,
                              checkpoint_lvl=mlp_checkpoint_lvl, **factory_kwargs)
        else:
            raise RuntimeError('MLP type not supported')
    return mlp_cls


def create_block(config, layer_idx=None):
    mixer_cls = create_mixer_cls(config, layer_idx)
    mlp_cls = create_mlp_cls(config, layer_idx)
    norm_cls = partial(MyRMSNorm, eps=config.layer_norm_epsilon)
    # TD [2022-07-30]: Force residual in fp32, seems to make fp16 training more stable
    residual_in_fp32 = getattr(config, 'residual_in_fp32', False)
    resid_dropout1 = config.resid_pdrop if layer_idx is None or layer_idx > 0 else config.embd_pdrop
    prenorm = getattr(config, 'prenorm', True)

    block = Block(
        config.hidden_size, mixer_cls, mlp_cls, norm_cls=norm_cls,
        prenorm=prenorm, resid_dropout1=resid_dropout1, resid_dropout2=config.resid_pdrop,
        fused_dropout_add_ln=getattr(config, 'fused_dropout_add_ln', False),
        residual_in_fp32=residual_in_fp32
    )
    block.layer_idx = layer_idx
    return block


class FlashLlamaPreTrainedModel(PreTrainedModel):
    config_class = FlashLlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Block"]

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, FlashLlamaModel):
            module.gradient_checkpointing = value

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class FlashLlamaModel(FlashLlamaPreTrainedModel):

    def __init__(self, config: FlashLlamaConfig):
        super().__init__(config)
        assert config.activation_function in ['gelu', 'gelu_new', 'gelu_fast', 'gelu_approx',
                                              'relu', 'sqrelu', 'glu', 'swiglu', 'geglu']

        # TD [2022-07-30]: Force residual in fp32, seems to make fp16 training more stable
        self.residual_in_fp32 = getattr(config, 'residual_in_fp32', False)
        self.padding_idx = config.pad_token_id
        # These 2 options are for OPT-350m
        self.prenorm = getattr(config, 'prenorm', True)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        self.layers = nn.ModuleList([create_block(config, layer_idx=i)
                                     for i in range(config.num_hidden_layers)])

        self.fused_dropout_add_ln = getattr(config, 'fused_dropout_add_ln', False)
        if self.fused_dropout_add_ln:
            if ((not self.parallel_block and dropout_add_layer_norm is None)
                    or (self.parallel_block and dropout_add_layer_norm_parallel_residual is None)):
                raise ImportError('dropout_layer_norm is not installed')
        if self.prenorm:
            self.drop_f = nn.Dropout(config.resid_pdrop)
            self.ln_f = MyRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(self, input_ids, **kwargs):

        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:

                if self.prenorm:
                    hidden_states, residual = torch.utils.checkpoint.checkpoint(layer, hidden_states, residual, {})
                else:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            # None for past_key_value
                            return module(*inputs)

                        return custom_forward

                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(layer),
                        {}
                    )
            else:
                if self.prenorm:
                    hidden_states, residual = layer(hidden_states, residual,
                                                    mixer_kwargs={})
                else:
                    hidden_states = layer(hidden_states, mixer_kwargs={})

        if self.prenorm:
            if not self.fused_dropout_add_ln:
                dropped = self.drop_f(hidden_states)
                residual = (dropped + residual) if residual is not None else dropped
                hidden_states = self.ln_f(residual.to(dtype=self.ln_f.weight.dtype))
            else:
                hidden_states = dropout_add_layer_norm(
                    hidden_states, residual, self.ln_f.weight, self.ln_f.bias,
                    self.drop_f.p if self.training else 0.0, self.ln_f.eps, prenorm=False,
                    residual_in_fp32=self.residual_in_fp32
                            )

        return hidden_states


class FlashLlamaForCausalLM(FlashLlamaPreTrainedModel):
    def __init__(self, config: FlashLlamaConfig):
        super().__init__(config)

        self.model = FlashLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def forward(self, input_ids, labels=None, **kwargs):
        hidden_states = self.model(input_ids)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        return CausalLMOutput(
            loss=loss,
            logits=logits,
        )

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        return {"input_ids": input_ids}
