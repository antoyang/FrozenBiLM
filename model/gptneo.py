# coding=utf-8
# Copyright 2021 The Eleuther AI and HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch GPT Neo model."""


import os
from typing import Optional, Tuple, Union, Iterable, Callable
import inspect
import torch.distributed as dist

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers import GPTNeoConfig

from transformers import LogitsProcessorList, StoppingCriteriaList
from transformers import Constraint
from transformers.generation_utils import (
    GreedySearchOutput,
    SampleOutput,
    BeamSearchOutput,
    BeamSampleOutput,
    GreedySearchEncoderDecoderOutput,
    GreedySearchDecoderOnlyOutput,
    validate_stopping_criteria,
)
from transformers.generation_beam_search import (
    BeamScorer,
    BeamSearchScorer,
    ConstrainedBeamSearchScorer,
)

_CONFIG_FOR_DOC = "GPTNeoConfig"
_TOKENIZER_FOR_DOC = "GPT2Tokenizer"

GPT_NEO_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "EleutherAI/gpt-neo-1.3B",
    # See all GPTNeo models at https://huggingface.co/models?filter=gpt_neo
]

_CHECKPOINT_FOR_DOC = "EleutherAI/gpt-neo-1.3B"

import torch.nn.functional as F
from typing import Union, List


def load_tf_weights_in_gpt_neo(model, config, gpt_neo_checkpoint_path):
    """Load tf checkpoints in a pytorch model"""
    try:
        import re

        import tensorflow as tf
    except ImportError:
        print(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(gpt_neo_checkpoint_path)
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        if "global_step" not in name and "adam" not in name:
            array = tf.train.load_variable(tf_path, name)
            array = tf.dtypes.cast(array.squeeze(), tf.float32).numpy()
            name = name.replace("attn/q", "attn/attention/q_proj/w")
            name = name.replace("attn/k", "attn/attention/k_proj/w")
            name = name.replace("attn/v", "attn/attention/v_proj/w")
            name = name.replace("attn/o", "attn/attention/out_proj/w")
            name = name.replace("norm_1", "ln_1")
            name = name.replace("norm_2", "ln_2")
            name = name.replace(
                "attn/compute_output_bias/o_b", "attn/attention/out_proj/b"
            )
            name = name.replace("conv1d_main/c_fc/kernel", "c_fc/w")
            name = name.replace("conv1d_main/c_fc/bias", "c_fc/b")
            name = name.replace("conv1d_main/c_proj/kernel", "c_proj/w")
            name = name.replace("conv1d_main/c_proj/bias", "c_proj/b")

            names.append(name)
            arrays.append(array)

    for name, array in zip(names, arrays):
        name = name[5:]  # skip "gpt2/"
        name = name.split("/")
        pointer = model.transformer
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+\d+", m_name):
                scope_names = re.split(r"(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == "w" or scope_names[0] == "g":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "b":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "wpe" or scope_names[0] == "wte":
                pointer = getattr(pointer, scope_names[0])
                pointer = getattr(pointer, "weight")
            else:
                pointer = getattr(pointer, scope_names[0])
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]

        if name[-1] == "w" and name[-2] in [
            "out_proj",
            "k_proj",
            "q_proj",
            "v_proj",
            "c_proj",
            "c_fc",
        ]:
            array = array.transpose()

        if name == ["wte"]:
            # if vocab is padded, then trim off the padding embeddings
            array = array[: config.vocab_size]

        if pointer.shape != array.shape:
            raise ValueError(
                f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched {name}"
            )

        print(f"Initialize PyTorch weight {name}")
        pointer.data = torch.from_numpy(array)

    # init the final linear layer using word embeddings
    embs = model.transformer.wte.weight
    lin = nn.Linear(embs.size()[1], embs.size()[0], bias=False)
    lin.weight = embs
    model.set_output_embeddings(lin)
    return model


class GPTNeoSelfAttention(nn.Module):
    def __init__(self, config, attention_type):
        super().__init__()

        max_positions = config.max_position_embeddings
        bias = torch.tril(
            torch.ones((max_positions, max_positions), dtype=torch.uint8)
        ).view(1, 1, max_positions, max_positions)

        # local causal self attention is a sliding window where each token can only attend to the previous
        # window_size tokens. This is implemented by updating the causal mask such that for each token
        # all other tokens are masked except the previous window_size tokens.
        if attention_type == "local":
            bias = torch.bitwise_xor(bias, torch.tril(bias, -config.window_size))

        self.register_buffer("bias", bias)
        self.register_buffer("masked_bias", torch.tensor(-1e9))

        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.resid_dropout)

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
            )

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        # Keep the attention weights computation in fp32 to avoid overflow issues
        query = query.to(torch.float32)
        key = key.to(torch.float32)

        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.bias[
            :, :, key_length - query_length : key_length, :key_length
        ].bool()
        attn_weights = torch.where(
            causal_mask, attn_weights, self.masked_bias.to(attn_weights.dtype)
        )

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights.to(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        layer_past=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
    ):

        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        attn_output, attn_weights = self._attn(
            query, key, value, attention_mask, head_mask
        )

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


class GPTNeoAttention(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.layer_id = layer_id
        self.attention_layers = config.attention_layers
        self.attention_type = self.attention_layers[layer_id]

        if self.attention_type in ["global", "local"]:
            self.attention = GPTNeoSelfAttention(config, self.attention_type)
        else:
            raise NotImplementedError(
                "Only attn layer types 'global' and 'local' exist, but got `config.attention_layers`: "
                f"{config.attention_layers}. Select attn layer types from ['global', 'local'] only."
            )

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        return self.attention(
            hidden_states,
            attention_mask=attention_mask,
            layer_past=layer_past,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )


class GPTNeoMLP(nn.Module):
    def __init__(
        self, intermediate_size, config
    ):  # in MLP: intermediate_size= 4 * hidden_size
        super().__init__()
        embed_dim = config.hidden_size
        self.c_fc = nn.Linear(embed_dim, intermediate_size)
        self.c_proj = nn.Linear(intermediate_size, embed_dim)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_dropout)

    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class GPTNeoBlock(nn.Module):
    def __init__(
        self,
        config,
        layer_id,
    ):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = (
            config.intermediate_size
            if config.intermediate_size is not None
            else 4 * hidden_size
        )
        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPTNeoAttention(config, layer_id)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = GPTNeoMLP(inner_dim, config)

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)


class GPTNeoPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = GPTNeoConfig
    load_tf_weights = load_tf_weights_in_gpt_neo
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = True

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear,)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, GPTNeoModel):
            module.gradient_checkpointing = value


class GPTNeoModel(GPTNeoPreTrainedModel):
    def __init__(
        self,
        config,
        max_feats=10,
        features_dim=768,
        freeze_lm=False,
        ft_ln=False,
    ):
        super().__init__(config)

        self.embed_dim = config.hidden_size
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)
        self.drop = nn.Dropout(config.embed_dropout)
        self.h = nn.ModuleList(
            [
                GPTNeoBlock(
                    config,
                    layer_id=i,
                )
                for i in range(config.num_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        self.features_dim = features_dim
        if self.features_dim:
            self.linear_video = nn.Linear(features_dim, config.hidden_size)
        self.max_feats = max_feats
        if freeze_lm:
            for n, p in self.named_parameters():
                if not "linear_video" in n:
                    if ft_ln and "ln" in n:
                        continue
                    else:
                        p.requires_grad_(False)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_video_embedding(self, video):
        video = self.linear_video(video)
        return video

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        video=None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
            if self.features_dim and video is not None:
                video = self.get_video_embedding(video)
                inputs_embeds = torch.cat([video, inputs_embeds], 1)
                input_shape = inputs_embeds[:, :, 0].shape

        # Attention mask.
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x num_heads x N x N
        # head_mask has shape n_layer x batch x num_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)

        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if position_ids is None:
            position_ids = torch.arange(
                past_length,
                input_shape[-1] + past_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    print(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, use_cache, output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (
                    outputs[2 if use_cache else 1],
                )

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    presents,
                    all_hidden_states,
                    all_self_attentions,
                ]
                if v is not None
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class GPTNeoForCausalLM(GPTNeoPreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"h\.\d+\.attn\.masked_bias",
        r"lm_head\.weight",
        r"h\.\d+\.attn\.attention\.bias",
    ]
    _keys_to_ignore_on_save = [r"lm_head.weight"]

    def __init__(
        self,
        config,
        max_feats=10,
        features_dim=768,
        freeze_lm=False,
        freeze_mlm=False,
        ft_ln=False,
    ):
        """
        :param config: LM configuration
        :param max_feats: maximum number of frames used by the model
        :param features_dim: embedding dimension of the visual features, set = 0 for text-only mode
        :param freeze_lm: whether to freeze or not the language model (Transformer encoder + token embedder)
        :param ft_ln: whether to finetune or not the normalization layers
        """
        super().__init__(config)
        self.transformer = GPTNeoModel(
            config,
            max_feats,
            features_dim,
            freeze_lm,
            ft_ln,
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.features_dim = features_dim
        if freeze_mlm:
            for n, p in self.lm_head.named_parameters():
                if ft_ln and "ln" in n:
                    continue
                else:
                    p.requires_grad_(False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)
            video = None  # computations are saved
        else:
            video = kwargs.get("video", None)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "video": video,
        }

    def forward(
        self,
        video=None,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            video=video,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Compute loss in fp32 to match with mesh-tf version
            # https://github.com/EleutherAI/gpt-neo/blob/89ce74164da2fb16179106f54e2269b5da8db333/models/gpt2/gpt2.py#L179
            if (
                self.features_dim and video is not None
            ):  # ignore the label predictions for visual tokens
                video_shape = video[:, :, 0].size()
                video_labels = torch.tensor(
                    [[-100] * video_shape[1]] * video_shape[0],
                    dtype=torch.long,
                    device=labels.device,
                )
                labels = torch.cat([video_labels, labels], 1)
            lm_logits = lm_logits.to(torch.float32)

            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

            lm_logits = lm_logits.to(hidden_states.dtype)
            loss = loss.to(hidden_states.dtype)

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    @staticmethod
    def _reorder_cache(
        past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PretrainedModel.beam_search`] or
        [`~PretrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(
            tuple(
                past_state.index_select(0, beam_idx.to(past_state.device))
                for past_state in layer_past
            )
            for layer_past in past
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        do_sample: Optional[bool] = None,
        early_stopping: Optional[bool] = None,
        num_beams: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        typical_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        bad_words_ids: Optional[Iterable[int]] = None,
        bos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        length_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        encoder_no_repeat_ngram_size: Optional[int] = None,
        num_return_sequences: Optional[int] = None,
        max_time: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        decoder_start_token_id: Optional[int] = None,
        use_cache: Optional[bool] = None,
        num_beam_groups: Optional[int] = None,
        diversity_penalty: Optional[float] = None,
        prefix_allowed_tokens_fn: Optional[
            Callable[[int, torch.Tensor], List[int]]
        ] = None,
        logits_processor: Optional[LogitsProcessorList] = LogitsProcessorList(),
        stopping_criteria: Optional[StoppingCriteriaList] = StoppingCriteriaList(),
        constraints: Optional[List[Constraint]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        forced_bos_token_id: Optional[int] = None,
        forced_eos_token_id: Optional[int] = None,
        remove_invalid_values: Optional[bool] = None,
        synced_gpus: Optional[bool] = False,
        **model_kwargs,
    ) -> Union[
        GreedySearchOutput,
        SampleOutput,
        BeamSearchOutput,
        BeamSampleOutput,
        torch.LongTensor,
    ]:
        r"""
        Generates sequences for models with a language modeling head. The method currently supports greedy decoding,
        multinomial sampling, beam-search decoding, and beam-search multinomial sampling.

        Apart from `inputs`, all the arguments below will default to the value of the attribute of the same name inside
        the [`PretrainedConfig`] of the model. The default values indicated are the default values of those config.

        Most of these parameters are explained in more detail in [this blog
        post](https://huggingface.co/blog/how-to-generate).

        Parameters:
            inputs (`torch.Tensor` of shape `(batch_size, sequence_length)`, `(batch_size, sequence_length,
            feature_dim)` or `(batch_size, num_channels, height, width)`, *optional*):
                The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the
                method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs`
                should of in the format of `input_ids`. For encoder-decoder models *inputs* can represent any of
                `input_ids`, `input_values`, `input_features`, or `pixel_values`.
            max_length (`int`, *optional*, defaults to `model.config.max_length`):
                The maximum length of the sequence to be generated.
            max_new_tokens (`int`, *optional*, defaults to None):
                The maximum numbers of tokens to generate, ignore the current number of tokens. Use either
                `max_new_tokens` or `max_length` but not both, they serve the same purpose.
            min_length (`int`, *optional*, defaults to 10):
                The minimum length of the sequence to be generated.
            do_sample (`bool`, *optional*, defaults to `False`):
                Whether or not to use sampling ; use greedy decoding otherwise.
            early_stopping (`bool`, *optional*, defaults to `False`):
                Whether to stop the beam search when at least `num_beams` sentences are finished per batch or not.
            num_beams (`int`, *optional*, defaults to 1):
                Number of beams for beam search. 1 means no beam search.
            temperature (`float`, *optional*, defaults to 1.0):
                The value used to module the next token probabilities.
            top_k (`int`, *optional*, defaults to 50):
                The number of highest probability vocabulary tokens to keep for top-k-filtering.
            top_p (`float`, *optional*, defaults to 1.0):
                If set to float < 1, only the most probable tokens with probabilities that add up to `top_p` or higher
                are kept for generation.
            repetition_penalty (`float`, *optional*, defaults to 1.0):
                The parameter for repetition penalty. 1.0 means no penalty. See [this
                paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            bos_token_id (`int`, *optional*):
                The id of the *beginning-of-sequence* token.
            eos_token_id (`int`, *optional*):
                The id of the *end-of-sequence* token.
            length_penalty (`float`, *optional*, defaults to 1.0):
                Exponential penalty to the length. 1.0 means no penalty. Set to values < 1.0 in order to encourage the
                model to generate shorter sequences, to a value > 1.0 in order to encourage the model to produce longer
                sequences.
            no_repeat_ngram_size (`int`, *optional*, defaults to 0):
                If set to int > 0, all ngrams of that size can only occur once.
            encoder_no_repeat_ngram_size (`int`, *optional*, defaults to 0):
                If set to int > 0, all ngrams of that size that occur in the `encoder_input_ids` cannot occur in the
                `decoder_input_ids`.
            bad_words_ids(`List[List[int]]`, *optional*):
                List of token ids that are not allowed to be generated. In order to get the token ids of the words that
                should not appear in the generated text, use `tokenizer(bad_words, add_prefix_space=True,
                add_special_tokens=False).input_ids`.
            num_return_sequences(`int`, *optional*, defaults to 1):
                The number of independently computed returned sequences for each element in the batch.
            max_time(`float`, *optional*, defaults to None):
                The maximum amount of time you allow the computation to run for in seconds. generation will still
                finish the current pass after allocated time has been passed.
            attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values are in `[0, 1]`, 1 for tokens
                that are not masked, and 0 for masked tokens. If not provided, will default to a tensor the same shape
                as `input_ids` that masks the pad token. [What are attention masks?](../glossary#attention-mask)
            decoder_start_token_id (`int`, *optional*):
                If an encoder-decoder model starts decoding with a different token than *bos*, the id of that token.
            use_cache: (`bool`, *optional*, defaults to `True`):
                Whether or not the model should use the past last key/values attentions (if applicable to the model) to
                speed up decoding.
            num_beam_groups (`int`, *optional*, defaults to 1):
                Number of groups to divide `num_beams` into in order to ensure diversity among different groups of
                beams. [this paper](https://arxiv.org/pdf/1610.02424.pdf) for more details.
            diversity_penalty (`float`, *optional*, defaults to 0.0):
                This value is subtracted from a beam's score if it generates a token same as any beam from other group
                at a particular time. Note that `diversity_penalty` is only effective if `group beam search` is
                enabled.
            prefix_allowed_tokens_fn: (`Callable[[int, torch.Tensor], List[int]]`, *optional*):
                If provided, this function constraints the beam search to allowed tokens only at each step. If not
                provided no constraint is applied. This function takes 2 arguments: the batch ID `batch_id` and
                `input_ids`. It has to return a list with the allowed tokens for the next generation step conditioned
                on the batch ID `batch_id` and the previously generated tokens `inputs_ids`. This argument is useful
                for constrained generation conditioned on the prefix, as described in [Autoregressive Entity
                Retrieval](https://arxiv.org/abs/2010.00904).
            logits_processor (`LogitsProcessorList`, *optional*):
                 Custom logits processors that complement the default logits processors built from arguments and a
                 model's config. If a logit processor is passed that is already created with the arguments or a model's
                 config an error is thrown. This feature is intended for advanced users.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                 Custom stopping criteria that complement the default stopping criteria built from arguments and a
                 model's config. If a stopping criteria is passed that is already created with the arguments or a
                 model's config an error is thrown. This feature is intended for advanced users.
            constraints (`List[Constraint]`, *optional*):
                 Custom constraints that can be added to the generation to ensure that the output will contain the use
                 of certain tokens as defined by `Constraint` objects, in the most sensible way possible.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to `False`):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
            forced_bos_token_id (`int`, *optional*):
                The id of the token to force as the first generated token after the `decoder_start_token_id`. Useful
                for multilingual models like [mBART](../model_doc/mbart) where the first generated token needs to be
                the target language token.
            forced_eos_token_id (`int`, *optional*):
                The id of the token to force as the last generated token when `max_length` is reached.
            remove_invalid_values (`bool`, *optional*):
                Whether to remove possible *nan* and *inf* outputs of the model to prevent the generation method to
                crash. Note that using `remove_invalid_values` can slow down generation.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If the model
                is an encoder-decoder model, encoder specific kwargs should not be prefixed and decoder specific kwargs
                should be prefixed with *decoder_*.

        Return:
            [`~file_utils.ModelOutput`] or `torch.LongTensor`: A [`~file_utils.ModelOutput`] (if
            `return_dict_in_generate=True` or when `config.return_dict_in_generate=True`) or a `torch.FloatTensor`.

                If the model is *not* an encoder-decoder model (`model.config.is_encoder_decoder=False`), the possible
                [`~file_utils.ModelOutput`] types are:

                    - [`~generation_utils.GreedySearchDecoderOnlyOutput`],
                    - [`~generation_utils.SampleDecoderOnlyOutput`],
                    - [`~generation_utils.BeamSearchDecoderOnlyOutput`],
                    - [`~generation_utils.BeamSampleDecoderOnlyOutput`]

                If the model is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible
                [`~file_utils.ModelOutput`] types are:

                    - [`~generation_utils.GreedySearchEncoderDecoderOutput`],
                    - [`~generation_utils.SampleEncoderDecoderOutput`],
                    - [`~generation_utils.BeamSearchEncoderDecoderOutput`],
                    - [`~generation_utils.BeamSampleEncoderDecoderOutput`]

        """
        # 1. Set generation parameters if not already defined
        bos_token_id = (
            bos_token_id if bos_token_id is not None else self.config.bos_token_id
        )
        num_beams = num_beams if num_beams is not None else self.config.num_beams
        length_penalty = (
            length_penalty if length_penalty is not None else self.config.length_penalty
        )
        early_stopping = (
            early_stopping if early_stopping is not None else self.config.early_stopping
        )
        num_beam_groups = (
            num_beam_groups
            if num_beam_groups is not None
            else self.config.num_beam_groups
        )
        do_sample = do_sample if do_sample is not None else self.config.do_sample
        num_return_sequences = (
            num_return_sequences
            if num_return_sequences is not None
            else self.config.num_return_sequences
        )

        pad_token_id = (
            pad_token_id if pad_token_id is not None else self.config.pad_token_id
        )
        eos_token_id = (
            eos_token_id if eos_token_id is not None else self.config.eos_token_id
        )

        if eos_token_id is None and hasattr(self.config, "decoder"):
            eos_token_id = self.config.decoder.eos_token_id

        if pad_token_id is None and eos_token_id is not None:
            # special case if pad_token_id is not defined
            print(
                f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation."
            )
            pad_token_id = eos_token_id

        output_scores = (
            output_scores if output_scores is not None else self.config.output_scores
        )
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.config.return_dict_in_generate
        )

        # 2. Define model inputs
        # inputs_tensor has to be defined
        # model_input_name is defined if model-specific keyword input is passed
        # otherwise model_input_name is None
        # all model-specific keyword inputs are removed from `model_kwargs`
        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
            inputs, bos_token_id, model_kwargs
        )
        batch_size = inputs_tensor.shape[0]

        # 3. Define other model kwargs
        model_kwargs["output_attentions"] = output_attentions
        model_kwargs["output_hidden_states"] = output_hidden_states
        model_kwargs["use_cache"] = use_cache

        accepts_attention_mask = "attention_mask" in set(
            inspect.signature(self.forward).parameters.keys()
        )
        requires_attention_mask = "encoder_outputs" not in model_kwargs

        if (
            model_kwargs.get("attention_mask", None) is None
            and requires_attention_mask
            and accepts_attention_mask
        ):
            model_kwargs[
                "attention_mask"
            ] = self._prepare_attention_mask_for_generation(
                inputs_tensor, pad_token_id, eos_token_id
            )

        if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
            # if model is encoder decoder encoder_outputs are created
            # and added to `model_kwargs`
            model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name
            )

        # 4. Prepare `input_ids` which will be used for auto-regressive generation
        if self.config.is_encoder_decoder:
            input_ids = self._prepare_decoder_input_ids_for_generation(
                batch_size,
                decoder_start_token_id=decoder_start_token_id,
                bos_token_id=bos_token_id,
                model_kwargs=model_kwargs,
            )
        else:
            # if decoder-only then inputs_tensor has to be `input_ids`
            input_ids = inputs_tensor

        # 5. Prepare `max_length` depending on other stopping criteria
        # if `max_new_tokens` is passed, but not `max_length` -> set `max_length = max_new_tokens`
        if max_length is None and max_new_tokens is not None:
            max_length = max_new_tokens + input_ids.shape[-1]
        elif max_length is not None and max_new_tokens is not None:
            # Both are set, this is odd, raise a warning
            print(
                "Both `max_length` and `max_new_tokens` have been set "
                f"but they serve the same purpose. `max_length` {max_length} "
                f"will take priority over `max_new_tokens` {max_new_tokens}.",
                UserWarning,
            )
        # default to config if still None
        max_length = max_length if max_length is not None else self.config.max_length

        if input_ids.shape[-1] >= max_length:
            input_ids_string = (
                "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
            )
            print(
                f"Input length of {input_ids_string} is {input_ids.shape[-1]}, but ``max_length`` is set to {max_length}. "
                "This can lead to unexpected behavior. You should consider increasing ``config.max_length`` or ``max_length``."
            )

        # 6. determine generation mode
        is_constraint_gen_mode = constraints is not None
        is_greedy_gen_mode = (
            (num_beams == 1)
            and (num_beam_groups == 1)
            and do_sample is False
            and constraints is None
        )
        is_sample_gen_mode = (
            (num_beams == 1)
            and (num_beam_groups == 1)
            and do_sample is True
            and constraints is None
        )
        is_beam_gen_mode = (
            (num_beams > 1)
            and (num_beam_groups == 1)
            and do_sample is False
            and constraints is None
        )
        is_beam_sample_gen_mode = (
            (num_beams > 1)
            and (num_beam_groups == 1)
            and do_sample is True
            and constraints is None
        )
        is_group_beam_gen_mode = (
            (num_beams > 1) and (num_beam_groups > 1) and constraints is None
        )

        if num_beam_groups > num_beams:
            raise ValueError(
                "`num_beam_groups` has to be smaller or equal to `num_beams`"
            )
        if is_group_beam_gen_mode and do_sample is True:
            raise ValueError(
                "Diverse beam search cannot be used in sampling mode. Make sure that `do_sample` is set to `False`."
            )

        # 7. prepare distribution pre_processing samplers
        logits_processor = self._get_logits_processor(
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            encoder_no_repeat_ngram_size=encoder_no_repeat_ngram_size,
            encoder_input_ids=inputs_tensor,
            bad_words_ids=bad_words_ids,
            min_length=min_length,
            max_length=max_length,
            eos_token_id=eos_token_id,
            forced_bos_token_id=forced_bos_token_id,
            forced_eos_token_id=forced_eos_token_id,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            num_beams=num_beams,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
            remove_invalid_values=remove_invalid_values,
            logits_processor=logits_processor,
        )

        # 8. prepare stopping criteria
        stopping_criteria = self._get_stopping_criteria(
            max_length=max_length,
            max_time=max_time,
            stopping_criteria=stopping_criteria,
        )

        # 9. go into different generation modes
        if is_greedy_gen_mode:
            if num_return_sequences > 1:
                raise ValueError(
                    f"num_return_sequences has to be 1, but is {num_return_sequences} when doing greedy search."
                )

            # 10. run greedy search
            return self.greedy_search(
                input_ids,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif is_sample_gen_mode:
            # 10. prepare logits warper
            logits_warper = self._get_logits_warper(
                top_k=top_k,
                top_p=top_p,
                typical_p=typical_p,
                temperature=temperature,
                num_beams=num_beams,
            )

            # 11. expand input_ids with `num_return_sequences` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids,
                expand_size=num_return_sequences,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )

            # 12. run sample
            return self.sample(
                input_ids,
                logits_processor=logits_processor,
                logits_warper=logits_warper,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif is_beam_gen_mode:
            if num_return_sequences > num_beams:
                raise ValueError(
                    "`num_return_sequences` has to be smaller or equal to `num_beams`."
                )

            if stopping_criteria.max_length is None:
                raise ValueError(
                    "`max_length` needs to be a stopping_criteria for now."
                )

            # 10. prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=num_beams,
                device=self.device,
                length_penalty=length_penalty,
                do_early_stopping=early_stopping,
                num_beam_hyps_to_keep=num_return_sequences,
            )
            # 11. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids,
                expand_size=num_beams,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )
            # 12. run beam search
            return self.beam_search(
                input_ids,
                beam_scorer,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif is_beam_sample_gen_mode:
            # 10. prepare logits warper
            logits_warper = self._get_logits_warper(
                top_k=top_k,
                top_p=top_p,
                typical_p=typical_p,
                temperature=temperature,
                num_beams=num_beams,
            )

            if stopping_criteria.max_length is None:
                raise ValueError(
                    "`max_length` needs to be a stopping_criteria for now."
                )
            # 11. prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size * num_return_sequences,
                num_beams=num_beams,
                device=self.device,
                length_penalty=length_penalty,
                do_early_stopping=early_stopping,
            )

            # 12. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids,
                expand_size=num_beams * num_return_sequences,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )

            # 13. run beam sample
            return self.beam_sample(
                input_ids,
                beam_scorer,
                logits_processor=logits_processor,
                logits_warper=logits_warper,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif is_group_beam_gen_mode:
            if num_return_sequences > num_beams:
                raise ValueError(
                    "`num_return_sequences` has to be smaller or equal to `num_beams`."
                )

            if num_beams % num_beam_groups != 0:
                raise ValueError(
                    "`num_beams` should be divisible by `num_beam_groups` for group beam search."
                )

            if stopping_criteria.max_length is None:
                raise ValueError(
                    "`max_length` needs to be a stopping_criteria for now."
                )

            # 10. prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=num_beams,
                max_length=stopping_criteria.max_length,
                device=self.device,
                length_penalty=length_penalty,
                do_early_stopping=early_stopping,
                num_beam_hyps_to_keep=num_return_sequences,
                num_beam_groups=num_beam_groups,
            )
            # 11. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids,
                expand_size=num_beams,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )
            # 12. run beam search
            return self.group_beam_search(
                input_ids,
                beam_scorer,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif is_constraint_gen_mode:
            if num_return_sequences > num_beams:
                raise ValueError(
                    "`num_return_sequences` has to be smaller or equal to `num_beams`."
                )

            if stopping_criteria.max_length is None:
                raise ValueError(
                    "`max_length` needs to be a stopping_criteria for now."
                )

            if num_beams <= 1:
                raise ValueError(
                    "`num_beams` needs to be greater than 1 for constrained genertation."
                )

            if do_sample:
                raise ValueError(
                    "`do_sample` needs to be false for constrained generation."
                )

            if num_beam_groups is not None and num_beam_groups > 1:
                raise ValueError(
                    "`num_beam_groups` not supported yet for constrained generation."
                )

            # 10. prepare beam search scorer
            constrained_beam_scorer = ConstrainedBeamSearchScorer(
                constraints=constraints,
                batch_size=batch_size,
                num_beams=num_beams,
                device=self.device,
                length_penalty=length_penalty,
                do_early_stopping=early_stopping,
                num_beam_hyps_to_keep=num_return_sequences,
            )
            # 11. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids,
                expand_size=num_beams,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )
            # 12. run beam search
            return self.constrained_beam_search(
                input_ids,
                constrained_beam_scorer=constrained_beam_scorer,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

    def greedy_search(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: Optional[bool] = False,
        **model_kwargs,
    ) -> Union[GreedySearchOutput, torch.LongTensor]:
        r"""
        Generates sequences for models with a language modeling head using greedy decoding.

        Parameters:

            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.

            max_length (`int`, *optional*, defaults to 20):
                **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
                tokens. The maximum length of the sequence to be generated.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`int`, *optional*):
                The id of the *end-of-sequence* token.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to `False`):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            model_kwargs:
                Additional model specific keyword arguments will be forwarded to the `forward` function of the model.
                If model is an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation_utils.GreedySearchDecoderOnlyOutput`], [`~generation_utils.GreedySearchEncoderDecoderOutput`]
            or `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation_utils.GreedySearchDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation_utils.GreedySearchEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.

        """
        # init values
        logits_processor = (
            logits_processor if logits_processor is not None else LogitsProcessorList()
        )
        stopping_criteria = (
            stopping_criteria
            if stopping_criteria is not None
            else StoppingCriteriaList()
        )
        if max_length is not None:
            print(
                "`max_length` is deprecated in this function, use `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(
                stopping_criteria, max_length
            )
        pad_token_id = (
            pad_token_id if pad_token_id is not None else self.config.pad_token_id
        )
        eos_token_id = (
            eos_token_id if eos_token_id is not None else self.config.eos_token_id
        )
        output_scores = (
            output_scores if output_scores is not None else self.config.output_scores
        )
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = (
            () if (return_dict_in_generate and output_attentions) else None
        )
        cross_attentions = (
            () if (return_dict_in_generate and output_attentions) else None
        )
        decoder_hidden_states = (
            () if (return_dict_in_generate and output_hidden_states) else None
        )

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = (
                model_kwargs["encoder_outputs"].get("attentions")
                if output_attentions
                else None
            )
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states")
                if output_hidden_states
                else None
            )

        # keep track of which sequences are already finished
        unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
        cur_len = input_ids.shape[-1]

        this_peer_finished = False  # used by synced_gpus only
        while True:

            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(
                    0.0 if this_peer_finished else 1.0
                ).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,)
                        if self.config.is_encoder_decoder
                        else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # pre-process distribution
            next_tokens_scores = logits_processor(input_ids, next_token_logits)

            # argmax
            next_tokens = torch.argmax(next_tokens_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError(
                        "If `eos_token_id` is defined, make sure that `pad_token_id` is defined."
                    )
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (
                    1 - unfinished_sequences
                )

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            cur_len = cur_len + 1

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    (next_tokens != eos_token_id).long()
                )

            # stop when each sentence is finished, or if we exceed the maximum length
            if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GreedySearchEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return GreedySearchDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return input_ids

    def score(
        self, input_ids, target_ids, **model_kwargs  # V L1 (repeated, BS 1)  # V L2
    ):
        scores = torch.zeros(input_ids.size(0), target_ids.size(1))
        for l in range(target_ids.size(1)):
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
            )

            next_token_logits = outputs.logits[:, -1, :]
            next_token_probs = next_token_logits.softmax(-1)
            next_tokens = target_ids[:, l]  # follow the target
            scores[:, l] = next_token_probs[
                torch.arange(len(outputs.logits)), next_tokens
            ]

            # update generated ids, model inputs for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )

        return scores
