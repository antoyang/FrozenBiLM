# coding=utf-8
# Copyright 2020 Microsoft and the Hugging Face Inc. team.
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
""" PyTorch DeBERTa-v2 model. """

import math
from collections.abc import Sequence
from typing import Tuple, Optional

import numpy as np
import torch
from torch import _softmax_backward_data, nn
from torch.nn import CrossEntropyLoss, LayerNorm

from model.adapter import Adapter
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    # BaseModelOutput,
    ModelOutput,
    MaskedLMOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers import DebertaV2Config


_CONFIG_FOR_DOC = "DebertaV2Config"
_TOKENIZER_FOR_DOC = "DebertaV2Tokenizer"
_CHECKPOINT_FOR_DOC = "microsoft/deberta-v2-xlarge"

DEBERTA_V2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/deberta-v2-xlarge",
    "microsoft/deberta-v2-xxlarge",
    "microsoft/deberta-v2-xlarge-mnli",
    "microsoft/deberta-v2-xxlarge-mnli",
]


class BaseModelOutput(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states and attentions.
    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    position_embeddings: torch.FloatTensor = None
    attention_mask: torch.BoolTensor = None


# Copied from transformers.models.deberta.modeling_deberta.ContextPooler
class ContextPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.pooler_hidden_size, config.pooler_hidden_size)
        self.dropout = StableDropout(config.pooler_dropout)
        self.config = config

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.

        context_token = hidden_states[:, 0]
        context_token = self.dropout(context_token)
        pooled_output = self.dense(context_token)
        pooled_output = ACT2FN[self.config.pooler_hidden_act](pooled_output)
        return pooled_output

    @property
    def output_dim(self):
        return self.config.hidden_size


# Copied from transformers.models.deberta.modeling_deberta.XSoftmax with deberta->deberta_v2
class XSoftmax(torch.autograd.Function):
    """
    Masked Softmax which is optimized for saving memory

    Args:
        input (:obj:`torch.tensor`): The input tensor that will apply softmax.
        mask (:obj:`torch.IntTensor`): The mask matrix where 0 indicate that element will be ignored in the softmax calculation.
        dim (int): The dimension that will apply softmax

    Example::

          import torch
          from transformers.models.deberta_v2.modeling_deberta_v2 import XSoftmax

          # Make a tensor
          x = torch.randn([4,20,100])

          # Create a mask
          mask = (x>0).int()

          y = XSoftmax.apply(x, mask, dim=-1)
    """

    @staticmethod
    def forward(self, input, mask, dim):
        self.dim = dim
        rmask = ~(mask.bool())

        output = input.masked_fill(rmask, float("-inf"))
        output = torch.softmax(output, self.dim)
        output.masked_fill_(rmask, 0)
        self.save_for_backward(output)
        return output

    @staticmethod
    def backward(self, grad_output):
        (output,) = self.saved_tensors
        inputGrad = _softmax_backward_data(grad_output, output, self.dim, output)
        return inputGrad, None, None


# Copied from transformers.models.deberta.modeling_deberta.DropoutContext
class DropoutContext(object):
    def __init__(self):
        self.dropout = 0
        self.mask = None
        self.scale = 1
        self.reuse_mask = True


# Copied from transformers.models.deberta.modeling_deberta.get_mask
def get_mask(input, local_context):
    if not isinstance(local_context, DropoutContext):
        dropout = local_context
        mask = None
    else:
        dropout = local_context.dropout
        dropout *= local_context.scale
        mask = local_context.mask if local_context.reuse_mask else None

    if dropout > 0 and mask is None:
        mask = (1 - torch.empty_like(input).bernoulli_(1 - dropout)).bool()

    if isinstance(local_context, DropoutContext):
        if local_context.mask is None:
            local_context.mask = mask

    return mask, dropout


# Copied from transformers.models.deberta.modeling_deberta.XDropout
class XDropout(torch.autograd.Function):
    """Optimized dropout function to save computation and memory by using mask operation instead of multiplication."""

    @staticmethod
    def forward(ctx, input, local_ctx):
        mask, dropout = get_mask(input, local_ctx)
        ctx.scale = 1.0 / (1 - dropout)
        if dropout > 0:
            ctx.save_for_backward(mask)
            return input.masked_fill(mask, 0) * ctx.scale
        else:
            return input

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.scale > 1:
            (mask,) = ctx.saved_tensors
            return grad_output.masked_fill(mask, 0) * ctx.scale, None
        else:
            return grad_output, None


# Copied from transformers.models.deberta.modeling_deberta.StableDropout
class StableDropout(nn.Module):
    """
    Optimized dropout module for stabilizing the training

    Args:
        drop_prob (float): the dropout probabilities
    """

    def __init__(self, drop_prob):
        super().__init__()
        self.drop_prob = drop_prob
        self.count = 0
        self.context_stack = None

    def forward(self, x):
        """
        Call the module

        Args:
            x (:obj:`torch.tensor`): The input tensor to apply dropout
        """
        if self.training and self.drop_prob > 0:
            return XDropout.apply(x, self.get_context())
        return x

    def clear_context(self):
        self.count = 0
        self.context_stack = None

    def init_context(self, reuse_mask=True, scale=1):
        if self.context_stack is None:
            self.context_stack = []
        self.count = 0
        for c in self.context_stack:
            c.reuse_mask = reuse_mask
            c.scale = scale

    def get_context(self):
        if self.context_stack is not None:
            if self.count >= len(self.context_stack):
                self.context_stack.append(DropoutContext())
            ctx = self.context_stack[self.count]
            ctx.dropout = self.drop_prob
            self.count += 1
            return ctx
        else:
            return self.drop_prob


# Copied from transformers.models.deberta.modeling_deberta.DebertaSelfOutput with DebertaLayerNorm->LayerNorm
class DebertaV2SelfOutput(nn.Module):
    def __init__(self, config, ds_factor, dropout):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, config.layer_norm_eps)
        self.dropout = StableDropout(config.hidden_dropout_prob)
        self.ds_factor = ds_factor
        if self.ds_factor:
            self.adapter = Adapter(ds_factor, config.hidden_size, dropout=dropout)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        if self.ds_factor:
            hidden_states = self.adapter(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# Copied from transformers.models.deberta.modeling_deberta.DebertaAttention with Deberta->DebertaV2
class DebertaV2Attention(nn.Module):
    def __init__(self, config, ds_factor, dropout):
        super().__init__()
        self.self = DisentangledSelfAttention(config)
        self.output = DebertaV2SelfOutput(config, ds_factor, dropout)
        self.config = config

    def forward(
        self,
        hidden_states,
        attention_mask,
        return_att=False,
        query_states=None,
        relative_pos=None,
        rel_embeddings=None,
    ):
        self_output = self.self(
            hidden_states,
            attention_mask,
            return_att,
            query_states=query_states,
            relative_pos=relative_pos,
            rel_embeddings=rel_embeddings,
        )
        if return_att:
            self_output, att_matrix = self_output
        if query_states is None:
            query_states = hidden_states
        attention_output = self.output(self_output, query_states)

        if return_att:
            return (attention_output, att_matrix)
        else:
            return attention_output


# Copied from transformers.models.bert.modeling_bert.BertIntermediate with Bert->DebertaV2
class DebertaV2Intermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from transformers.models.deberta.modeling_deberta.DebertaOutput with DebertaLayerNorm->LayerNorm
class DebertaV2Output(nn.Module):
    def __init__(self, config, ds_factor, dropout):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, config.layer_norm_eps)
        self.dropout = StableDropout(config.hidden_dropout_prob)
        self.config = config
        self.ds_factor = ds_factor
        if self.ds_factor:
            self.adapter = Adapter(ds_factor, config.hidden_size, dropout=dropout)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        if self.ds_factor:
            hidden_states = self.adapter(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# Copied from transformers.models.deberta.modeling_deberta.DebertaLayer with Deberta->DebertaV2
class DebertaV2Layer(nn.Module):
    def __init__(
        self,
        config,
        ds_factor_attn,
        ds_factor_ff,
        dropout,
    ):
        super().__init__()
        self.attention = DebertaV2Attention(config, ds_factor_attn, dropout)
        self.intermediate = DebertaV2Intermediate(config)
        self.output = DebertaV2Output(config, ds_factor_ff, dropout)

    def forward(
        self,
        hidden_states,
        attention_mask,
        return_att=False,
        query_states=None,
        relative_pos=None,
        rel_embeddings=None,
    ):
        attention_output = self.attention(
            hidden_states,
            attention_mask,
            return_att=return_att,
            query_states=query_states,
            relative_pos=relative_pos,
            rel_embeddings=rel_embeddings,
        )
        if return_att:
            attention_output, att_matrix = attention_output
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        if return_att:
            return (layer_output, att_matrix)
        else:
            return layer_output


class ConvLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        kernel_size = getattr(config, "conv_kernel_size", 3)
        groups = getattr(config, "conv_groups", 1)
        self.conv_act = getattr(config, "conv_act", "tanh")
        self.conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=groups,
        )
        self.LayerNorm = LayerNorm(config.hidden_size, config.layer_norm_eps)
        self.dropout = StableDropout(config.hidden_dropout_prob)
        self.config = config

    def forward(self, hidden_states, residual_states, input_mask):
        out = (
            self.conv(hidden_states.permute(0, 2, 1).contiguous())
            .permute(0, 2, 1)
            .contiguous()
        )
        rmask = (1 - input_mask).bool()
        out.masked_fill_(rmask.unsqueeze(-1).expand(out.size()), 0)
        out = ACT2FN[self.conv_act](self.dropout(out))

        layer_norm_input = residual_states + out
        output = self.LayerNorm(layer_norm_input).to(layer_norm_input)

        if input_mask is None:
            output_states = output
        else:
            if input_mask.dim() != layer_norm_input.dim():
                if input_mask.dim() == 4:
                    input_mask = input_mask.squeeze(1).squeeze(1)
                input_mask = input_mask.unsqueeze(2)

            input_mask = input_mask.to(output.dtype)
            output_states = output * input_mask

        return output_states


class DebertaV2Encoder(nn.Module):
    """Modified BertEncoder with relative position bias support"""

    def __init__(
        self,
        config,
        ds_factor_attn,
        ds_factor_ff,
        dropout,
    ):
        super().__init__()

        self.layer = nn.ModuleList(
            [
                DebertaV2Layer(
                    config,
                    ds_factor_attn,
                    ds_factor_ff,
                    dropout,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.relative_attention = getattr(config, "relative_attention", False)

        if self.relative_attention:
            self.max_relative_positions = getattr(config, "max_relative_positions", -1)
            if self.max_relative_positions < 1:
                self.max_relative_positions = config.max_position_embeddings

            self.position_buckets = getattr(config, "position_buckets", -1)
            pos_ebd_size = self.max_relative_positions * 2

            if self.position_buckets > 0:
                pos_ebd_size = self.position_buckets * 2

            self.rel_embeddings = nn.Embedding(pos_ebd_size, config.hidden_size)

        self.norm_rel_ebd = [
            x.strip()
            for x in getattr(config, "norm_rel_ebd", "none").lower().split("|")
        ]

        if "layer_norm" in self.norm_rel_ebd:
            self.LayerNorm = LayerNorm(
                config.hidden_size, config.layer_norm_eps, elementwise_affine=True
            )

        self.conv = (
            ConvLayer(config) if getattr(config, "conv_kernel_size", 0) > 0 else None
        )

    def get_rel_embedding(self):
        rel_embeddings = self.rel_embeddings.weight if self.relative_attention else None
        if rel_embeddings is not None and ("layer_norm" in self.norm_rel_ebd):
            rel_embeddings = self.LayerNorm(rel_embeddings)
        return rel_embeddings

    def get_attention_mask(self, attention_mask):
        if attention_mask.dim() <= 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = extended_attention_mask * extended_attention_mask.squeeze(
                -2
            ).unsqueeze(-1)
            attention_mask = attention_mask.byte()
        elif attention_mask.dim() == 3:
            attention_mask = attention_mask.unsqueeze(1)

        return attention_mask

    def get_rel_pos(self, hidden_states, query_states=None, relative_pos=None):
        if self.relative_attention and relative_pos is None:
            q = (
                query_states.size(-2)
                if query_states is not None
                else hidden_states.size(-2)
            )
            relative_pos = build_relative_position(
                q,
                hidden_states.size(-2),
                bucket_size=self.position_buckets,
                max_position=self.max_relative_positions,
            )
        return relative_pos

    def forward(
        self,
        hidden_states,
        attention_mask,
        output_hidden_states=True,
        output_attentions=False,
        query_states=None,
        relative_pos=None,
        return_dict=True,
    ):
        if attention_mask.dim() <= 2:
            input_mask = attention_mask
        else:
            input_mask = (attention_mask.sum(-2) > 0).byte()
        attention_mask = self.get_attention_mask(attention_mask)
        relative_pos = self.get_rel_pos(hidden_states, query_states, relative_pos)

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        if isinstance(hidden_states, Sequence):
            next_kv = hidden_states[0]
        else:
            next_kv = hidden_states
        rel_embeddings = self.get_rel_embedding()
        output_states = next_kv
        for i, layer_module in enumerate(self.layer):

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (output_states,)

            output_states = layer_module(
                next_kv,
                attention_mask,
                output_attentions,
                query_states=query_states,
                relative_pos=relative_pos,
                rel_embeddings=rel_embeddings,
            )
            if output_attentions:
                output_states, att_m = output_states

            if i == 0 and self.conv is not None:
                output_states = self.conv(hidden_states, output_states, input_mask)

            if query_states is not None:
                query_states = output_states
                if isinstance(hidden_states, Sequence):
                    next_kv = hidden_states[i + 1] if i + 1 < len(self.layer) else None
            else:
                next_kv = output_states

            if output_attentions:
                all_attentions = all_attentions + (att_m,)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (output_states,)

        if not return_dict:
            return tuple(
                v
                for v in [output_states, all_hidden_states, all_attentions]
                if v is not None
            )
        return BaseModelOutput(
            last_hidden_state=output_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


def make_log_bucket_position(relative_pos, bucket_size, max_position):
    sign = np.sign(relative_pos)
    mid = bucket_size // 2
    abs_pos = np.where(
        (relative_pos < mid) & (relative_pos > -mid), mid - 1, np.abs(relative_pos)
    )
    log_pos = (
        np.ceil(np.log(abs_pos / mid) / np.log((max_position - 1) / mid) * (mid - 1))
        + mid
    )
    bucket_pos = np.where(abs_pos <= mid, relative_pos, log_pos * sign).astype(np.int)
    return bucket_pos


def build_relative_position(query_size, key_size, bucket_size=-1, max_position=-1):
    """
    Build relative position according to the query and key

    We assume the absolute position of query :math:`P_q` is range from (0, query_size) and the absolute position of key
    :math:`P_k` is range from (0, key_size), The relative positions from query to key is :math:`R_{q \\rightarrow k} =
    P_q - P_k`

    Args:
        query_size (int): the length of query
        key_size (int): the length of key
        bucket_size (int): the size of position bucket
        max_position (int): the maximum allowed absolute position

    Return:
        :obj:`torch.LongTensor`: A tensor with shape [1, query_size, key_size]

    """
    q_ids = np.arange(0, query_size)
    k_ids = np.arange(0, key_size)
    rel_pos_ids = q_ids[:, None] - np.tile(k_ids, (q_ids.shape[0], 1))
    if bucket_size > 0 and max_position > 0:
        rel_pos_ids = make_log_bucket_position(rel_pos_ids, bucket_size, max_position)
    rel_pos_ids = torch.tensor(rel_pos_ids, dtype=torch.long)
    rel_pos_ids = rel_pos_ids[:query_size, :]
    rel_pos_ids = rel_pos_ids.unsqueeze(0)
    return rel_pos_ids


@torch.jit.script
# Copied from transformers.models.deberta.modeling_deberta.c2p_dynamic_expand
def c2p_dynamic_expand(c2p_pos, query_layer, relative_pos):
    return c2p_pos.expand(
        [
            query_layer.size(0),
            query_layer.size(1),
            query_layer.size(2),
            relative_pos.size(-1),
        ]
    )


@torch.jit.script
# Copied from transformers.models.deberta.modeling_deberta.p2c_dynamic_expand
def p2c_dynamic_expand(c2p_pos, query_layer, key_layer):
    return c2p_pos.expand(
        [
            query_layer.size(0),
            query_layer.size(1),
            key_layer.size(-2),
            key_layer.size(-2),
        ]
    )


@torch.jit.script
# Copied from transformers.models.deberta.modeling_deberta.pos_dynamic_expand
def pos_dynamic_expand(pos_index, p2c_att, key_layer):
    return pos_index.expand(
        p2c_att.size()[:2] + (pos_index.size(-2), key_layer.size(-2))
    )


class DisentangledSelfAttention(nn.Module):
    """
    Disentangled self-attention module

    Parameters:
        config (:obj:`DebertaV2Config`):
            A model config class instance with the configuration to build a new model. The schema is similar to
            `BertConfig`, for more details, please refer :class:`~transformers.DebertaV2Config`

    """

    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.num_attention_heads = config.num_attention_heads
        _attention_head_size = config.hidden_size // config.num_attention_heads
        self.attention_head_size = getattr(
            config, "attention_head_size", _attention_head_size
        )
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=True)
        self.key_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=True)
        self.value_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=True)

        self.share_att_key = getattr(config, "share_att_key", False)
        self.pos_att_type = (
            config.pos_att_type if config.pos_att_type is not None else []
        )
        self.relative_attention = getattr(config, "relative_attention", False)

        if self.relative_attention:
            self.position_buckets = getattr(config, "position_buckets", -1)
            self.max_relative_positions = getattr(config, "max_relative_positions", -1)
            if self.max_relative_positions < 1:
                self.max_relative_positions = config.max_position_embeddings
            self.pos_ebd_size = self.max_relative_positions
            if self.position_buckets > 0:
                self.pos_ebd_size = self.position_buckets

            self.pos_dropout = StableDropout(config.hidden_dropout_prob)

            if not self.share_att_key:
                if "c2p" in self.pos_att_type or "p2p" in self.pos_att_type:
                    self.pos_key_proj = nn.Linear(
                        config.hidden_size, self.all_head_size, bias=True
                    )
                if "p2c" in self.pos_att_type or "p2p" in self.pos_att_type:
                    self.pos_query_proj = nn.Linear(
                        config.hidden_size, self.all_head_size
                    )

        self.dropout = StableDropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x, attention_heads):
        new_x_shape = x.size()[:-1] + (attention_heads, -1)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))

    def forward(
        self,
        hidden_states,
        attention_mask,
        return_att=False,
        query_states=None,
        relative_pos=None,
        rel_embeddings=None,
    ):
        """
        Call the module

        Args:
            hidden_states (:obj:`torch.FloatTensor`):
                Input states to the module usually the output from previous layer, it will be the Q,K and V in
                `Attention(Q,K,V)`

            attention_mask (:obj:`torch.ByteTensor`):
                An attention mask matrix of shape [`B`, `N`, `N`] where `B` is the batch size, `N` is the maximum
                sequence length in which element [i,j] = `1` means the `i` th token in the input can attend to the `j`
                th token.

            return_att (:obj:`bool`, optional):
                Whether return the attention matrix.

            query_states (:obj:`torch.FloatTensor`, optional):
                The `Q` state in `Attention(Q,K,V)`.

            relative_pos (:obj:`torch.LongTensor`):
                The relative position encoding between the tokens in the sequence. It's of shape [`B`, `N`, `N`] with
                values ranging in [`-max_relative_positions`, `max_relative_positions`].

            rel_embeddings (:obj:`torch.FloatTensor`):
                The embedding of relative distances. It's a tensor of shape [:math:`2 \\times
                \\text{max_relative_positions}`, `hidden_size`].


        """
        if query_states is None:
            query_states = hidden_states
        query_layer = self.transpose_for_scores(
            self.query_proj(query_states), self.num_attention_heads
        )
        key_layer = self.transpose_for_scores(
            self.key_proj(hidden_states), self.num_attention_heads
        )
        value_layer = self.transpose_for_scores(
            self.value_proj(hidden_states), self.num_attention_heads
        )

        rel_att = None
        # Take the dot product between "query" and "key" to get the raw attention scores.
        scale_factor = 1
        if "c2p" in self.pos_att_type:
            scale_factor += 1
        if "p2c" in self.pos_att_type:
            scale_factor += 1
        if "p2p" in self.pos_att_type:
            scale_factor += 1
        scale = math.sqrt(query_layer.size(-1) * scale_factor)
        attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2)) / scale
        if self.relative_attention:
            rel_embeddings = self.pos_dropout(rel_embeddings)
            rel_att = self.disentangled_attention_bias(
                query_layer, key_layer, relative_pos, rel_embeddings, scale_factor
            )

        if rel_att is not None:
            attention_scores = attention_scores + rel_att
        attention_scores = attention_scores
        attention_scores = attention_scores.view(
            -1,
            self.num_attention_heads,
            attention_scores.size(-2),
            attention_scores.size(-1),
        )

        # bsz x height x length x dimension
        attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.bmm(
            attention_probs.view(
                -1, attention_probs.size(-2), attention_probs.size(-1)
            ),
            value_layer,
        )
        context_layer = (
            context_layer.view(
                -1,
                self.num_attention_heads,
                context_layer.size(-2),
                context_layer.size(-1),
            )
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        new_context_layer_shape = context_layer.size()[:-2] + (-1,)
        context_layer = context_layer.view(*new_context_layer_shape)
        if return_att:
            return (context_layer, attention_probs)
        else:
            return context_layer

    def disentangled_attention_bias(
        self, query_layer, key_layer, relative_pos, rel_embeddings, scale_factor
    ):
        if relative_pos is None:
            q = query_layer.size(-2)
            relative_pos = build_relative_position(
                q,
                key_layer.size(-2),
                bucket_size=self.position_buckets,
                max_position=self.max_relative_positions,
            )
        if relative_pos.dim() == 2:
            relative_pos = relative_pos.unsqueeze(0).unsqueeze(0)
        elif relative_pos.dim() == 3:
            relative_pos = relative_pos.unsqueeze(1)
        # bsz x height x query x key
        elif relative_pos.dim() != 4:
            raise ValueError(
                f"Relative position ids must be of dim 2 or 3 or 4. {relative_pos.dim()}"
            )

        att_span = self.pos_ebd_size
        relative_pos = relative_pos.long().to(query_layer.device)

        rel_embeddings = rel_embeddings[
            self.pos_ebd_size - att_span : self.pos_ebd_size + att_span, :
        ].unsqueeze(0)
        if self.share_att_key:
            pos_query_layer = self.transpose_for_scores(
                self.query_proj(rel_embeddings), self.num_attention_heads
            ).repeat(query_layer.size(0) // self.num_attention_heads, 1, 1)
            pos_key_layer = self.transpose_for_scores(
                self.key_proj(rel_embeddings), self.num_attention_heads
            ).repeat(query_layer.size(0) // self.num_attention_heads, 1, 1)
        else:
            if "c2p" in self.pos_att_type or "p2p" in self.pos_att_type:
                pos_key_layer = self.transpose_for_scores(
                    self.pos_key_proj(rel_embeddings), self.num_attention_heads
                ).repeat(
                    query_layer.size(0) // self.num_attention_heads, 1, 1
                )  # .split(self.all_head_size, dim=-1)
            if "p2c" in self.pos_att_type or "p2p" in self.pos_att_type:
                pos_query_layer = self.transpose_for_scores(
                    self.pos_query_proj(rel_embeddings), self.num_attention_heads
                ).repeat(
                    query_layer.size(0) // self.num_attention_heads, 1, 1
                )  # .split(self.all_head_size, dim=-1)

        score = 0
        # content->position
        if "c2p" in self.pos_att_type:
            scale = math.sqrt(pos_key_layer.size(-1) * scale_factor)
            c2p_att = torch.bmm(query_layer, pos_key_layer.transpose(-1, -2))
            c2p_pos = torch.clamp(relative_pos + att_span, 0, att_span * 2 - 1)
            c2p_att = torch.gather(
                c2p_att,
                dim=-1,
                index=c2p_pos.squeeze(0).expand(
                    [query_layer.size(0), query_layer.size(1), relative_pos.size(-1)]
                ),
            )
            score += c2p_att / scale

        # position->content
        if "p2c" in self.pos_att_type or "p2p" in self.pos_att_type:
            scale = math.sqrt(pos_query_layer.size(-1) * scale_factor)
            if key_layer.size(-2) != query_layer.size(-2):
                r_pos = build_relative_position(
                    key_layer.size(-2),
                    key_layer.size(-2),
                    bucket_size=self.position_buckets,
                    max_position=self.max_relative_positions,
                ).to(query_layer.device)
                r_pos = r_pos.unsqueeze(0)
            else:
                r_pos = relative_pos

            p2c_pos = torch.clamp(-r_pos + att_span, 0, att_span * 2 - 1)
            if query_layer.size(-2) != key_layer.size(-2):
                pos_index = relative_pos[:, :, :, 0].unsqueeze(-1)

        if "p2c" in self.pos_att_type:
            p2c_att = torch.bmm(key_layer, pos_query_layer.transpose(-1, -2))
            p2c_att = torch.gather(
                p2c_att,
                dim=-1,
                index=p2c_pos.squeeze(0).expand(
                    [query_layer.size(0), key_layer.size(-2), key_layer.size(-2)]
                ),
            ).transpose(-1, -2)
            if query_layer.size(-2) != key_layer.size(-2):
                p2c_att = torch.gather(
                    p2c_att,
                    dim=-2,
                    index=pos_index.expand(
                        p2c_att.size()[:2] + (pos_index.size(-2), key_layer.size(-2))
                    ),
                )
            score += p2c_att / scale

        # position->position
        if "p2p" in self.pos_att_type:
            pos_query = pos_query_layer[:, :, att_span:, :]
            p2p_att = torch.matmul(pos_query, pos_key_layer.transpose(-1, -2))
            p2p_att = p2p_att.expand(query_layer.size()[:2] + p2p_att.size()[2:])
            if query_layer.size(-2) != key_layer.size(-2):
                p2p_att = torch.gather(
                    p2p_att,
                    dim=-2,
                    index=pos_index.expand(
                        query_layer.size()[:2] + (pos_index.size(-2), p2p_att.size(-1))
                    ),
                )
            p2p_att = torch.gather(
                p2p_att,
                dim=-1,
                index=c2p_pos.expand(
                    [
                        query_layer.size(0),
                        query_layer.size(1),
                        query_layer.size(2),
                        relative_pos.size(-1),
                    ]
                ),
            )
            score += p2p_att

        return score


# Copied from transformers.models.deberta.modeling_deberta.DebertaEmbeddings with DebertaLayerNorm->LayerNorm
class DebertaV2Embeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(
        self,
        config,
        features_dim,
    ):
        super().__init__()
        pad_token_id = getattr(config, "pad_token_id", 0)
        self.embedding_size = getattr(config, "embedding_size", config.hidden_size)
        self.word_embeddings = nn.Embedding(
            config.vocab_size, self.embedding_size, padding_idx=pad_token_id
        )

        self.position_biased_input = getattr(config, "position_biased_input", True)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, self.embedding_size
        )  # it is used for the decoder anyway

        if config.type_vocab_size > 0:
            self.token_type_embeddings = nn.Embedding(
                config.type_vocab_size, self.embedding_size
            )

        if self.embedding_size != config.hidden_size:
            self.embed_proj = nn.Linear(
                self.embedding_size, config.hidden_size, bias=False
            )
        self.LayerNorm = LayerNorm(config.hidden_size, config.layer_norm_eps)
        self.dropout = StableDropout(config.hidden_dropout_prob)
        self.config = config

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1))
        )

        self.features_dim = features_dim
        if self.features_dim:
            self.linear_video = nn.Linear(features_dim, config.hidden_size)

    def get_video_embedding(self, video):
        video = self.linear_video(video)
        return video

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        mask=None,
        inputs_embeds=None,
        video=None,
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
            if self.features_dim and video is not None:
                video = self.get_video_embedding(video)
                inputs_embeds = torch.cat([video, inputs_embeds], 1)
                input_shape = inputs_embeds[:, :, 0].shape

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_shape, dtype=torch.long, device=self.position_ids.device
            )

        if self.position_embeddings is not None:
            position_embeddings = self.position_embeddings(position_ids.long())
        else:
            position_embeddings = torch.zeros_like(inputs_embeds)

        embeddings = inputs_embeds
        if self.position_biased_input:
            embeddings += position_embeddings
        if self.config.type_vocab_size > 0:
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embeddings += token_type_embeddings

        if self.embedding_size != self.config.hidden_size:
            embeddings = self.embed_proj(embeddings)

        embeddings = self.LayerNorm(embeddings)

        if mask is not None:
            if mask.dim() != embeddings.dim():
                if mask.dim() == 4:
                    mask = mask.squeeze(1).squeeze(1)
                mask = mask.unsqueeze(2)
            mask = mask.to(embeddings.dtype)

            embeddings = embeddings * mask

        embeddings = self.dropout(embeddings)
        return {
            "embeddings": embeddings,
            "position_embeddings": position_embeddings,
        }


# Copied from transformers.models.deberta.modeling_deberta.DebertaPreTrainedModel with Deberta->DebertaV2


class DebertaV2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = DebertaV2Config
    base_model_prefix = "deberta"
    _keys_to_ignore_on_load_missing = ["position_ids"]
    _keys_to_ignore_on_load_unexpected = ["position_embeddings"]

    def __init__(self, config):
        super().__init__(config)
        self._register_load_state_dict_pre_hook(self._pre_load_hook)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _pre_load_hook(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """
        Removes the classifier if it doesn't have the correct number of labels.
        """
        self_state = self.state_dict()
        if (
            ("classifier.weight" in self_state)
            and ("classifier.weight" in state_dict)
            and self_state["classifier.weight"].size()
            != state_dict["classifier.weight"].size()
        ):
            print(
                f"The checkpoint classifier head has a shape {state_dict['classifier.weight'].size()} and this model "
                f"classifier head has a shape {self_state['classifier.weight'].size()}. Ignoring the checkpoint "
                f"weights. You should train your model on new data."
            )
            del state_dict["classifier.weight"]
            if "classifier.bias" in state_dict:
                del state_dict["classifier.bias"]


# Copied from transformers.models.deberta.modeling_deberta.DebertaModel with Deberta->DebertaV2
class DebertaV2Model(DebertaV2PreTrainedModel):
    def __init__(
        self,
        config,
        max_feats=10,
        features_dim=768,
        freeze_lm=False,
        ds_factor_attn=8,
        ds_factor_ff=8,
        ft_ln=False,
        dropout=0.1,
    ):
        super().__init__(config)

        self.embeddings = DebertaV2Embeddings(
            config,
            features_dim,
        )
        self.encoder = DebertaV2Encoder(
            config,
            ds_factor_attn,
            ds_factor_ff,
            dropout,
        )
        self.z_steps = 0
        self.config = config

        self.features_dim = features_dim
        self.max_feats = max_feats
        if freeze_lm:
            for n, p in self.named_parameters():
                if (not "linear_video" in n) and (not "adapter" in n):
                    if ft_ln and "LayerNorm" in n:
                        continue
                    else:
                        p.requires_grad_(False)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings.word_embeddings = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        raise NotImplementedError(
            "The prune function is not implemented in DeBERTa model."
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        video=None,
        video_mask=None,
    ):
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
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)

        if self.features_dim and video is not None:
            if video_mask is None:
                video_shape = video[:, :, 0].size()
                video_mask = torch.ones(video_shape, device=device)
            attention_mask = torch.cat([video_mask, attention_mask], 1)
            input_shape = attention_mask.size()

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            mask=attention_mask,
            inputs_embeds=inputs_embeds,
            video=video,
        )
        embedding_output, position_embeddings = (
            embedding_output["embeddings"],
            embedding_output["position_embeddings"],
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask,
            output_hidden_states=True,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )
        encoded_layers = encoder_outputs[1]

        if self.z_steps > 1:
            hidden_states = encoded_layers[-2]
            layers = [self.encoder.layer[-1] for _ in range(self.z_steps)]
            query_states = encoded_layers[-1]
            rel_embeddings = self.encoder.get_rel_embedding()
            attention_mask = self.encoder.get_attention_mask(attention_mask)
            rel_pos = self.encoder.get_rel_pos(embedding_output)
            for layer in layers[1:]:
                query_states = layer(
                    hidden_states,
                    attention_mask,
                    return_att=False,
                    query_states=query_states,
                    relative_pos=rel_pos,
                    rel_embeddings=rel_embeddings,
                )
                encoded_layers.append(query_states)

        sequence_output = encoded_layers[-1]

        if not return_dict:
            return (sequence_output,) + encoder_outputs[
                (1 if output_hidden_states else 2) :
            ]

        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states
            if output_hidden_states
            else None,
            attentions=encoder_outputs.attentions,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )


# Copied from transformers.models.deberta.modeling_deberta.DebertaForMaskedLM with Deberta->DebertaV2
class DebertaV2ForMaskedLM(DebertaV2PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(
        self,
        config,
        max_feats=10,
        features_dim=768,
        freeze_lm=True,
        freeze_mlm=True,
        ds_factor_attn=8,
        ds_factor_ff=8,
        ft_ln=True,
        dropout=0.1,
        n_ans=0,
        freeze_last=True,
    ):
        """
        :param config: BiLM configuration
        :param max_feats: maximum number of frames used by the model
        :param features_dim: embedding dimension of the visual features, set = 0 for text-only mode
        :param freeze_lm: whether to freeze or not the language model (Transformer encoder + token embedder)
        :param freeze_mlm: whether to freeze or not the MLM head
        :param ds_factor_attn: downsampling factor for the adapter after self-attention, no adapter if set to 0
        :param ds_factor_ff: downsampling factor for the adapter after feed-forward, no adapter if set to 0
        :param ft_ln: whether to finetune or not the normalization layers
        :param dropout: dropout probability in the adapter
        :param n_ans: number of answers in the downstream vocabulary, set = 0 during cross-modal training
        :param freeze_last: whether to freeze or not the answer embedding module
        """
        super().__init__(config)

        self.deberta = DebertaV2Model(
            config,
            max_feats,
            features_dim,
            freeze_lm,
            ds_factor_attn,
            ds_factor_ff,
            ft_ln,
            dropout,
        )
        self.lm_predictions = DebertaV2OnlyMLMHead(config)
        self.features_dim = features_dim
        if freeze_mlm:
            for n, p in self.lm_predictions.named_parameters():
                if ft_ln and "LayerNorm" in n:
                    continue
                else:
                    p.requires_grad_(False)

        self.init_weights()
        self.n_ans = n_ans
        if n_ans:
            self.answer_embeddings = nn.Embedding(
                n_ans, self.deberta.embeddings.embedding_size
            )
            self.answer_bias = nn.Parameter(torch.zeros(n_ans))
            if freeze_last:
                self.answer_embeddings.requires_grad_(False)
                self.answer_bias.requires_grad_(False)

    def get_output_embeddings(self):
        return self.lm_predictions.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_predictions.lm_head.decoder = new_embeddings

    def set_answer_embeddings(self, a2tok, freeze_last=True):
        a2v = self.deberta.embeddings.word_embeddings(a2tok)
        pad_token_id = getattr(self.config, "pad_token_id", 0)
        sum_tokens = (a2tok != pad_token_id).sum(1, keepdims=True)  # n_ans
        if len(a2v) != self.n_ans:  # reinitialize the answer embeddings
            assert not self.training
            self.n_ans = len(a2v)
            self.answer_embeddings = nn.Embedding(
                self.n_ans, self.deberta.embeddings.embedding_size
            ).to(self.device)
            self.answer_bias.requires_grad = False
            self.answer_bias.resize_(self.n_ans)
        self.answer_embeddings.weight.data = torch.div(
            (a2v * (a2tok != pad_token_id).float()[:, :, None]).sum(1),
            sum_tokens.clamp(min=1),
        )  # n_ans
        a2b = self.lm_predictions.lm_head.bias[a2tok]
        self.answer_bias.weight = torch.div(
            (a2b * (a2tok != pad_token_id).float()).sum(1), sum_tokens.clamp(min=1)
        )
        if freeze_last:
            self.answer_embeddings.requires_grad_(False)
            self.answer_bias.requires_grad_(False)

    def emd_context_layer(self, encoder_layers, z_states, attention_mask, encoder):
        if attention_mask.dim() <= 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            att_mask = extended_attention_mask.byte()
            attention_mask = att_mask * att_mask.squeeze(-2).unsqueeze(-1)
        elif attention_mask.dim() == 3:
            attention_mask = attention_mask.unsqueeze(1)
        hidden_states = encoder_layers[-2]
        if not self.config.position_biased_input:
            layers = [encoder.layer[-1] for _ in range(2)]
            z_states += hidden_states
            query_states = z_states
            query_mask = attention_mask
            outputs = []
            rel_embeddings = encoder.get_rel_embedding()

            for layer in layers:
                output = layer(
                    hidden_states,
                    query_mask,
                    return_att=False,
                    query_states=query_states,
                    relative_pos=None,
                    rel_embeddings=rel_embeddings,
                )
                query_states = output
                outputs.append(query_states)
        else:
            outputs = [encoder_layers[-1]]

        return outputs

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        return_dict=None,
        video=None,
        video_mask=None,
        mlm=False,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        """

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
            video=video,
            video_mask=video_mask,
        )

        if labels is not None:
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

        # sequence_output = outputs[0]
        modified = self.emd_context_layer(
            encoder_layers=outputs["hidden_states"],
            z_states=outputs["position_embeddings"].repeat(
                input_ids.shape[0] // len(outputs["position_embeddings"]), 1, 1
            ),
            attention_mask=outputs["attention_mask"],
            encoder=self.deberta.encoder,
        )
        bias = None
        if self.n_ans and (not mlm):  # downstream mode
            embeddings = self.answer_embeddings.weight
            bias = self.answer_bias
        else:
            embeddings = self.deberta.embeddings.word_embeddings.weight
        prediction_scores = self.lm_predictions(modified[-1], embeddings, bias)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token

            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size),
                labels.view(-1),  # labels[labels > 0].view(-1)
            )

        if not return_dict:
            output = (prediction_scores,) + outputs[1:]
            return (
                ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
            )

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# copied from transformers.models.bert.BertPredictionHeadTransform with bert -> deberta
class DebertaV2PredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


# copied from transformers.models.bert.BertLMPredictionHead with bert -> deberta
class DebertaV2LMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        )  # only for compatiblity

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias  # only for compatiblity

    def forward(self, hidden_states, embedding_weight, bias=None):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        if bias is not None:
            logits = (
                torch.matmul(hidden_states, embedding_weight.t().to(hidden_states))
                + bias
            )
        else:
            logits = (
                torch.matmul(hidden_states, embedding_weight.t().to(hidden_states))
                + self.bias
            )
        return logits


# copied from transformers.models.bert.BertOnlyMLMHead with bert -> deberta
class DebertaV2OnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # self.predictions = DebertaV2LMPredictionHead(config)
        self.lm_head = DebertaV2LMPredictionHead(config)

    def forward(self, sequence_output, embedding_weight, bias=None):
        prediction_scores = self.lm_head(sequence_output, embedding_weight, bias=bias)
        return prediction_scores
