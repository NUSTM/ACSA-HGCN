# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""PyTorch BERT model."""

from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import json
import logging
import math
import os
import sys
import pdb
from io import open

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MultiLabelSoftMarginLoss, BCEWithLogitsLoss
from torchcrf import CRF
import torch.nn.functional as F

from bert_utils.file_utils import cached_path, WEIGHTS_NAME, CONFIG_NAME

logger = logging.getLogger(__name__)

PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin",

}
PRETRAINED_CONFIG_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-config.json",
}
BERT_CONFIG_NAME = 'bert_config.json'
TF_WEIGHTS_NAME = 'model.ckpt'

def prune_linear_layer(layer, index, dim=0):
    """ Prune a linear layer (a model parameters) to keep only entries in index.
        Return the pruned layer as a new layer with requires_grad=True.
        Used to remove heads.
    """
    index = index.to(layer.weight.device)
    W = layer.weight.index_select(dim, index).clone().detach()
    if layer.bias is not None:
        if dim == 1:
            b = layer.bias.clone().detach()
        else:
            b = layer.bias[index].clone().detach()
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(layer.weight.device)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    if layer.bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(b.contiguous())
        new_layer.bias.requires_grad = True
    return new_layer


def load_tf_weights_in_bert(model, tf_checkpoint_path):
    """ Load tf checkpoints in a pytorch model
    """
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        print("Loading a TensorFlow models in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions.")
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    print("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        print("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split('/')
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(n in ["adam_v", "adam_m", "global_step"] for n in name):
            print("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r'[A-Za-z]+_\d+', m_name):
                l = re.split(r'_(\d+)', m_name)
            else:
                l = [m_name]
            if l[0] == 'kernel' or l[0] == 'gamma':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'output_bias' or l[0] == 'beta':
                pointer = getattr(pointer, 'bias')
            elif l[0] == 'output_weights':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'squad':
                pointer = getattr(pointer, 'classifier')
            else:
                try:
                    pointer = getattr(pointer, l[0])
                except AttributeError:
                    print("Skipping {}".format("/".join(name)))
                    continue
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        if m_name[-11:] == '_embeddings':
            pointer = getattr(pointer, 'weight')
        elif m_name == 'kernel':
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        print("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)
    return model


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """
    def __init__(self,
                 vocab_size_or_config_json_file,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02,
                 layer_norm_eps=1e-12):
        """Constructs BertConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
            layer_norm_eps: The epsilon used by LayerNorm.
        """
        if isinstance(vocab_size_or_config_json_file, str) or (sys.version_info[0] == 2
                        and isinstance(vocab_size_or_config_json_file, unicode)):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
            self.layer_norm_eps = layer_norm_eps
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path):
        """ Save this instance to a json file."""
        with open(json_file_path, "w", encoding='utf-8') as writer:
            writer.write(self.to_json_string())

try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
except ImportError:
    logger.info("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex .")
    class BertLayerNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-12):
            """Construct a layernorm module in the TF style (epsilon inside the square root).
            """
            super(BertLayerNorm, self).__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

        def forward(self, x):
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x + self.bias

class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config, output_attentions=False, keep_multihead_output=False):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.output_attentions = output_attentions
        self.keep_multihead_output = keep_multihead_output
        self.multihead_output = None

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, head_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
        if self.keep_multihead_output:
            self.multihead_output = context_layer
            self.multihead_output.retain_grad()

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        if self.output_attentions:
            return attention_probs, context_layer
        return context_layer


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config, output_attentions=False, keep_multihead_output=False):
        super(BertAttention, self).__init__()
        self.output_attentions = output_attentions
        self.self = BertSelfAttention(config, output_attentions=output_attentions,
                                              keep_multihead_output=keep_multihead_output)
        self.output = BertSelfOutput(config)

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.self.num_attention_heads, self.self.attention_head_size)
        for head in heads:
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)
        # Update hyper params
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads

    def forward(self, input_tensor, attention_mask, head_mask=None):
        self_output = self.self(input_tensor, attention_mask, head_mask)
        if self.output_attentions:
            attentions, self_output = self_output
        attention_output = self.output(self_output, input_tensor)
        if self.output_attentions:
            return attentions, attention_output
        return attention_output


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config, output_attentions=False, keep_multihead_output=False):
        super(BertLayer, self).__init__()
        self.output_attentions = output_attentions
        self.attention = BertAttention(config, output_attentions=output_attentions,
                                               keep_multihead_output=keep_multihead_output)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, head_mask=None):
        attention_output = self.attention(hidden_states, attention_mask, head_mask)
        if self.output_attentions:
            attentions, attention_output = attention_output
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        if self.output_attentions:
            return attentions, layer_output
        return layer_output


class BertEncoder(nn.Module):
    def __init__(self, config, output_attentions=False, keep_multihead_output=False):
        super(BertEncoder, self).__init__()
        self.output_attentions = output_attentions
        layer = BertLayer(config, output_attentions=output_attentions,
                                  keep_multihead_output=keep_multihead_output)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True, head_mask=None):
        all_encoder_layers = []
        all_attentions = []
        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states, attention_mask, head_mask[i])
            if self.output_attentions:
                attentions, hidden_states = hidden_states
                all_attentions.append(attentions)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        if self.output_attentions:
            return all_attentions, all_encoder_layers
        return all_encoder_layers


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(bert_model_embedding_weights.size(1),
                                 bert_model_embedding_weights.size(0),
                                 bias=False)
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super(BertOnlyNSPHead, self).__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertPreTrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class BertPreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, config, *inputs, **kwargs):
        super(BertPreTrainedModel, self).__init__()
        if not isinstance(config, BertConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `BertConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        self.config = config

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        """
        Instantiate a BertPreTrainedModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.

        Params:
            pretrained_model_name_or_path: either:
                - a str with the name of a pre-trained model to load selected in the list of:
                    . `bert-base-uncased`
                    . `bert-large-uncased`
                    . `bert-base-cased`
                    . `bert-large-cased`
                    . `bert-base-multilingual-uncased`
                    . `bert-base-multilingual-cased`
                    . `bert-base-chinese`
                    . `bert-base-german-cased`
                    . `bert-large-uncased-whole-word-masking`
                    . `bert-large-cased-whole-word-masking`
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `pytorch_model.bin` a PyTorch dump of a BertForPreTraining instance
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `model.chkpt` a TensorFlow checkpoint
            from_tf: should we load the weights from a locally saved TensorFlow checkpoint
            cache_dir: an optional path to a folder in which the pre-trained models will be cached.
            state_dict: an optional state dictionnary (collections.OrderedDict object) to use instead of Google pre-trained models
            *inputs, **kwargs: additional input for the specific Bert class
                (ex: num_labels for BertForSequenceClassification)
        """
        state_dict = kwargs.get('state_dict', None)
        kwargs.pop('state_dict', None)
        cache_dir = kwargs.get('cache_dir', None)
        kwargs.pop('cache_dir', None)
        from_tf = kwargs.get('from_tf', False)
        kwargs.pop('from_tf', None)

        if pretrained_model_name_or_path in PRETRAINED_MODEL_ARCHIVE_MAP:
            archive_file = PRETRAINED_MODEL_ARCHIVE_MAP[pretrained_model_name_or_path]
            config_file = PRETRAINED_CONFIG_ARCHIVE_MAP[pretrained_model_name_or_path]
        else:
            if from_tf:
                # Directly load from a TensorFlow checkpoint
                archive_file = os.path.join(pretrained_model_name_or_path, TF_WEIGHTS_NAME)
                config_file = os.path.join(pretrained_model_name_or_path, BERT_CONFIG_NAME)
            else:
                archive_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
                config_file = os.path.join(pretrained_model_name_or_path, CONFIG_NAME)
        # redirect to the cache, if necessary
        try:
            resolved_archive_file = cached_path(archive_file, cache_dir=cache_dir)
        except EnvironmentError:
            if pretrained_model_name_or_path in PRETRAINED_MODEL_ARCHIVE_MAP:
                logger.error(
                    "Couldn't reach server at '{}' to download pretrained weights.".format(
                        archive_file))
            else:
                logger.error(
                    "Model name '{}' was not found in model name list ({}). "
                    "We assumed '{}' was a path or url but couldn't find any file "
                    "associated to this path or url.".format(
                        pretrained_model_name_or_path,
                        ', '.join(PRETRAINED_MODEL_ARCHIVE_MAP.keys()),
                        archive_file))
            return None
        try:
            resolved_config_file = cached_path(config_file, cache_dir=cache_dir)
        except EnvironmentError:
            if pretrained_model_name_or_path in PRETRAINED_CONFIG_ARCHIVE_MAP:
                logger.error(
                    "Couldn't reach server at '{}' to download pretrained model configuration file.".format(
                        config_file))
            else:
                logger.error(
                    "Model name '{}' was not found in model name list ({}). "
                    "We assumed '{}' was a path or url but couldn't find any file "
                    "associated to this path or url.".format(
                        pretrained_model_name_or_path,
                        ', '.join(PRETRAINED_CONFIG_ARCHIVE_MAP.keys()),
                        config_file))
            return None
        # if resolved_archive_file == archive_file and resolved_config_file == config_file:
        #     logger.info("loading weights file {}".format(archive_file))
        #     logger.info("loading configuration file {}".format(config_file))
        # else:
        #     logger.info("loading weights file {} from cache at {}".format(
        #         archive_file, resolved_archive_file))
        #     logger.info("loading configuration file {} from cache at {}".format(
        #         config_file, resolved_config_file))
        # Load config
        config = BertConfig.from_json_file(resolved_config_file)
        # logger.info("Model config {}".format(config))
        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        if state_dict is None and not from_tf:
            state_dict = torch.load(resolved_archive_file, map_location='cpu')
        if from_tf:
            # Directly load from a TensorFlow checkpoint
            return load_tf_weights_in_bert(model, weights_path)
        # Load from a PyTorch state_dict
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')
        start_prefix = ''
        if not hasattr(model, 'bert') and any(s.startswith('bert.') for s in state_dict.keys()):
            start_prefix = 'bert.'
        load(model, prefix=start_prefix)
        # if len(missing_keys) > 0:
        #     logger.info("Weights of {} not initialized from pretrained model: {}".format(
        #         model.__class__.__name__, missing_keys))
        # if len(unexpected_keys) > 0:
        #     logger.info("Weights from pretrained model not used in {}: {}".format(
        #         model.__class__.__name__, unexpected_keys))
        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                               model.__class__.__name__, "\n\t".join(error_msgs)))
        return model


class BertModel(BertPreTrainedModel):
    """BERT model ("Bidirectional Embedding Representations from a Transformer").

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model
        `output_attentions`: If True, also output attentions weights computed by the model at each layer. Default: False
        `keep_multihead_output`: If True, saves output of the multi-head attention module with its gradient.
            This can be used to compute head importance metrics. Default: False

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output as described below. Default: `True`.
        `head_mask`: an optional torch.Tensor of shape [num_heads] or [num_layers, num_heads] with indices between 0 and 1.
            It's a mask to be used to nullify some heads of the transformer. 1.0 => head is fully masked, 0.0 => head is not masked.


    Outputs: Tuple of (encoded_layers, pooled_output)
        `encoded_layers`: controled by `output_all_encoded_layers` argument:
            - `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
                of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each
                encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
            - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
                to the last attention block of shape [batch_size, sequence_length, hidden_size],
        `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
            classifier pretrained on top of the hidden state associated to the first character of the
            input (`CLS`) to train on the Next-Sentence task (see BERT's paper).

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = modeling.BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = modeling.BertModel(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config, output_attentions=False, keep_multihead_output=False):
        super(BertModel, self).__init__(config)
        self.output_attentions = output_attentions
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config, output_attentions=output_attentions,
                                           keep_multihead_output=keep_multihead_output)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def get_multihead_outputs(self):
        """ Gather all multi-head outputs.
            Return: list (layers) of multihead module outputs with gradients
        """
        return [layer.attention.self.multihead_output for layer in self.encoder.layer]

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=True, head_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand_as(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers,
                                      head_mask=head_mask)
        if self.output_attentions:
            all_attentions, encoded_layers = encoded_layers
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        if self.output_attentions:
            return all_attentions, encoded_layers, pooled_output
        return encoded_layers, pooled_output


class self_attention_layer(nn.Module):
    def __init__(self, n_hidden):
        """
        Self-attention layer
        * n_hidden [int]: hidden layer number (equal to 2*n_hidden if bi-direction)
        """
        super(self_attention_layer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(n_hidden, 1, bias=False)
        )

    def init_weights(self):
        """
        Initialize all the weights and biases for this layer
        """
        for m in self.attention.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, 0.02)
                nn.init.uniform_(m.bias, -0.02, 0.02)

    def forward(self, inputs, mask=None):
        """
        Forward calculation of the layer
        * inputs [tensor]: input tensor (batch_size * max_seq_len * n_hidden)
        * seq_len [tensor]: sequence length (batch_size,)
        - outputs [tensor]: attention output (batch_size * n_hidden)
        """
        if inputs.dim() != 3 :
            raise ValueError("! Wrong dimemsion of the inputs parameters.")

        now_batch_size, max_seq_len, _ = inputs.size()
        alpha = self.attention(inputs).contiguous().view(now_batch_size, 1, max_seq_len)
        exp = torch.exp(alpha)

        if mask is not None:
            # mask = get_mask(inputs, seq_len)
            # mask = mask.contiguous().view(now_batch_size, 1, max_seq_len)
            mask = mask.unsqueeze(1)
            exp = exp * mask.float()

        sum_exp = exp.sum(-1, True) + 1e-9
        softmax_exp = exp / sum_exp.expand_as(exp).contiguous().view(now_batch_size, 1, max_seq_len)
        outputs = torch.bmm(softmax_exp, inputs).squeeze(-2)
        return outputs

class CNNLayer(nn.Module):
    def __init__(self, input_size, in_channels, out_channels,
                 kernel_width, act_fun=nn.ReLU, drop_prob=0.1):
        """Initilize CNN layer.

        Args:
            input_size [int]: embedding dim or the last dim of the input
            in_channels [int]: number of channels for inputs
            out_channels [int]: number of channels for outputs
            kernel_width [int]: the width on sequence for the first dim of kernel
            act_fun [torch.nn.modules.activation]: activation function
            drop_prob [float]: drop out ratio
        """
        super(CNNLayer, self).__init__()

        self.input_size = input_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_width = kernel_width

        self.conv = nn.Conv2d(in_channels, out_channels, (kernel_width, input_size))
        self.drop_out = nn.Dropout(drop_prob)

        assert callable(act_fun), TypeError("Type error of 'act_fun', use functions like nn.ReLU/nn.Tanh.")
        self.act_fun = act_fun()

    def forward(self, inputs, mask=None, out_type='max'):
        """Forward propagation.

        Args:
            inputs [tensor]: input tensor (batch_size * in_channels * max_seq_len * input_size)
                             or (batch_size * max_seq_len * input_size)
            mask [tensor]: mask matrix (batch_size * max_seq_len)
            out_type [str]: use 'max'/'mean'/'all' to choose

        Returns:
            outputs [tensor]: output tensor (batch_size * out_channels) or (batch_size * left_len * n_hidden)
        """
        # auto extend 3d inputs
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(1).repeat(1, self.in_channels, 1, 1)
        assert inputs.dim() == 4 and inputs.size(1) == self.in_channels, "Dimension error of 'inputs'."
        assert inputs.size(-1) == self.input_size, "Dimension error of 'inputs'."

        now_batch_size, _, max_seq_len, _ = inputs.size()
        assert max_seq_len >= self.kernel_width, "Dimension error of 'inputs'."
        assert out_type in ['max', 'mean', 'all'], ValueError(
            "Value error of 'out_type', only accepts 'max'/'mean'/'all'."
        )

        # calculate the seq_len after convolution
        left_len = max_seq_len - self.kernel_width + 1

        # auto generate full-one mask
        if mask is None:
            mask = torch.ones((now_batch_size, left_len), device=inputs.device)
        assert mask.dim() == 2, "Dimension error of 'mask'."
        mask = mask[:, -left_len:].unsqueeze(1)

        outputs = self.conv(inputs)
        outputs = self.drop_out(outputs)
        outputs = outputs.reshape(-1, self.out_channels, left_len)

        outputs = self.act_fun(outputs)  # batch_size * out_channels * left_len

        # all modes need to consider mask
        if out_type == 'max':
            outputs = outputs.masked_fill(~mask.bool(), -1e10)
            outputs = F.max_pool1d(outputs, left_len).reshape(-1, self.out_channels)
            isinf = outputs.eq(-1e10)
            outputs = outputs.masked_fill(isinf, 0)
        elif out_type == 'mean':
            outputs = outputs.masked_fill(~mask.bool(), 0)
            lens = torch.sum(mask, dim=-1)
            outputs = torch.sum(outputs, dim=-1) / (lens.float() + 1e-9)
        elif out_type == 'all':
            outputs = outputs.masked_fill(~mask.bool(), 0)
            outputs = outputs.transpose(1, 2)  # batch_size * left_len * out_channels
        return outputs

class BertForSequenceClassification(BertPreTrainedModel):

    def __init__(self, config, num_labels=2, output_attentions=False, keep_multihead_output=False):
        super(BertForSequenceClassification, self).__init__(config)
        self.output_attentions = output_attentions
        self.num_labels = num_labels
        self.bert = BertModel(config, output_attentions=output_attentions,
                                      keep_multihead_output=keep_multihead_output)
        self.bert1 = copy.deepcopy(self.bert)
        self.attenion = nn.ModuleList([self_attention_layer(config.hidden_size) for _ in range(2)])
        self.W = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size, bias=False),
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size*2, num_labels)

        self.LayerNorm = BertLayerNorm(128, eps=config.layer_norm_eps)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, aspect_mask=None, labels=None, head_mask=None):
        pooled_outputs, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False, head_mask=head_mask)
        pooled_output = self.attenion[0](pooled_outputs, aspect_mask)
        pooled_outputs1, _ = self.bert1(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False, head_mask=head_mask)
        all_pooled_output = self.attenion[1](pooled_outputs1, attention_mask)
        pooled_output = torch.cat([pooled_output, all_pooled_output], -1)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        # seq_length = pooled_outputs.size(1)
        # pooled_output = torch.norm(pooled_outputs, p=2, dim=-1, keepdim=True)
        # pooled_output = pooled_output * pooled_output
        # pooled_output = pooled_output.repeat(1, 1, seq_length)
        # pooled_output_transpose = pooled_output.permute(0, 2, 1)
        # last_dist = pooled_output + pooled_output_transpose - 2 * torch.bmm(pooled_output, pooled_output_transpose)
        # last_dist = 1.0 / last_dist
        # last_dist = self.LayerNorm(last_dist)
        # features = torch.tanh(self.W(torch.bmm(last_dist, pooled_outputs)))
        # features = torch.tanh(self.W(torch.bmm(last_dist, features)))
        # # features = self.attenion(features, attention_mask)
        # logits = self.classifier(features[:, 0, :])

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        elif self.output_attentions:
            return all_attentions, logits
        return logits


class RNN_layer(nn.Module):
    def __init__(self, input_size, n_hidden, n_layer, drop_prob=0,
                 bi_direction=True, mode="GRU"):
        """
        LSTM layer
        * input_size [int]: embedding dim or the last dim of the input
        * n_hidden [int]: number of hidden layer nodes
        * n_layer [int]: number of classify layers
        * n_hidden [int]: number of hidden layer nodes
        * drop_prob [float]: drop out ratio
        * bi_direction [bool]: use bi-direction model or not
        * mode [str]: use 'tanh'/'LSTM'/'GRU' for core model
        """
        super(RNN_layer, self).__init__()
        mode_model = {'tanh': nn.RNN, 'LSTM': nn.LSTM, 'GRU': nn.GRU}
        self.n_hidden = n_hidden
        self.n_layer = n_layer
        self.bi_direction_num = 2 if bi_direction else 1
        self.mode = mode
        try:
            model = mode_model[mode]
        except:
            raise ValueError("! Parameter 'mode' only receives 'tanh'/'LSTM'/'GRU'.")
        self.rnn = model(
            input_size=input_size,
            hidden_size=n_hidden,
            num_layers=n_layer,
            bias=True,
            batch_first=True,
            dropout=drop_prob if n_layer > 1 else 0,
            bidirectional=bi_direction
        )

    def forward(self, inputs, seq_len=None, out_type='all'):
        """
        Forward calculation of the layer
        * inputs [tensor]: input tensor (batch_size * max_seq_len * input_size)
        * seq_len [tensor]: sequence length (batch_size,)
        * out_type [str]: use 'all'/'last' to choose
        - outputs [tensor]: the last layer (batch_size * max_seq_len * (bi_direction*n_hidden))
        - h_last [tensor]: the last time step of the last layer (batch_size * (bi_direction*n_hidden))
        """
        if inputs.dim() != 3:
            raise ValueError("! Wrong dimemsion of the inputs parameters.")

        now_batch_size, max_seq_len, _ = inputs.size()
        if seq_len is not None:
            sort_seq_len, sort_index = torch.sort(seq_len, descending=True)  # sort seq_len
            _, unsort_index = torch.sort(sort_index, dim=0, descending=False)  # get back index
            sort_seq_len = torch.index_select(sort_seq_len, 0, torch.nonzero(sort_seq_len).contiguous().view(-1))
            n_pad = sort_index.size(0) - sort_seq_len.size(0)
            inputs = torch.index_select(inputs, 0, sort_index[:sort_seq_len.size(0)])
            now_batch_size, max_seq_len, _ = inputs.size()
            inputs = nn.utils.rnn.pack_padded_sequence(inputs, sort_seq_len, batch_first=True)

        self.rnn.flatten_parameters()
        if self.mode == 'tanh' or self.mode == 'GRU':
            outputs, h_last = self.rnn(inputs)  # h_last: (n_layer*bi_direction) * batch_size * n_hidden
        elif self.mode == 'LSTM':
            outputs, (h_last, _) = self.rnn(inputs)  # h_last: (n_layer*bi_direction) * batch_size * n_hidden
        if out_type == 'all':
            if seq_len is not None:
                outputs, _ = nn.utils.rnn.pad_packed_sequence(
                    outputs, batch_first=True, total_length=max_seq_len  # batch_size * seq_len * (2)n_hidden
                )
                outputs = F.pad(outputs, (0, 0, 0, 0, 0, n_pad))
                outputs = torch.index_select(outputs, 0, unsort_index)
            return outputs
        elif out_type == 'last':
            h_last = h_last.contiguous().view(
                self.n_layer, self.bi_direction_num, now_batch_size, self.n_hidden
            )  # n_layer * bi_direction * batch_size * n_hidden
            h_last = torch.reshape(
                h_last[-1].transpose(0, 1),
                [now_batch_size, self.bi_direction_num * self.n_hidden]
            )  # batch_size * (bi_direction*n_hidden)
            if seq_len is not None:
                h_last = F.pad(h_last, (0, 0, 0, n_pad))
                h_last = torch.index_select(h_last, 0, unsort_index)
            return h_last
        else:
            raise ValueError("! Wrong value of parameter 'out-type', accepts 'all'/'last' only.")


class DenseLayer(nn.Module):
    def __init__(self, config):
        super(DenseLayer, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        pooled_output = self.dense(hidden_states)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertForTokenClassification(BertPreTrainedModel):

    def __init__(self, config, num_labels=2, output_attentions=False, keep_multihead_output=False):
        super(BertForTokenClassification, self).__init__(config)
        self.output_attentions = output_attentions
        self.num_labels = num_labels
        self.crf = CRF(num_labels, batch_first=True)
        self.dense = DenseLayer(config)
        self.Bigru = RNN_layer(768, 768, 1)
        self.dense_output = nn.Sequential(
            nn.Linear(768*2, 768),
            nn.Tanh(),
            nn.Linear(768, num_labels)
        )
        self.bert = BertModel(config, output_attentions=output_attentions,
                                      keep_multihead_output=keep_multihead_output)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, labels, token_type_ids=None, attention_mask=None, head_mask=None):
        outputs = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False, head_mask=head_mask)
        if self.output_attentions:
            all_attentions, sequence_output, _ = outputs
        else:
            sequence_output, _ = outputs

        sequence_output = self.dense(sequence_output)
        sequence_output = self.dropout(sequence_output)
        seq_len = torch.sum(attention_mask, dim=-1)
        sequence_output = self.Bigru(sequence_output, seq_len)
        max_seq_len = input_ids.size()[1]
        sequence_output = sequence_output.view(-1, 768*2)
        sequence_output = self.dense_output(sequence_output)
        sequence_output = F.log_softmax(sequence_output, dim=-1)
        sequence_output = sequence_output.view(-1, max_seq_len, self.num_labels)
        pdb.set_trace()
        loss = - self.crf(sequence_output, labels, mask=attention_mask.byte(), reduction='mean')
        pred_tags = self.crf.decode(sequence_output, mask=attention_mask.byte())

        return pred_tags, loss


class BertForFullySharedTokenClassification(BertPreTrainedModel):

    def __init__(self, config, num_labels=2, output_attentions=False, keep_multihead_output=False, data_flag=0):
        super(BertForFullySharedTokenClassification, self).__init__(config)
        self.output_attentions = output_attentions
        self.num_labels = num_labels
        self.data_flag = data_flag
        self.crf = nn.ModuleList([CRF(num_labels, batch_first=True) for _ in range(2)])
        self.dense = nn.ModuleList([DenseLayer(config) for _ in range(2)])
        self.Bigru = nn.ModuleList([RNN_layer(768, 768, 1) for _ in range(2)])
        self.dense_output = nn.ModuleList([nn.Sequential(
            nn.Linear(768*2, 768),
            nn.Tanh(),
            nn.Linear(768, num_labels)
        ) for _ in range(2)])
        self.bert = BertModel(config, output_attentions=output_attentions,
                                      keep_multihead_output=keep_multihead_output)
        self.dropout = nn.ModuleList([nn.Dropout(config.hidden_dropout_prob) for _ in range(2)])
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, labels, token_type_ids=None, attention_mask=None, head_mask=None, is_training=True):
        outputs = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False, head_mask=head_mask)
        if self.output_attentions:
            all_attentions, sequence_output, _ = outputs
        else:
            sequence_output, _ = outputs

        out_module = None
        if is_training:
            out_module = self.data_flag
            self.data_flag = 1 - self.data_flag
        else:
            out_module = 1

        sequence_output = self.dense[out_module](sequence_output)
        sequence_output = self.dropout[out_module](sequence_output)
        seq_len = torch.sum(attention_mask, dim=-1)
        sequence_output = self.Bigru[out_module](sequence_output, seq_len)
        max_seq_len = input_ids.size()[1]
        sequence_output = sequence_output.view(-1, 768*2)
        sequence_output = self.dense_output[out_module](sequence_output)
        sequence_output = F.log_softmax(sequence_output, dim=-1)
        sequence_output = sequence_output.view(-1, max_seq_len, self.num_labels)
        loss = - self.crf[out_module](sequence_output, labels, mask=attention_mask.byte(), reduction='mean')
        pred_tags = self.crf[out_module].decode(sequence_output, mask=attention_mask.byte())

        return pred_tags, loss


class BertForCrossDomainTokenClassification(BertPreTrainedModel):

    def __init__(self, config, num_labels=2, output_attentions=False, keep_multihead_output=False, data_flag=0):
        super(BertForCrossDomainTokenClassification, self).__init__(config)
        self.output_attentions = output_attentions
        self.num_labels = num_labels
        self.data_flag = data_flag
        self.bert = BertModel(config, output_attentions=output_attentions,
                            keep_multihead_output=keep_multihead_output)
        self.bert_st = nn.ModuleList([copy.deepcopy(self.bert) for _ in range(2)])
        self.dense = nn.ModuleList([DenseLayer(config) for _ in range(3)])
        self.dropout = nn.ModuleList([nn.Dropout(config.hidden_dropout_prob) for _ in range(3)])
        self.Bigru = nn.ModuleList([RNN_layer(768, 768, 1) for _ in range(3)])
        self.dense_output = nn.ModuleList([nn.Sequential(
            nn.Linear(768*4, 768),
            nn.Tanh(),
            nn.Linear(768, num_labels)
        ) for _ in range(2)])
        self.crf = nn.ModuleList([CRF(num_labels, batch_first=True) for _ in range(2)])

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, labels, token_type_ids=None, attention_mask=None, head_mask=None, is_training=True):
        outputs = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False, head_mask=head_mask)
        if self.output_attentions:
            all_attentions, common_output, _ = outputs
        else:
            common_output, _ = outputs

        common_output = self.dense[2](common_output)
        common_output = self.dropout[2](common_output)
        seq_len = torch.sum(attention_mask, dim=-1)
        common_output = self.Bigru[2](common_output, seq_len)
        common_output = common_output.view(-1, 768*2)

        out_module = None
        if is_training:
            out_module = self.data_flag
            self.data_flag = 1 - self.data_flag
        else:
            out_module = 1

        sequence_output, _= self.bert_st[out_module](input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False, head_mask=head_mask)

        sequence_output = self.dense[out_module](sequence_output)
        sequence_output = self.dropout[out_module](sequence_output)
        sequence_output = self.Bigru[out_module](sequence_output, seq_len)
        max_seq_len = input_ids.size()[1]
        sequence_output = sequence_output.view(-1, 768*2)
        # pdb.set_trace()
        sequence_output = torch.cat([sequence_output, common_output], -1)
        sequence_output = self.dense_output[out_module](sequence_output)
        sequence_output = F.log_softmax(sequence_output, dim=-1)
        sequence_output = sequence_output.view(-1, max_seq_len, self.num_labels)
        loss = - self.crf[out_module](sequence_output, labels, mask=attention_mask.byte(), reduction='mean')
        pred_tags = self.crf[out_module].decode(sequence_output, mask=attention_mask.byte())

        return pred_tags, loss


class Baseline_0Classification(BertPreTrainedModel):

    def __init__(self, config, num_labels=2, output_attentions=False, keep_multihead_output=False):
        super(Baseline_0Classification, self).__init__(config)
        self.output_attentions = output_attentions
        self.num_labels = num_labels
        self.bert = BertModel(config, output_attentions=output_attentions,
                                      keep_multihead_output=keep_multihead_output)
        self.attenion = self_attention_layer(config.hidden_size)
        self.W = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size, bias=False),
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

        self.LayerNorm = BertLayerNorm(128, eps=config.layer_norm_eps)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, head_mask=None):
        pooled_outputs, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False, head_mask=head_mask)
        pooled_output = self.attenion(pooled_outputs, attention_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            return loss
        elif self.output_attentions:
            return all_attentions, logits
        return logits


class CLSForJCSClassification(BertPreTrainedModel):

    def __init__(self, config, num_labels=2, output_attentions=False, keep_multihead_output=False):
        super(CLSForJCSClassification, self).__init__(config)
        self.output_attentions = output_attentions
        self.num_labels = num_labels
        self.bert = BertModel(config, output_attentions=output_attentions,
                                      keep_multihead_output=keep_multihead_output)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, head_mask=None):
        _, pooled_outputs = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False, head_mask=head_mask)
        pooled_output = self.dropout(pooled_outputs)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            return loss
        elif self.output_attentions:
            return all_attentions, logits
        return logits


class CLSINDForJCSClassification(BertPreTrainedModel):

    def __init__(self, config, num_labels=2, output_attentions=False, keep_multihead_output=False):
        super(CLSINDForJCSClassification, self).__init__(config)
        self.output_attentions = output_attentions
        self.num_labels = [num_labels//3, 3]
        self.bert = BertModel(config, output_attentions=output_attentions,
                                      keep_multihead_output=keep_multihead_output)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.ModuleList([nn.Linear(config.hidden_size, self.num_labels[1]) for i in range(self.num_labels[0])])
        # self.classifier = nn.Linear(config.hidden_size, num_labels)

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, head_mask=None):
        _, pooled_outputs = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False, head_mask=head_mask)
        pooled_output = self.dropout(pooled_outputs)
        # logits = self.classifier(pooled_output)
        logits = torch.cat([self.classifier[i](pooled_output) for i in range(self.num_labels[0])], dim=1)

        if labels is not None:
            sm = torch.nn.Softmax(dim = -1)
            final_sentiment_logits = - torch.log(sm(logits))
            final_sentiment_logits = final_sentiment_logits * labels.float()
            loss = torch.mean(torch.sum(final_sentiment_logits, dim=-1))
            return loss
        elif self.output_attentions:
            return all_attentions, logits
        return logits


class Baseline_2Classification(BertPreTrainedModel):

    def __init__(self, config, num_labels=2, output_attentions=False, keep_multihead_output=False):
        super(Baseline_2Classification, self).__init__(config)
        self.output_attentions = output_attentions
        self.num_labels = [num_labels//3, 3]
        self.bert = BertModel(config, output_attentions=output_attentions,
                                      keep_multihead_output=keep_multihead_output)
        self.attenion = self_attention_layer(config.hidden_size)
        # self.attenion = nn.ModuleList([self_attention_layer(config.hidden_size) for _ in range(num_labels//3)])
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.ModuleList([nn.Linear(config.hidden_size, self.num_labels[1]) for i in range(self.num_labels[0])])
        # self.classifier = nn.Linear(config.hidden_size, num_labels)

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, head_mask=None):
        pooled_outputs, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False, head_mask=head_mask)
        pooled_output = self.attenion(pooled_outputs, attention_mask)
        pooled_output = self.dropout(pooled_output)
        logits = torch.cat([self.classifier[i](pooled_output) for i in range(self.num_labels[0])], dim=1)

        # pooled_output = [self.attenion[i](pooled_outputs, attention_mask) for i in range(self.num_labels[0])]
        # pooled_output = [self.dropout(pooled_output[i]) for i in range(self.num_labels[0])]
        # logits = torch.cat([self.classifier[i](pooled_output[i]) for i in range(self.num_labels[0])], dim=1)

        if labels is not None:
            sm = torch.nn.Softmax(dim = -1)
            final_sentiment_logits = - torch.log(sm(logits))
            final_sentiment_logits = final_sentiment_logits * labels.float()
            loss = torch.mean(torch.sum(final_sentiment_logits, dim=-1))
            return loss
        elif self.output_attentions:
            return all_attentions, logits
        return logits


class BertForHierJCSClassification(BertPreTrainedModel):

    def __init__(self, config, num_labels=2, output_attentions=False, keep_multihead_output=False):
        super(BertForHierJCSClassification, self).__init__(config)
        self.output_attentions = output_attentions
        self.num_labels = [num_labels, 3]
        self.bert = BertModel(config, output_attentions=output_attentions,
                                      keep_multihead_output=keep_multihead_output)
        self.attenion = nn.ModuleList([self_attention_layer(config.hidden_size) for _ in range(2)])
        self.W = nn.ModuleList([
            nn.Linear(config.hidden_size, config.hidden_size, bias=False) for _ in range(num_labels)
        ])
        self.dropout = nn.ModuleList([nn.Dropout(config.hidden_dropout_prob) for _ in range(2)])
        self.classifier = nn.ModuleList([nn.Linear(config.hidden_size, self.num_labels[int(i>=1)]) for i in range(num_labels+1)])

        self.LayerNorm = BertLayerNorm(128, eps=config.layer_norm_eps)
        self.apply(self.init_bert_weights)
        # pdb.set_trace()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, cate_labels=None, senti_labels=None, head_mask=None):
        pooled_outputs, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False, head_mask=head_mask)
        category_pooled_output = self.attenion[0](pooled_outputs, attention_mask)
        category_pooled_output = self.dropout[0](category_pooled_output)
        # sentiment_pooled_output = self.attenion[1](pooled_outputs, attention_mask)
        # sentiment_pooled_output = self.dropout[1](sentiment_pooled_output)
        sentiment_pooled_output = self.dropout[1](pooled_output)

        category_logits = self.classifier[0](category_pooled_output)

        bs = category_logits.shape[0]
        hidden_size = category_pooled_output.shape[-1]
        cate_mask = category_logits.ge(.0).float().view(bs, self.num_labels[0], 1).repeat(1, 1, hidden_size)
        # cate_mask = category_logits.ge(.0).float().view(bs, self.num_labels[0], 1).repeat(1, 1, 3)
        # cate_mask = torch.sigmoid(category_logits).view(bs, self.num_labels[0], 1).repeat(1, 1, 3)
        # pdb.set_trace()
        c_feature = torch.unsqueeze(category_pooled_output, -1)
        s_feature = torch.unsqueeze(sentiment_pooled_output, 1)

        sentiment_logits = torch.cat([self.classifier[i+1](torch.index_select(cate_mask, 1, torch.cuda.LongTensor([i])) * self.W[i](torch.matmul(s_feature, torch.matmul(c_feature, torch.unsqueeze(self.classifier[0].weight[i], 0)) ) )) for i in range(self.num_labels[0])], dim=1)
        # sentiment_logits = sentiment_logits * cate_mask
        # pdb.set_trace()

        cate_loss_fct = BCEWithLogitsLoss()
        cate_loss = cate_loss_fct(category_logits.view(-1, cate_labels.shape[-1]), cate_labels.view(-1, cate_labels.shape[-1]).float() )
        # senti_loss_fct = CrossEntropyLoss()
        sm = torch.nn.Softmax(dim = -1)
        final_sentiment_logits = - torch.log(sm(sentiment_logits))
        final_sentiment_logits = final_sentiment_logits * senti_labels.float()
        senti_loss = torch.mean(torch.sum(final_sentiment_logits, dim=-1))
        loss = cate_loss + senti_loss
        # elif self.output_attentions:
        #     return all_attentions, category_logits, sentiment_logits
        return loss, cate_loss, senti_loss, category_logits, sentiment_logits

class BertForHierJCSCBaselinelassification(BertPreTrainedModel):

    def __init__(self, config, num_labels=2, output_attentions=False, keep_multihead_output=False):
        super(BertForHierJCSCBaselinelassification, self).__init__(config)
        self.output_attentions = output_attentions
        self.num_labels = [num_labels, 3]
        self.bert = BertModel(config, output_attentions=output_attentions,
                                      keep_multihead_output=keep_multihead_output)
        self.attenion = nn.ModuleList([self_attention_layer(config.hidden_size) for _ in range(2)])
        self.W = nn.ModuleList([nn.Linear(config.hidden_size, self.num_labels[1]) for _ in range(num_labels)])
        self.dropout = nn.ModuleList([nn.Dropout(config.hidden_dropout_prob) for _ in range(2)])
        self.classifier = nn.Linear(config.hidden_size, self.num_labels[0])

        self.LayerNorm = BertLayerNorm(128, eps=config.layer_norm_eps)
        self.apply(self.init_bert_weights)
        # pdb.set_trace()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, cate_labels=None, senti_labels=None, head_mask=None):
        pooled_outputs, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False, head_mask=head_mask)
        category_pooled_output = self.attenion[0](pooled_outputs, attention_mask)
        category_pooled_output = self.dropout[0](category_pooled_output)
        # sentiment_pooled_output = self.attenion[1](pooled_outputs, attention_mask)
        # sentiment_pooled_output = self.dropout[1](sentiment_pooled_output)
        sentiment_pooled_output = self.dropout[1](pooled_output)

        category_logits = self.classifier(category_pooled_output)

        bs = category_logits.shape[0]
        hidden_size = category_pooled_output.shape[-1]
        # cate_mask = category_logits.ge(.0).float().view(bs, 30, 1).repeat(1, 1, 3)
        # cate_mask = torch.sigmoid(category_logits).view(bs, 30, 1).repeat(1, 1, 3)
        # pdb.set_trace()
        c_feature = torch.unsqueeze(category_pooled_output, -1)
        s_feature = torch.unsqueeze(sentiment_pooled_output, 1)

        sentiment_logits = torch.cat([self.W[i](s_feature) for i in range(self.num_labels[0])], dim=1)
        # sentiment_logits = sentiment_logits * cate_mask
        # pdb.set_trace()

        cate_loss_fct = BCEWithLogitsLoss()
        cate_loss = cate_loss_fct(category_logits.view(-1, cate_labels.shape[-1]), cate_labels.view(-1, cate_labels.shape[-1]).float() )
        # senti_loss_fct = CrossEntropyLoss()
        sm = torch.nn.Softmax(dim = -1)
        final_sentiment_logits = - torch.log(sm(sentiment_logits))
        # pdb.set_trace()
        final_sentiment_logits = final_sentiment_logits * senti_labels.float()
        senti_loss = torch.mean(torch.sum(final_sentiment_logits, dim=-1))
        loss = cate_loss + senti_loss
        return loss, cate_loss, senti_loss, category_logits, sentiment_logits

class Baseline_1Classification(BertPreTrainedModel):

    def __init__(self, config, num_labels=2, output_attentions=False, keep_multihead_output=False):
        super(Baseline_1Classification, self).__init__(config)
        self.output_attentions = output_attentions
        self.num_labels = [num_labels, 4]
        self.bert = BertModel(config, output_attentions=output_attentions,
                                      keep_multihead_output=keep_multihead_output)
        self.attenion = self_attention_layer(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.ModuleList([nn.Linear(config.hidden_size, self.num_labels[1]) for i in range(self.num_labels[0])])
        # self.classifier = nn.Linear(config.hidden_size, num_labels)

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, senti_labels=None, head_mask=None):
        pooled_outputs, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False, head_mask=head_mask)
        pooled_output = self.attenion(pooled_outputs, attention_mask)
        pooled_output = self.dropout(pooled_output)
        # logits = self.classifier(pooled_output)
        logits = torch.cat([torch.unsqueeze(self.classifier[i](pooled_output), 1) for i in range(self.num_labels[0])], dim=1)

        sm = torch.nn.Softmax(dim = -1)
        final_sentiment_logits = - torch.log(sm(logits))
        try:
            final_sentiment_logits = final_sentiment_logits * senti_labels.float()
        except:
            pdb.set_trace()
        loss = torch.mean(torch.sum(final_sentiment_logits, dim=-1))
        return logits, loss

class HierClassification(BertPreTrainedModel):

    def __init__(self, config, num_labels=2, output_attentions=False, keep_multihead_output=False):
        super(HierClassification, self).__init__(config)
        self.output_attentions = output_attentions
        self.num_labels = [num_labels, 3]
        self.bert = BertModel(config, output_attentions=output_attentions,
                                      keep_multihead_output=keep_multihead_output)
        self.attention = nn.ModuleList([self_attention_layer(config.hidden_size) for _ in range(num_labels)])
        self.W = nn.ModuleList([
            nn.Linear(2*config.hidden_size, config.hidden_size) for _ in range(num_labels)
        ])
        # self.Wc = nn.ModuleList([
        #     nn.Linear(config.hidden_size, config.hidden_size) for _ in range(num_labels)
        # ])
        # self.Bigru = RNN_layer(config.hidden_size, 768, 1)
        # self.dense_output = nn.Sequential(
        #     nn.Linear(768*2, 768),
        #     nn.Tanh(),
        # )
        self.dropout = nn.ModuleList([nn.Dropout(config.hidden_dropout_prob) for _ in range(2)])
        self.classifier_cate = nn.ModuleList([nn.Linear(config.hidden_size, 1) for i in range(num_labels)])
        self.classifier_senti = nn.ModuleList([nn.Linear(config.hidden_size, self.num_labels[1]) for i in range(num_labels)])

        self.LayerNorm = BertLayerNorm(128, eps=config.layer_norm_eps)
        self.apply(self.init_bert_weights)
        # pdb.set_trace()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, cate_labels=None, senti_labels=None, head_mask=None):
        pooled_outputs, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=True, head_mask=head_mask)
        category_pooled_output = [self.attention[i](pooled_outputs[-1], attention_mask) for i in range(self.num_labels[0])]
        category_pooled_output = [self.dropout[0](category_pooled_output[i]) for i in range(self.num_labels[0])]

        sentiment_pooled_output = self.dropout[1](pooled_output)

        category_logits = torch.cat([self.classifier_cate[i](category_pooled_output[i]) for i in range(self.num_labels[0])], dim=-1)

        bs = category_logits.shape[0]
        hidden_size = category_pooled_output[0].shape[-1]

        # cate_mask = category_logits.ge(.0).float().view(bs, self.num_labels[0], 1).repeat(1, 1, hidden_size)
        # cate_mask = torch.unsqueeze(cate_labels.float(), -1).repeat(1, 1, hidden_size)

        c_feature = [torch.unsqueeze(category_pooled_output[i], 1) for i in range(self.num_labels[0])]
        s_feature = torch.unsqueeze(sentiment_pooled_output, 1)

        # sentiment_logits = torch.cat([self.classifier_senti[i](torch.index_select(cate_mask, 1, torch.cuda.LongTensor([i])) * self.W[i](torch.matmul(s_feature, torch.matmul(c_feature[i], self.classifier_cate[i].weight) ) )) for i in range(self.num_labels[0])], dim=1)
        # sentiment_logits = torch.cat([self.classifier_senti[i](self.W[i](torch.cat([s_feature, c_feature[i]], -1)) ) for i in range(self.num_labels[0])], dim=1)
        # sentiment_logits = torch.cat([self.classifier_senti[i](self.W[i](torch.cat([s_feature, torch.unsqueeze(self.classifier_cate[i](category_pooled_output[i]), 1)], -1)) ) for i in range(self.num_labels[0])], dim=1)
        # sentiment_logits = torch.cat([self.classifier_senti[i](torch.index_select(cate_mask, 1, torch.cuda.LongTensor([i])) * torch.tanh(self.W[i](torch.cat([s_feature, self.Wc[i](c_feature[i])], -1))) ) for i in range(self.num_labels[0])], dim=1)
        sentiment_logits = torch.cat([self.classifier_senti[i](s_feature) for i in range(self.num_labels[0])], dim=1)

        # c_features = torch.cat([c_feature[i] for i in range(self.num_labels[0])], dim=1)
        # seq_len = torch.cuda.IntTensor([self.num_labels[0] for i in range(bs)])
        # c_features = self.Bigru(c_features, seq_len)
        # c_features = self.dense_output(c_features)
        # sentiment_logits = torch.cat([self.classifier_senti[i](self.W[i](torch.cat([s_feature, torch.unsqueeze(c_features[:, i, :], 1)], -1))) for i in range(self.num_labels[0])], dim=1)
        # pdb.set_trace()

        cate_loss_fct = BCEWithLogitsLoss()
        cate_loss = cate_loss_fct(category_logits.view(-1, cate_labels.shape[-1]), cate_labels.view(-1, cate_labels.shape[-1]).float() )
        # senti_loss_fct = CrossEntropyLoss()
        sm = torch.nn.Softmax(dim = -1)
        final_sentiment_logits = - torch.log(sm(sentiment_logits))
        final_sentiment_logits = final_sentiment_logits * senti_labels.float()
        senti_loss = torch.mean(torch.sum(final_sentiment_logits, dim=-1))
        loss = cate_loss + senti_loss
        return loss, cate_loss, senti_loss, category_logits, sentiment_logits


class MultiheadAttentionLayer(nn.Module):
    def __init__(self, input_size, n_head=8, drop_prob=0.1):
        """Initilize multi-head attention layer.

        Args:
            input_size [int]: embedding dim or the last dim of the input
            n_head [int]: number of attention heads
            drop_prob [float]: drop out ratio
        """
        super(MultiheadAttentionLayer, self).__init__()

        self.input_size = input_size
        self.attention = nn.MultiheadAttention(input_size, n_head, drop_prob)

    def forward(self, query, key=None, value=None, query_mask=None, key_mask=None):
        """Forward propagation.

        Args:
            query [tensor]: query tensor (batch_size * max_seq_len_query * input_size)
            key [tensor]: key tensor (batch_size * max_seq_len_key * input_size)
            value [tensor]: value tensor (batch_size * max_seq_len_key * input_size)
            query_mask [tensor]: query mask matrix (batch_size * max_seq_len_query)
            key_mask [tensor]: key mask matrix (batch_size * max_seq_len_query)

        Returns:
            outputs [tensor]: output tensor (batch_size * max_seq_len_query * input_size)
        """
        assert query.dim() == 3, "Dimension error of 'query'."
        assert query.size(-1) == self.input_size, "Dimension error of 'query'."
        # set key = query
        if key is None:
            key, key_mask = query, query_mask
        assert key.dim() == 3, "Dimension error of 'key'."
        assert key.size(-1) == self.input_size, "Dimension error of 'key'."
        # set value = key
        value = key if value is None else value
        assert value.dim() == 3, "Dimension error of 'value'."
        assert value.size(-1) == self.input_size, "Dimension error of 'value'."

        assert query.size(0) == key.size(0) == value.size(0), "Dimension match error."
        assert key.size(1) == value.size(1), "Dimension match error of 'key' and 'value'."
        assert query.size(2) == key.size(2) == value.size(2), "Dimension match error."

        if query_mask is not None:
            assert query_mask.dim() == 2, "Dimension error of 'query_mask'."
            assert query_mask.shape == query.shape[:2], "Dimension match error of 'query' and 'query_mask'."

        # auto generate full-one mask
        if key_mask is None:
            key_mask = torch.ones(key.shape[:2], device=query.device)
        assert key_mask.dim() == 2, "Dimension error of 'key_mask'."
        assert key_mask.shape == key.shape[:2], "Dimension match error of 'key' and 'key_mask'."

        # transpose dimension batch_size and max_seq_len
        query, key, value = query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1)
        outputs, _ = self.attention(query, key, value, key_padding_mask=~key_mask.bool())
        # transpose back
        outputs = outputs.transpose(0, 1)
        if query_mask is not None:
            query_mask = query_mask.unsqueeze(2)
            outputs = outputs.masked_fill(~query_mask.bool(), 0)
        return outputs


class TransformerLayer(nn.Module):
    def __init__(self, input_size, n_head=12, feed_dim=None, drop_prob=0.1):
        """Initilize transformer layer.

        Args:
            input_size [int]: embedding dim or the last dim of the input
            n_head [int]: number of attention heads
            feed_dim [int]: hidden matrix dimension
            drop_prob [float]: drop out ratio
        """
        super(TransformerLayer, self).__init__()

        self.input_size = input_size
        self.feed_dim = 4 * self.input_size if feed_dim is None else feed_dim

        self.attention = MultiheadAttentionLayer(input_size, n_head, drop_prob)
        self.drop_out_1 = nn.Dropout(drop_prob)
        self.norm_1 = nn.LayerNorm(input_size)
        self.linear_1 = nn.Linear(input_size, self.feed_dim)
        self.linear_2 = nn.Linear(self.feed_dim, input_size)
        self.drop_out_3 = nn.Dropout(drop_prob)
        self.norm_2 = nn.LayerNorm(input_size)

    def forward(self, query, key=None, value=None, query_mask=None,
                key_mask=None, out_type='all'):
        """Forward propagation.

        Args:
            query [tensor]: query tensor (batch_size * max_seq_len_query * input_size)
            key [tensor]: key tensor (batch_size * max_seq_len_key * input_size)
            value [tensor]: value tensor (batch_size * max_seq_len_key * input_size)
            query_mask [tensor]: query mask matrix (batch_size * max_seq_len_query)
            key_mask [tensor]: key mask matrix (batch_size * max_seq_len_key)
            out_type [str]: use 'first'/'all' to choose

        Returns:
            outputs [tensor]: output tensor (batch_size * input_size)
                              or (batch_size * max_seq_len_query * input_size)
        """
        assert query.dim() == 3, "Dimension error of 'query'."
        assert query.size(-1) == self.input_size, "Dimension error of 'query'."
        assert out_type in ['first', 'all'], ValueError(
            "Value error of 'out_type', only accepts 'first'/'all'."
        )

        outputs = self.attention(query, key, value, query_mask, key_mask)
        # residual connection
        outputs = query + self.drop_out_1(outputs)
        outputs = self.norm_1(outputs)

        temp = self.linear_2(ACT2FN["gelu"](self.linear_1(outputs)))
        # residual connection
        outputs = outputs + self.drop_out_3(temp)
        outputs = self.norm_2(outputs)

        if query_mask is not None:
            query_mask = query_mask.unsqueeze(2)
            outputs = outputs.masked_fill(~query_mask.bool(), 0)

        if out_type == 'first':
            outputs = outputs[:, 0, :]
        return outputs


class Hier_transJCSClassification(BertPreTrainedModel):

    def __init__(self, config, num_labels=2, output_attentions=False, keep_multihead_output=False):
        super(Hier_transJCSClassification, self).__init__(config)
        self.output_attentions = output_attentions
        self.num_labels = [num_labels, 3]
        self.bert = BertModel(config, output_attentions=output_attentions,
                                      keep_multihead_output=keep_multihead_output)
        self.attention = nn.ModuleList([self_attention_layer(config.hidden_size) for _ in range(num_labels)])
        self.W = nn.ModuleList([
            nn.Linear(2*config.hidden_size, config.hidden_size) for _ in range(num_labels)
        ])
        # self.Bigru = nn.ModuleList([RNN_layer(2*config.hidden_size, 768, 1) for _ in range(2)])
        # self.transformer_layer = nn.ModuleList([BertLayer(config, output_attentions=output_attentions,
        #                           keep_multihead_output=keep_multihead_output) for i in range(2)])
        self.transformer_layer = nn.ModuleList([TransformerLayer(config.hidden_size) for i in range(3)])
        self.dense_output = nn.ModuleList([nn.Sequential(
            nn.Linear(768*2, 768),
            nn.Tanh(),
        ) for _ in range(3)])
        # self.BiBert = nn.ModuleList([copy.deepcopy(self.bert) for _ in range(2)])
        # self.crf = CRF(num_labels, batch_first=True)
        self.dropout = nn.ModuleList([nn.Dropout(config.hidden_dropout_prob) for _ in range(4)])
        self.classifier_cate = nn.ModuleList([nn.Linear(config.hidden_size, 1) for i in range(num_labels)])
        self.classifier_senti = nn.ModuleList([nn.Linear(config.hidden_size, self.num_labels[1]) for i in range(num_labels)])

        self.LayerNorm = nn.ModuleList([BertLayerNorm(2*config.hidden_size, eps=config.layer_norm_eps) for _ in range(2)])
        self.trans_W = nn.Linear(num_labels, 3, bias=False)
        self.apply(self.init_bert_weights)

        # pdb.set_trace()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, cate_labels=None, senti_labels=None, head_mask=None):
        pooled_outputs, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=True, head_mask=head_mask)
        category_pooled_output = [self.attention[i](pooled_outputs[-1], attention_mask) for i in range(self.num_labels[0])]
        category_pooled_output = [self.dropout[0](category_pooled_output[i]) for i in range(self.num_labels[0])]

        sentiment_pooled_output = self.dropout[1](pooled_output)

        # category_logits = torch.cat([self.classifier_cate[i](category_pooled_output[i]) for i in range(self.num_labels[0])], dim=-1)

        bs = sentiment_pooled_output.shape[0]
        hidden_size = category_pooled_output[0].shape[-1]

        c_feature = [torch.unsqueeze(category_pooled_output[i], 1) for i in range(self.num_labels[0])]
        s_feature = torch.unsqueeze(sentiment_pooled_output, 1)

        #model 1
        # sequence_output = self.dense_output[out_module](sequence_output)
        # sequence_output = F.log_softmax(sequence_output, dim=-1)
        # sequence_output = sequence_output.view(-1, max_seq_len, self.num_labels)
        # loss = - self.crf(sequence_output, labels, mask=attention_mask.byte(), reduction='mean')
        # pred_tags = self.crf.decode(sequence_output, mask=attention_mask.byte())

        #model 2
        # c_features = torch.cat([c_feature[i] for i in range(self.num_labels[0])], dim=1)
        # seq_len = torch.cuda.IntTensor([self.num_labels[0] for i in range(bs)])
        # c_features = self.Bigru[0](c_features, seq_len)
        # c_features = self.dense_output[0](c_features)
        # c_features = self.LayerNorm[0](c_features)
        # # category_logits = torch.cat([self.classifier_cate[i](c_features[:, i, :]) for i in range(self.num_labels[0])], dim=-1)
        # sentiment_logits = torch.cat([self.classifier_senti[i](self.W[i](torch.cat([s_feature, torch.unsqueeze(c_features[:, i, :], 1)], -1))) for i in range(self.num_labels[0])], dim=1)

        #model B : senti fusion
        # c_features = torch.cat([c_feature[i] for i in range(self.num_labels[0])], dim=1)
        # seq_len = torch.cuda.IntTensor([[1 if (j<self.num_labels[0]) else 0 for j in range(attention_mask.shape[-1])] for i in range(bs)])
        # s_feature = s_feature.repeat(1, self.num_labels[0], 1)
        # final_feature = torch.cat([c_features, s_feature], dim=-1)
        # final_feature = self.dense_output[0](final_feature)
        # final_feature = self.dropout[2](final_feature)
        # final_feature = self.transformer_layer[0](final_feature)
        # # final_feature = self.LayerNorm[0](final_feature)
        # sentiment_logits = torch.cat([self.classifier_senti[i](torch.unsqueeze(final_feature[:, i, :], 1) ) for i in range(self.num_labels[0])], dim=1)
        # pdb.set_trace()

        #model C : senti fusion-cate
        # c_features = torch.cat([c_feature[i] for i in range(self.num_labels[0])], dim=1)
        # seq_len = seq_len = torch.cuda.IntTensor([[1 if (j<self.num_labels[0]) else 0 for j in range(attention_mask.shape[-1])] for i in range(bs)])
        # s_feature = s_feature.repeat(1, self.num_labels[0], 1)
        # final_feature = torch.cat([c_features, s_feature], dim=-1)
        # final_feature = self.dense_output[0](final_feature)
        # final_feature = self.dropout[2](final_feature)
        # final_feature = self.transformer_layer[0](final_feature)
        # sentiment_logits = torch.cat([self.classifier_senti[i](torch.unsqueeze(final_feature[:, i, :], 1) ) for i in range(self.num_labels[0])], dim=1)
        # c_features = torch.cat([c_features, final_feature], dim=-1)
        # category_logits = torch.cat([self.classifier_cate[i](self.W[i](c_features[:, i, :])) for i in range(self.num_labels[0])], dim=-1)

        #model 3.1 : cate-senti fusion
        # c_features = torch.cat([c_feature[i] for i in range(self.num_labels[0])], dim=1)
        # seq_len = torch.cuda.IntTensor([self.num_labels[0] for i in range(bs)])
        # s_feature = s_feature.repeat(1, self.num_labels[0], 1)
        # final_feature = torch.cat([c_features, s_feature], dim=-1)
        # final_feature = self.Bigru[0](final_feature, seq_len)
        # final_feature = self.LayerNorm[0](final_feature)
        # final_feature = self.dropout[2](final_feature)
        # final_feature = self.dense_output[0](final_feature)
        # category_logits = torch.cat([self.classifier_cate[i](final_feature[:, i, :] ) for i in range(self.num_labels[0])], dim=-1)
        # s_features = torch.cat([s_feature, final_feature], dim=-1)
        # s_features = self.Bigru[1](s_features, seq_len)
        # s_features = self.LayerNorm[1](s_features)
        # s_features= self.dropout[3](s_features)
        # s_features = self.dense_output[1](s_features)
        # sentiment_logits = torch.cat([self.classifier_senti[i](torch.unsqueeze(s_features[:, i, :], 1)) for i in range(self.num_labels[0])], dim=1)

        #model D : senti-cate fusion
        c_features = torch.cat([c_feature[i] for i in range(self.num_labels[0])], dim=1)
        seq_len = torch.cuda.IntTensor([[1 if (j<self.num_labels[0]) else 0 for j in range(attention_mask.shape[-1])] for i in range(bs)])
        s_feature = s_feature.repeat(1, self.num_labels[0], 1)

        final_feature = torch.cat([c_features, s_feature], dim=-1)
        final_feature = self.dense_output[0](final_feature)
        # final_feature = self.dropout[2](final_feature)
        final_feature = self.transformer_layer[0](final_feature)
        sentiment_logits = torch.cat([self.classifier_senti[i](torch.unsqueeze(final_feature[:, i, :], 1) ) for i in range(self.num_labels[0])], dim=1)

        c_features = torch.cat([c_features, final_feature], dim=-1)
        c_features = self.dense_output[1](c_features)
        # # # c_features = self.dropout[3](c_features)
        c_features = self.transformer_layer[1](c_features)
        category_logits = torch.cat([self.classifier_cate[i](c_features[:, i, :]) for i in range(self.num_labels[0])], dim=-1)

        final_feature = torch.cat([final_feature, c_features], dim=-1)
        final_feature = self.dense_output[2](final_feature)
        # # # # final_feature = self.transformer_layer[2](final_feature)
        sentiment_logits = torch.cat([self.classifier_senti[i](torch.unsqueeze(final_feature[:, i, :], 1) ) for i in range(self.num_labels[0])], dim=1)

        #cate -> senti constraint
        # trans_weight = self.trans_W.repeat(bs, 1, 1)
        # pdb.set_trace()
        # cate_constraint = torch.cat([torch.mul(category_logits[i, :], self.trans_W.weight) for i in range(bs)], dim=0)
        # sentiment_logits = torch.mul(sentiment_logits, cate_constraint.reshape([-1, self.num_labels[0], self.num_labels[1]]))

        #inter-transformer
        # final_feature = self.transformer_layer[0](s_feature, key=c_features)
        # sentiment_logits = torch.cat([self.classifier_senti[i](torch.unsqueeze(final_feature[:, i, :], 1) ) for i in range(self.num_labels[0])], dim=1)

        # c_features = self.transformer_layer[1](c_features, key=final_feature)
        # category_logits = torch.cat([self.classifier_cate[i](c_features[:, i, :]) for i in range(self.num_labels[0])], dim=-1)

        cate_loss_fct = BCEWithLogitsLoss()
        cate_loss = cate_loss_fct(category_logits.view(-1, cate_labels.shape[-1]), cate_labels.view(-1, cate_labels.shape[-1]).float() )

        sm = torch.nn.Softmax(dim = -1)
        final_sentiment_logits = - torch.log(sm(sentiment_logits))
        final_sentiment_logits = final_sentiment_logits * senti_labels.float()
        senti_loss = torch.mean(torch.sum(final_sentiment_logits, dim=-1))
        loss = cate_loss + senti_loss
        return loss, cate_loss, senti_loss, category_logits, sentiment_logits


class GCNclassification(BertPreTrainedModel):

    def __init__(self, config, num_labels=2, output_attentions=False, keep_multihead_output=False):
        super(GCNclassification, self).__init__(config)
        self.output_attentions = output_attentions
        self.num_labels = [num_labels, 3]
        self.bert = BertModel(config, output_attentions=output_attentions,
                                      keep_multihead_output=keep_multihead_output)
        self.attention = nn.ModuleList([self_attention_layer(config.hidden_size) for _ in range(num_labels)])
        self.W = nn.ModuleList([
            nn.Linear(2*config.hidden_size, config.hidden_size) for _ in range(num_labels)
        ])

        self.iter = 2

        self.gcn_W = nn.ModuleList(
            [nn.Linear(config.hidden_size, config.hidden_size, bias=False) for i in range(self.iter)]
        )
        self.cate_senti_gcnW = nn.ModuleList(
            [nn.Linear(config.hidden_size, config.hidden_size, bias=False) for i in range(self.iter*3)]
        )

        self.transformer_layer = nn.ModuleList([TransformerLayer(config.hidden_size) for i in range(3)])
        self.dense_output = nn.ModuleList([nn.Sequential(
            nn.Linear(768*2, 768),
            nn.Tanh(),
        ) for _ in range(self.iter)])
        self.cs_dense_output = nn.ModuleList([nn.Sequential(
            nn.Linear(768*2, 768),
            nn.Tanh(),
        ) for _ in range(self.iter)])
        self.dense_outputs = nn.ModuleList([nn.Sequential(
            nn.Linear(768, 768),
            nn.Tanh(),
        ) for _ in range(self.iter)])

        self.trade_weight = nn.Parameter(torch.ones(num_labels, 3))

        self.dropout = nn.ModuleList([nn.Dropout(config.hidden_dropout_prob) for _ in range(2+2*self.iter+3)])
        self.classifier_cate = nn.ModuleList([nn.Linear(config.hidden_size, 1) for i in range(num_labels)])
        self.classifier_senti = nn.ModuleList([nn.Linear(config.hidden_size, self.num_labels[1]) for i in range(num_labels)])

        # self.CateSentiNorm = nn.ModuleList([BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps) for _ in range(3*self.iter)])
        self.SentiNorm = nn.ModuleList([BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps) for _ in range(3*self.iter)])
        self.CateNorm = nn.ModuleList([BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps) for _ in range(self.iter)])
        self.apply(self.init_bert_weights)

    def forward(self, epoch, category_map, input_ids, token_type_ids=None, attention_mask=None, cate_labels=None, senti_labels=None, head_mask=None):
        pooled_outputs, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=True, head_mask=head_mask)
        category_pooled_output = [self.attention[i](pooled_outputs[-1], attention_mask) for i in range(self.num_labels[0])]
        category_pooled_output = [self.dropout[0](category_pooled_output[i]) for i in range(self.num_labels[0])]

        sentiment_pooled_output = self.dropout[1](pooled_output)

        bs = sentiment_pooled_output.shape[0]
        hidden_size = category_pooled_output[0].shape[-1]

        c_feature = torch.cat([torch.unsqueeze(category_pooled_output[i], 1) for i in range(self.num_labels[0])], dim=1)
        s_feature = torch.unsqueeze(sentiment_pooled_output, 1)
        s_feature = s_feature.repeat(1, self.num_labels[0], 1)

        map_hat = torch.add(category_map[0], torch.eye(category_map[0].shape[0]).cuda())
        deg = torch.div(map_hat, torch.diag(map_hat))

        for it in range(self.iter):
            if it > 0:
                s_feature = final_features
                c_feature = torch.cat([c_feature, s_feature], dim=-1)
                c_feature = self.dropout[2+2*(it-1)+1](self.cs_dense_output[it](c_feature))
                c_feature = self.gcn_W[it](torch.matmul(deg, c_feature))
                c_feature = self.CateNorm[it](c_feature)

            final_feature = torch.cat([c_feature, s_feature], dim=-1)
            final_feature = self.dense_output[it](final_feature)
            final_features = []

            for i in range(3):
                cur_feature = final_feature
                map_hat = torch.add(category_map[1][i], torch.eye(category_map[1][i].shape[0]).cuda())
                deg = torch.div(map_hat, torch.diag(map_hat))
                cur_feature = gelu(self.cate_senti_gcnW[3*it+i](torch.matmul(deg, cur_feature)))
                if i == 0:
                    final_features = self.dense_outputs[0](cur_feature)
                else:
                    final_features = torch.max(final_features, self.dense_outputs[0](cur_feature))

            final_features = self.SentiNorm[it](final_features)

        final_features = self.dropout[-1](final_features)

        sentiment_logits = torch.cat([self.classifier_senti[0](torch.unsqueeze(final_features[:, i, :], 1) ) for i in range(self.num_labels[0])], dim=1)
        category_logits = torch.cat([self.classifier_cate[i]((c_feature[:, i, :])) for i in range(self.num_labels[0])], dim=-1)
        cate_loss_fct = BCEWithLogitsLoss()
        cate_loss = cate_loss_fct(category_logits.view(-1, cate_labels.shape[-1]), cate_labels.view(-1, cate_labels.shape[-1]).float() )

        sm = torch.nn.Softmax(dim = -1)
        final_sentiment_logits = - torch.log(sm(sentiment_logits))
        final_sentiment_logits = final_sentiment_logits * senti_labels.float()
        senti_loss = torch.mean(torch.sum(final_sentiment_logits, dim=-1))
        loss = cate_loss + senti_loss
        return loss, cate_loss, senti_loss, category_logits, sentiment_logits