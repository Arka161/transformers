# coding=utf-8
# Copyright 2020, The MUST Authors and HuggingFace Inc.
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
""" must model configuration """
from collections import OrderedDict
from typing import Any, Dict, Iterable, Mapping, Optional

from transformers import PreTrainedTokenizer, TensorType

from ... import is_torch_available
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfigWithPast
from ...utils import logging


logger = logging.get_logger(__name__)

MUST_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "must-small": "https://huggingface.co/t5-small/resolve/main/config.json",
    "must-base": "https://huggingface.co/t5-base/resolve/main/config.json",
    "must-large": "https://huggingface.co/t5-large/resolve/main/config.json",
    "must-3b": "https://huggingface.co/t5-3b/resolve/main/config.json",
    "must-11b": "https://huggingface.co/t5-11b/resolve/main/config.json",
}


class MustConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.SwitchModel` or a
    :class:`~transformers.TFSwitchModel`. It is used to instantiate a Switch model according to the specified arguments,
    defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration
    to that of the Switch `switch-small <https://huggingface.co/switch-small>`__ architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.

    Arguments:
        vocab_size (:obj:`int`, `optional`, defaults to 32128):
            Vocabulary size of the Switch model. Defines the number of different tokens that can be represented by the
            :obj:`inputs_ids` passed when calling :class:`~transformers.SwitchModel` or :class:`~transformers.TFSwitchModel`.
        d_model (:obj:`int`, `optional`, defaults to 512):
            Size of the encoder layers and the pooler layer.
        d_kv (:obj:`int`, `optional`, defaults to 64):
            Size of the key, query, value projections per attention head. :obj:`d_kv` has to be equal to :obj:`d_model
            // num_heads`.
        d_ff (:obj:`int`, `optional`, defaults to 2048):
            Size of the intermediate feed forward layer in each :obj:`SwitchBlock`.
        num_layers (:obj:`int`, `optional`, defaults to 6):
            Number of hidden layers in the Transformer encoder.
        num_decoder_layers (:obj:`int`, `optional`):
            Number of hidden layers in the Transformer decoder. Will use the same value as :obj:`num_layers` if not
            set.
        num_heads (:obj:`int`, `optional`, defaults to 8):
            Number of attention heads for each attention layer in the Transformer encoder.
        relative_attention_num_buckets (:obj:`int`, `optional`, defaults to 32):
            The number of buckets to use for each attention layer.
        dropout_rate (:obj:`float`, `optional`, defaults to 0.1):
            The ratio for all dropout layers.
        layer_norm_eps (:obj:`float`, `optional`, defaults to 1e-6):
            The epsilon used by the layer normalization layers.
        initializer_factor (:obj:`float`, `optional`, defaults to 1):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).
        feed_forward_proj (:obj:`string`, `optional`, defaults to :obj:`"relu"`):
            Type of feed forward layer to be used. Should be one of :obj:`"relu"` or :obj:`"gated-gelu"`. Switchv1.1 uses
            the :obj:`"gated-gelu"` feed forward projection. Original Switch uses :obj:`"relu"`.
        use_cache (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
    """
    model_type = "must"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"hidden_size": "d_model", "num_attention_heads": "num_heads", "num_hidden_layers": "num_layers"}

    def __init__(
        self,
        vocab_size=32128,
        d_model=512,
        d_kv=64,
        d_ff=2048,
        num_layers=6,
        num_decoder_layers=None,
        num_heads=8,
        relative_attention_num_buckets=32,
        dropout_rate=0.1,
        layer_norm_epsilon=1e-6,
        initializer_factor=1.0,
        feed_forward_proj="relu",
        is_encoder_decoder=True,
        use_cache=True,
        pad_token_id=0,
        eos_token_id=1,
        capacity_factor : float = 1,
        drop_token : bool = False,
        n_experts : int = 2,
        load_balancing_loss_ceof : float = 0.01,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_kv = d_kv
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.num_decoder_layers = (
            num_decoder_layers if num_decoder_layers is not None else self.num_layers
        )  # default = symmetry
        self.num_heads = num_heads
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.dropout_rate = dropout_rate
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_factor = initializer_factor
        self.feed_forward_proj = feed_forward_proj
        self.use_cache = use_cache
        self.capacity_factor = capacity_factor
        self.n_experts=n_experts
        self.load_balancing_loss_ceof = load_balancing_loss_ceof
        self.drop_token = drop_token
        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            **kwargs,
        )

