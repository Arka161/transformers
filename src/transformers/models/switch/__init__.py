# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

# Copyright 2020 The HuggingFace Team. All rights reserved.
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

from typing import TYPE_CHECKING

from ...file_utils import (
    _LazyModule,
    is_flax_available,
    is_sentencepiece_available,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)


_import_structure = {
    "configuration_switch": ["Switch_PRETRAINED_CONFIG_ARCHIVE_MAP", "SwitchConfig", "SwitchOnnxConfig"],
}

if is_sentencepiece_available():
    _import_structure["tokenization_t5"] = ["T5Tokenizer"]

if is_tokenizers_available():
    _import_structure["tokenization_t5_fast"] = ["T5TokenizerFast"]

if is_torch_available():
    _import_structure["modeling_switch"] = [
        "Switch_PRETRAINED_MODEL_ARCHIVE_LIST",
        "SwitchEncoderModel",
        "SwitchForConditionalGeneration",
        "SwitchModel",
        "SwitchPreTrainedModel",
        "load_tf_weights_in_switch",
    ]

if is_tf_available():
    _import_structure["modeling_tf_switch"] = [
        "TF_Switch_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFSwitchEncoderModel",
        "TFSwitchForConditionalGeneration",
        "TFSwitchModel",
        "TFSwitchPreTrainedModel",
    ]

if is_flax_available():
    _import_structure["modeling_flax_switch"] = [
        "FlaxSwitchForConditionalGeneration",
        "FlaxSwitchModel",
        "FlaxSwitchPreTrainedModel",
    ]


if TYPE_CHECKING:
    from .configuration_switch import Switch_PRETRAINED_CONFIG_ARCHIVE_MAP, SwitchConfig, SwitchOnnxConfig

    if is_sentencepiece_available():
        from .tokenization_t5 import T5Tokenizer

    if is_tokenizers_available():
        from .tokenization_t5_fast import T5TokenizerFast

    if is_torch_available():
        from .modeling_switch import (
            Switch_PRETRAINED_MODEL_ARCHIVE_LIST,
            SwitchEncoderModel,
            SwitchForConditionalGeneration,
            SwitchModel,
            SwitchPreTrainedModel,
            load_tf_weights_in_switch,
        )

    if is_tf_available():
        from .modeling_tf_switch import (
            TF_Switch_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFSwitchEncoderModel,
            TFSwitchForConditionalGeneration,
            TFSwitchModel,
            TFSwitchPreTrainedModel,
        )

    if is_flax_available():
        from .modeling_flax_switch import FlaxSwitchForConditionalGeneration, FlaxSwitchModel, FlaxSwitchPreTrainedModel


else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
