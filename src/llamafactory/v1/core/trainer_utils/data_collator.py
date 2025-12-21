# Copyright 2025 the LlamaFactory team.
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

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data._utils.collate import default_collate

from ....extras.constants import IGNORE_INDEX
from ...plugins.data_plugins.template import QwenTemplate
from ...utils.types import Processor, Tensor


class DataCollator:
    """Default Data collator."""
    processor: "Processor" # processor name -> map to encode_messages function

    def __post_init__(self):
        # callback for text tokenizer
        self.tokenizer = self.processor.tokenizer if hasattr(self.processor, "tokenizer") else self.processor

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Tensor]:
        """Collate features into a batch."""
        batch = defaultdict(list)

        # batching features
        for feature in features:
            for key in feature.keys():
                batch[key].append(feature[key])

        for key in batch.keys():
            # process padding features
            if key in ["input_ids", "attention_mask", "position_ids"]:
                padding_value = self.tokenizer.pad_token_id if key == "input_ids" else 0
                batch[key] = pad_sequence(batch[key], batch_first=True, padding_value=padding_value)
            elif key in ["labels"]:
                batch[key] = pad_sequence(batch[key], batch_first=True, padding_value=IGNORE_INDEX)
            else:
                batch[key] = default_collate(batch[key])

        return batch
        # sft: messages
        # dpo: chosen_messages, rejected_messages


@dataclass
class DefaultCollator(DataCollator):
    """example for now"""
    processor: "Processor" # processor name -> map to encode_messages function
    template: QwenTemplate = QwenTemplate()

    def __call__(self, messages: list[list[dict[str, Any]]]) -> dict[str, Tensor]:
        features = []

        for message in messages:
            encoded_message = self.template.encode_messages(self.tokenizer, message)
            encoded_message = {k: torch.tensor(v, dtype=torch.long) for k, v in encoded_message.items()}
            features.append(encoded_message)

        return super().__call__(features)
