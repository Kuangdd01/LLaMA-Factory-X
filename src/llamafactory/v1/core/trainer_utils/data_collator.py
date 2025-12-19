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

from torch.nn.utils.rnn import pad_sequence  # TODO remove it to other file
from torch.utils.data._utils.collate import default_collate

from ....extras.constants import IGNORE_INDEX
from ...utils.types import Processor, Tensor, TorchDataset


class DataCollator:
    """Default Data collator."""

    def __init__(self, processor: Processor) -> None:
        self.processor = processor
        # get our template function here

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
                batch[key] = pad_sequence(batch[key], batch_first=True, padding_value=0)
            elif key in ["labels"]:
                batch[key] = pad_sequence(batch[key], batch_first=True, padding_value=IGNORE_INDEX)
            else:
                batch[key] = default_collate(batch[key])

        return batch
        # sft: messages
        # dpo: chosen_messages, rejected_messages


@dataclass
class BaseDataLoader:
    """Default DataLoader."""

    processor: Processor

    def __init__(self, dataset: TorchDataset) -> None:
        self.dataset = dataset
        # guidlines: fetch until get fixed batchsize.
        # save state_dict for buffer.
        # resume with state

        # 1. Init stateful dataloader (tokenize)
        # 2. Add to buffer (2 * max seq len per device)
        # 3. Yield batch indexes (micro batch * grad acc)
        #    a ) non pack + non dynamic
        #    b ) non pack + dynamic
        #    c ) pack + non dynamic
        #    d ) pack + dynamic
