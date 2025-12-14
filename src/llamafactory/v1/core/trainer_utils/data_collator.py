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

from collections import defaultdict, deque
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Callable, override

import torch
from torch.nn.utils.rnn import pad_sequence  # TODO remove it to other file
from torch.utils.data import IterableDataset
from torch.utils.data._utils.collate import default_collate
from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler

from ....extras.constants import IGNORE_INDEX
from ...utils.seqlen_utils import len2culen
from ...utils.types import Processor, Tensor, TorchDataset


class DataCollator:
    """Default Data collator."""

    def __init__(self, processor: Processor) -> None:
        self.processor = processor

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

class MappingDataCollator(DataCollator):
    """Data collator that supports mapping processed messages to input_ids, attention_mask, labels."""

    def __init__(self, processor: Processor, text_keys: list[str] = None,
                 mapping_fn: Callable[[Any], dict[str, Tensor]] = None, packing: bool = False):
        super().__init__(processor)
        self.text_keys = text_keys or ["messages"]
        self.mapping_fn = mapping_fn # TODO replacde mapping_fn with different func
        self.packing = packing

    def process_one(self, example: dict[str, Any]) -> dict[str, Tensor] | None:
        """Applies mapping to a single raw example. Returns None if invalid."""
        for key in self.text_keys:
            if key in example:
                return self.mapping_fn(example[key]) # broken with return None
        return None

    def collate_batch(self, batch_list: list[dict[str, Tensor]]) -> dict[str, Tensor]:
        """Collates a list of already processed tensors."""
        if not batch_list:
            return {}

        if self.packing:
            seqlens = torch.tensor([len(feature["input_ids"]) for feature in batch_list], dtype=torch.long)
            batch = {"cu_seqlens": len2culen(seqlens)}

            for input_name in batch_list[0].keys():
                if input_name in ("input_ids", "attention_mask", "labels", "position_ids"):
                    batch[input_name] = torch.cat([feature[input_name] for feature in batch_list])
                else:
                    batch[input_name] = default_collate([feature[input_name] for feature in batch_list])
            return batch
        else:
            return super().__call__(batch_list)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Tensor]:
        """Process raw features and collate with appropriate strategy."""
        processed_batch = []
        for example in features:
            processed = self.process_one(example)
            if processed:
                processed_batch.append(processed)
        return self.collate_batch(processed_batch)


@dataclass
class MicroBatchCollator:
    """MicroBatchCollator."""

    num_micro_batches: int
    mini_batch_collator: "DataCollator"

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Tensor]:
        micro_batch_size = len(features) // self.num_micro_batches
        micro_batches = [features[i : i + micro_batch_size] for i in range(0, len(features), micro_batch_size)]
        return [self.mini_batch_collator(micro_batch) for micro_batch in micro_batches]


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


@dataclass
class DataLoader(StatefulDataLoader):

    dataset: TorchDataset
    sampler: StatefulDistributedSampler # Must be a stateful sampler (e.g., torchdata or custom)
    collate_fn: MappingDataCollator
    batch_size: int = 1
    gradient_accumulation_steps: int = 1

    # Configuration
    _buffer_multiplier: int = 2

    # Internal State
    _sampler_iter: Iterator = None
    _raw_data_buffer: deque[Any] = field(default_factory=deque)
    _current_epoch: int = 0
    _exhausted: bool = False
    _stats: dict[str, int] = field(default_factory=lambda: {
        'valid_samples': 0,
        'invalid_samples': 0,
        'steps_yielded': 0
    })

    def __post_init__(self) -> None:
        super().__init__(
            dataset=self.dataset,
            batch_size=self.batch_size,
            sampler=self.sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
            pin_memory_device=self.pin_memory_device,
            drop_last=self.drop_last,
            prefetch_factor=self.prefetch_factor,
        )
        self.buffer_size = self.batch_size * self._buffer_multiplier
        # Flag to indicate we just loaded state and shouldn't clear buffer on first iter
        self._restored_state = False
        self._is_iterable = isinstance(self.dataset, IterableDataset) # do not support currently

        if self.batch_size % self.gradient_accumulation_steps != 0:
            raise ValueError(
                f"Batch size ({self.batch_size}) must be divisible by "
                f"gradient_accumulation_steps ({self.gradient_accumulation_steps})"
            )
            
        self.num_micro_batches = self.gradient_accumulation_steps
        self.micro_batch_size = self.batch_size // self.gradient_accumulation_steps

    @override
    def state_dict(self) -> dict[str, Any]:

        # Ensure sampler has a state_dict (standard in distributed training)
        sampler_state = self.sampler.state_dict() if hasattr(self.sampler, "state_dict") else {}

        return {
            "epoch": self._current_epoch,
            "sampler_state": sampler_state,
            "buffer_contents": list(self._raw_data_buffer),
            "stats": self._stats,
            "exhausted": self._exhausted
        }

    @override
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore state perfectly."""
        self._current_epoch = state_dict["epoch"]
        self._stats = state_dict["stats"]
        self._exhausted = state_dict.get("exhausted", False)

        # 1. Restore state_dict
        if not self._is_iterable and "sampler_state" in state_dict and self.sampler:
            self.sampler.load_state_dict(state_dict["sampler_state"])
        elif self._is_iterable and "dataset_state" in state_dict and hasattr(self.dataset, "load_state_dict"):
            self.dataset.load_state_dict(state_dict["dataset_state"])
        else:
            raise ValueError("Invalid state_dict for dataset or sampler")

        self._raw_data_buffer = deque(state_dict["buffer_contents"])

        self._restored_state = True

    def __iter__(self) -> Iterator[dict[str, Tensor]]:
        if hasattr(self.sampler, 'set_epoch'):
            self.sampler.set_epoch(self._current_epoch)

        # Re-initialize the sampler iterator.
        self._sampler_iter = iter(self.sampler)

        if self._restored_state:
            # If resuming: keep the restored buffer, disable flag
            self._restored_state = False
        else:
            self._raw_data_buffer.clear()
            self._exhausted = False
            self._stats['valid_samples'] = 0
            self._stats['invalid_samples'] = 0

        return self

    def _fetch_next_raw(self) -> Any:
        """Helper to manage buffer refilling from sampler."""
        # Refill buffer if needed and possible
        if len(self._raw_data_buffer) < self.batch_size and not self._exhausted:
            try:
                items_needed = self.buffer_size - len(self._raw_data_buffer)
                for _ in range(items_needed):
                    idx = next(self._sampler_iter)
                    # We fetch the actual data here
                    self._raw_data_buffer.append(self.dataset[idx])
            except StopIteration:
                self._exhausted = True

        if not self._raw_data_buffer:
            raise StopIteration

        return self._raw_data_buffer.popleft()

    def __next__(self) -> dict[str, Tensor]:
        batch_candidates = []

        while len(batch_candidates) < self.batch_size:
            try:
                raw_item = self._fetch_next_raw()

                processed_item = self.collate_fn.process_one(raw_item)

                if processed_item is not None:
                    batch_candidates.append(processed_item)
                    self._stats['valid_samples'] += 1
                else:
                    self._stats['invalid_samples'] += 1

            except StopIteration:
                break

        if not batch_candidates:
            self._current_epoch += 1
            raise StopIteration

        self._stats['steps_yielded'] += 1

        if self.gradient_accumulation_steps > 1:
            # split batch_candidates into micro_batches
            micro_batches = [batch_candidates[i : i + self.micro_batch_size] for i in range(0, len(batch_candidates), self.micro_batch_size)]
            return [self.collate_fn.collate_batch(micro_batch) for micro_batch in micro_batches]
        else:
            return self.collate_fn.collate_batch(batch_candidates)