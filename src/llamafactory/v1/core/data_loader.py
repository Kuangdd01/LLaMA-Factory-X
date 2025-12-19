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


import copy
import sys
from collections import deque
from collections.abc import Generator, Iterator
from dataclasses import dataclass
from typing import Optional

from torch.utils.data import StatefulDataLoader, StatefulDistributedSampler

from ..utils.batching_strategy import BaseBatchingStrategy
from ..utils.logging import get_logger
from ..utils.types import Processor, TorchDataset
from .trainer_utils.data_collator import DataCollator


logger = get_logger(__name__)

# base dataloader
class DistributedDataloader(StatefulDataLoader):
    """Base Distributed DataLoader."""
    dataset: "TorchDataset"
    sampler: "StatefulDistributedSampler"

    def set_epoch(self, epoch: int) -> None:
        if self.sampler is not None and hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)
        elif hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)

# inspired by VeOmni
class DataLoader:
    """Dynamic Batch Size DataLoader.

    Which should contain following components:
    1. dataloader
    2. batching_strategy, aka fetching strategy
        - if default, it will be a fixed batch size strategy
        - if dynamic, it will be a dynamic batch fetching strategy
    3. collate_fn
    4. length
    5. max_iterations, control of the training steps
    6. drop_last, if True, drop last batch if batch size < num_micro_batch
    """

    def __init__(self,
        dataloader: "DistributedDataloader",
        batching_strategy: "BaseBatchingStrategy",
        collate_fn: "DataCollator",
        num_micro_batch: int,
        length: Optional[int] = None,
        drop_last: bool = True
    ) -> None:
        self.batching_strategy = batching_strategy
        self.buffer = deque()
        self.item_buffer = deque()
        self.collate_fn = collate_fn
        self.num_micro_batch = num_micro_batch
        self.drop_last = drop_last
        self.step = 0

        self._dataloader = dataloader
        self._data_iter: Iterator = None
        self._resume = False
        self._batch_data_iter: Generator = None

        if length is None:
            try:
                self._length = len(self._dataloader)
            except Exception as e:
                logger.warning(f"failed to get length of dataloader: {e}, set to `sys.maxsize` instead.")
                self._length = sys.maxsize
        elif length == -1:
            self._length = sys.maxsize
        elif length > 0:
            self._length = length
        else:
            raise ValueError("length must be a positive integer or -1")

    def __len__(self):
        return self._length

    def __iter__(self) -> Iterator:
        if not self._resume:
            self.step = 0
            self._data_iter = iter(self._dataloader)
            self._batch_data_iter = self.batch_data_generator()

        self._resume = False
        return self

    def __next__(self) -> any:
        return next(self._batch_data_iter)

    def _get_processed_micro_batch(self):
        """Retrieves a micro-batch from strategy and applies collation."""
        micro_batch = self.batching_strategy.get_micro_batch(self.step)
        if self._collate_fn:
            micro_batch = self._collate_fn(micro_batch)
        return micro_batch

    def _pad_and_yield_batch(self, batch, last_micro_batch):
        """Pads the remaining batch with the last seen data and yields it."""
        while len(batch) < self.num_micro_batch:
            padding_batch = copy.deepcopy(last_micro_batch)
            padding_batch["is_padded"] = True
            batch.append(padding_batch)

        yield batch
        self.step += 1

    def batch_data_generator(self):
        batch = []
        last_micro_batch = None

        while True:
            if self._length and self.step >= self._length:
                return

            while self.batching_strategy.is_full_filled():
                micro_batch = self._get_processed_micro_batch()
                last_micro_batch = micro_batch # Keep track for potential padding
                batch.append(micro_batch)

                if len(batch) == self.num_micro_batch:
                    yield batch
                    self.step += 1
                    batch = []

            try:
                processing_item = next(self._data_iter)
            except StopIteration:
                break # Exit loop to handle draining/padding
            except Exception as e:
                logger.error(f"iterating dataset error: {e}")
                raise

            items = [processing_item] if isinstance(processing_item, dict) else processing_item
            for item in items:
                self.batching_strategy.put_item(item)

        if self.drop_last:
            return

        while not self.batching_strategy.empty():
            micro_batch = self._get_processed_micro_batch()
            last_micro_batch = micro_batch
            batch.append(micro_batch)

            if len(batch) == self.num_micro_batch:
                yield batch
                self.step += 1
                batch = []

        # Pad partial batch if necessary
        if batch and last_micro_batch:
            yield from self._pad_and_yield_batch(batch, last_micro_batch)

    def state_dict(self):
        state = self.__dict__.copy()
        for k in list(state.keys()):
            if k.startswith("_"):
                del state[k]

        # save dataloader state
        if hasattr(self._dataloader, "state_dict"):
            state["dataloader_state"] = self._dataloader.state_dict()
        elif hasattr(self._dataloader, "__getstate__"):
            state["dataloader_state"] = self._dataloader.__getstate__()

        if hasattr(self.batching_strategy, "state_dict"):
            state["batching_strategy_state"] = self.batching_strategy.state_dict()  # type: ignore
            del state["batching_strategy"]

        return copy.deepcopy(state)

    def load_state_dict(self, state: dict[str, any]):
        if state["num_micro_batch"] != self.num_micro_batch:
            logger.warning(
                f"num_micro_batch changed: [ {state['num_micro_batch']} -> {self.num_micro_batch} ], will clear prefetch buffer"
            )
            del state["num_micro_batch"]
        self.__dict__.update(state)
        self._resume = True

        if hasattr(self._dataloader, "load_state_dict"):
            self._dataloader.load_state_dict(state["dataloader_state"])
        elif hasattr(self._dataloader, "__getstate__"):
            self._dataloader.__setstate__(state["dataloader_state"])

        if "batching_strategy_state" in state:
            self.batching_strategy.load_state_dict(  # type: ignore
                state["batching_strategy_state"]
            )
            del state["batching_strategy_state"]

        self._data_iter = iter(self._dataloader)
        self._batch_data_iter = self.batch_data_generator()

    def set_epoch(self, epoch: int) -> None:
        if hasattr(self._dataloader, "set_epoch"):
            self._dataloader.set_epoch(epoch)


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
    def init_dataloader(self) -> None:
        ### init dataloader
        pass


    def __iter__(self) -> Iterator:
        pass

    def __next__(self) -> any:
        pass
