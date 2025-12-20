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

from abc import ABC, abstractmethod


class BaseBuffer(ABC):
    """Base class for batch buffers.
    Provides common functionality for storing and managing samples.
    """

    def __init__(self):
        self._buffer = []
        self._buffer_sample_lens = []
        self.all_token_cnt = 0

    def append(self, item: dict[str, any]):
        """Append a sample to the buffer.

        Args:
            item: a sample to append to the buffer.
                The sample should be a dict with the following keys:
                    - input_ids: torch.Tensor of shape (seq_len, )
                    - attention_mask: torch.Tensor of shape (seq_len, )
        """
        self._buffer.append(item)
        seq_len = item["attention_mask"].sum().item() if hasattr(item["attention_mask"].sum(), "item") else item["attention_mask"].sum()
        self._buffer_sample_lens.append(seq_len)
        self.all_token_cnt += seq_len

    def __len__(self):
        """Return the number of samples in the buffer."""
        return len(self._buffer)

    def merge(self, buffer_to_merge: "BaseBuffer"):
        """Merge the buffer with another buffer.

        Args:
            buffer_to_merge: the buffer to merge.
        """
        self.flush()
        buffer_to_merge.flush()
        for item in buffer_to_merge._buffer:
            self.append(item)

    @abstractmethod
    def get_samples(self, *args, **kwargs):
        """Get samples from the buffer.
        Subclasses should implement this method with their specific batching logic.
        """
        pass

    @abstractmethod
    def flush(self):
        """Flush the buffer.
        Subclasses should implement this method to clean up their internal state.
        """
        pass


class DynamicBatchBuffer(BaseBuffer):
    """A buffer to store samples for dynamic batch size.
    Batches are formed based on token count rather than sample count.
    """

    def __init__(self):
        super().__init__()
        self.del_idxs = []
        self.cur_idx = 0

    def get_samples(self, n_token_per_iter: int, force: bool = True):
        """Get samples from the buffer based on token count.

        Args:
            n_token_per_iter: the number of tokens to get.
            force: if True, the first sample will be returned even if it is not full.

        Returns:
            samples: a list of samples.
        """
        cum_seq_len = 0
        samples = []
        while self.cur_idx < len(self._buffer) and cum_seq_len < n_token_per_iter:
            seq_len = self._buffer_sample_lens[self.cur_idx]
            if self.cur_idx not in self.del_idxs and (
                (force is True and cum_seq_len == 0) or (seq_len <= n_token_per_iter - cum_seq_len)
            ):
                cum_seq_len += seq_len
                samples.append(self._buffer[self.cur_idx])
                self.del_idxs.append(self.cur_idx)
            self.cur_idx += 1
        assert len(samples) > 0
        return samples

    def flush(self):
        """Flush the buffer by removing deleted items and resetting indices.
        """
        self.cur_idx = 0
        self.all_token_cnt -= sum([self._buffer_sample_lens[idx] for idx in self.del_idxs])
        buffer_len = len(self._buffer)
        self._buffer = [self._buffer[idx] for idx in range(buffer_len) if idx not in self.del_idxs]
        self._buffer_sample_lens = [
            self._buffer_sample_lens[idx] for idx in range(buffer_len) if idx not in self.del_idxs
        ]
        self.del_idxs = []


class FixedBatchBuffer(BaseBuffer):
    """A buffer to store samples for fixed batch size.
    Batches are formed based on sample count.
    """

    def __init__(self, batch_size: int):
        """Initialize a fixed batch size buffer.

        Args:
            batch_size: the number of samples per batch.
        """
        super().__init__()
        self.batch_size = batch_size
        self.cur_idx = 0

    def get_samples(self):
        """Get a batch of samples from the buffer.
        Returns a batch of size batch_size if available, otherwise returns all remaining samples.

        Returns:
            samples: a list of samples (length <= batch_size).
        """
        remaining = len(self._buffer) - self.cur_idx

        if remaining == 0:
            return []

        n_samples = min(self.batch_size, remaining)
        samples = self._buffer[self.cur_idx : self.cur_idx + n_samples]
        self.cur_idx += n_samples

        return samples

    def flush(self):
        """Flush the buffer by removing processed items and resetting index.
        """
        if self.cur_idx > 0:
            # Remove processed items
            self.all_token_cnt -= sum([self._buffer_sample_lens[idx] for idx in range(self.cur_idx)])
            self._buffer = self._buffer[self.cur_idx:]
            self._buffer_sample_lens = self._buffer_sample_lens[self.cur_idx:]
            self.cur_idx = 0

    def is_full(self) -> bool:
        """Check if the buffer has enough samples for a full batch.

        Returns:
            True if buffer has at least batch_size samples, False otherwise.
        """
        return len(self._buffer) >= self.batch_size



class BaseBatchingQueue:
    """Basic class for batching queue."""
    @abstractmethod
    def is_full_filled(self) -> bool:
        """Judge whether the buffer is full and ready for constructing a micro batch."""
        pass

    @abstractmethod
    def put_item(self, item: dict[str, any]):
        """Put an data item into the buffer."""
        pass

    @abstractmethod
    def get_micro_batch(self, step: int) -> any:
        """Get a micro batch from the buffer, Also can design fetch strategy here."""
        pass

    @abstractmethod
    def empty(self) -> bool:
        """Judge whether the buffer is empty. Save state for resume or other purposes."""
        pass

class FixedBatchSizeBatchingQueue(BaseBatchingQueue):
    """Fixed batch size batching queue implementation."""
    pass
