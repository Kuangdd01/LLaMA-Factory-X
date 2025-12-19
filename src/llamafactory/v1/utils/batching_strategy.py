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

from abc import abstractmethod


class BaseBatchingStrategy:
    """Basic class for batching strategy."""
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
