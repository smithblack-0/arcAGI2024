"""
Sliding window recurrent long_term_memories. This contains the mechanisms for sliding
window long_term_memories, including the creation and update mechanisms
"""

import torch
from torch import nn
from typing import Tuple, Any, Optional, List
from .base import RecurrentSelfAttention, recurrent_short_term_attention_registry


class Window:
    """
    A container holding the three sliding features.

    For memory usage reasons, we use a list rather than
    concatenating tensors together in the storage
    object, and only concatenate when getting
    the current result.

    You can then get the stacked version through the
    respective properties.
    """

    @property
    def query(self) -> torch.Tensor:
        return torch.stack(self._queries, dim=-2)

    @property
    def keys(self) -> torch.Tensor:
        return torch.stack(self._keys, dim=-2)

    @property
    def values(self) -> torch.Tensor:
        return torch.stack(self._values, dim=-2)

    def __init__(self, max_len: int):
        self._queries: List[torch.Tensor] = []
        self._keys: List[torch.Tensor] = []
        self._values: List[torch.Tensor] = []
        self.max_len = max_len

    def update(self,
               query: torch.Tensor,
               key: torch.Tensor,
               value: torch.Tensor
               ):
        """
        Integrates the given values as updates into the window.
        Pops off excess if needed.
        :param query: The query
        :param key: The key
        :param value: The value
        """
        # Insert the updates
        self._queries.append(query)
        self._keys.append(key)
        self._values.append(value)

        # If we are over length, pop the oldest feature.
        if len(self._queries) > self.max_len:
            self._queries.pop(0)
            self._keys.pop(0)
            self._values.pop(0)


@recurrent_short_term_attention_registry.register('SlidingWindowAttention')
class SlidingWindowAttention(RecurrentSelfAttention):
    """
    Maintains a sliding window of recurrent context,
    and updates it when called. Performs normal long_term_memories
    within this window.
    """

    def __init__(self,
                 d_model: int,
                 d_key: int,
                 d_value: int,
                 d_head: int,
                 num_heads: int,
                 dropout: float,
                 window_size: int
                 ):
        assert d_model % d_head == 0
        super().__init__(d_model, d_key, d_value, d_head, num_heads, dropout)
        self.window_size = window_size

        # Create the long_term_memories mechanism
        self.mha = nn.MultiheadAttention(d_model, num_heads, dropout,
                                         kdim=d_key, vdim=d_value, batch_first=True
                                         )

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                state: Optional[Window] = None
                ) -> Tuple[torch.Tensor, Any]:
        """
        Performs multiheaded long_term_memories against the sliding window, then
        updates the sliding window.
        :param query: The query
        :param key: The key
        :param value: The value
        :param state: The state
        :return: The tensor, and the new state
        """

        # Handle window setup if it has not already been made
        if state is None:
            state = Window(self.window_size)

        # Update the window with the inputs
        state.update(query, key, value)

        # Perform long_term_memories, return result
        return self.mha(query.unsqueeze(-2), state.keys, state.values)
