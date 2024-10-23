"""
Long term memories are stored in mechanism and ways that are designed
to continue operating successfully over MUCH longer durations during the
short term, and which is intended to allow the construction of a knowledge
base.

These memory banks will not necessarily be reset when the current task
is over but might also be reused as well.
"""

import torch
from torch import nn
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Union, Type, List, Any
from src.main.model import registry, virtual_layers

class RecurrentMemoryAttention(nn.Module, ABC):
    """
    This may be thought of as an attention mechanism
    that is usually going to be indirect, using
    kernel attention or something else.

    It also would be planned to store memories over the
    very long term, vs something short term like what
    is needed for the immediate task.

    It is a banked layer. That means that it actually
    contains, basically, a bunch of parallel layers that
    can be run next to each other.
    """

    def __init__(self,
                 d_model: int,
                 num_memories: int,
                 num_banks: int,
                 dropout: float,
                 ):
        super().__init__()

        self.d_model = d_model
        self.num_memories = num_memories
        self.dropout_rate = dropout
        self.num_banks = num_banks

    @abstractmethod
    def forward(self,
                tensor: torch.Tensor,
                bank_selection: banks.SelectionSpec,
                state: Any
                ) -> Tuple[torch.Tensor, Any]:
        """
        Heavily inspired by the source code from fast transformers, a set of builders
        with adjustable parameters to help me build my layers.
        """


recurrent_long_term_memory_registry = registry.TorchLayerRegistry[RecurrentMemoryAttention]("LongTermAttention",
                                                                                   RecurrentMemoryAttention)
