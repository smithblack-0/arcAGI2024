"""
The base recurrent deep_memories class, the interface contract,
and the builder registry all in one place. The outside world
learns how to interface from here.
"""

import torch
from torch import nn
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Union, Type, List, Any
from ..registry import TorchLayerRegistry


class RecurrentSelfAttention(nn.Module, ABC):
    """
    The base contract for the recurrent linear deep_memories
    mechanism. Includes the three slots for the standard deep_memories
    parameters, then an additional state slot as well.
    """

    def __init__(self,
                 d_model: int,
                 d_key: int,
                 d_value: int,
                 d_head: int,
                 num_heads: int,
                 dropout: float,
                 ):
        super().__init__()

        self.d_model = d_model
        self.d_key = d_key
        self.d_value = d_value
        self.d_head = d_head
        self.num_heads = num_heads
        self.dropout_rate = dropout

    @abstractmethod
    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                state: Any) -> Tuple[torch.Tensor, Any]:
        """
        Heavily inspired by the source code from fast transformers, a set of builders
        with adjustable parameters to help me build my layers.
        """


class MakeHeads(nn.Module):
    """
    A class designed to make heads.
    """


# create some of the registries

recurrent_short_term_attention_registry = TorchLayerRegistry[RecurrentSelfAttention]("ShortTermAttention",
                                                                                     RecurrentSelfAttention)
