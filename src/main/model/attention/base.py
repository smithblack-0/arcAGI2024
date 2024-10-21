"""
The base recurrent attention class, the interface contract,
and the builder registry all in one place. The outside world
learns how to interface from here.
"""

import torch
from torch import nn
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Union, Type, List, Any
from ..registry import TorchLayerRegistry
class RecurrentAttention(nn.Module, ABC):
    """
    The base contract for the recurrent linear attention
    mechanism. Includes the three slots for the standard attention
    parameters, then an additional state slot as well.
    """

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


# Create the self attention registry.
recurrent_self_attention_registry = TorchLayerRegistry[RecurrentAttention](
    d_model=int,
    d_key=int,
    d_value=int,
    num_heads=int,
    dropout=float
)
