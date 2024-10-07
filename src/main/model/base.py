from typing import Union, List, Tuple, Dict
from abc import ABC, abstractmethod

import torch
from torch import nn

# Types
TensorTree = Union[
    torch.Tensor,  # Base case: Tensor
    List['TensorTree'],  # Recursion for lists
    Tuple['TensorTree', ...],  # Recursion for tuples
    Dict[str, 'TensorTree']  # Recursion for dictionaries
]


# Core layers
class StatefulCore(nn.Module, ABC):
    """
    Any class which is going to involve managing
    state in this project is basically going to
    need to implement this.
    """
    @abstractmethod
    def setup_state(self, tensor: torch.Tensor)->TensorTree:
        """
        Sets up state based on the provided tensor of embeddings. Note that
        if you do not use state, just return an empty dict.

        :param tensor: The tensor of embeddings
        :return: Whatever state we need. Can be none.
        """
    @abstractmethod
    def forward(self, tensor: torch.Tensor, states: TensorTree)->Tuple[torch.Tensor, TensorTree]:
        """
        Performs the forward pass. Tensor is a tensor of embeddings, while states is any
        state information that needs to be tracked.
        :param tensor:
        :param states:
        :return:
        """
        pass

