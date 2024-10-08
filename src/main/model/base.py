from typing import Union, List, Tuple, Dict, Any
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
    def forward(self,
                embedding: torch.Tensor,
                states: TensorTree,
                *parameters: Any)->Tuple[torch.Tensor, TensorTree]:
        """
        Performs the forward pass. Tensor is a tensor of embeddings, while states is any
        state information that needs to be tracked.
        :param tensor: The embedding we are processing
        :param states: The states, if any, associated with the embedding
        :param parameters: Any additional parameters we might need
        :return:
        """
        pass

class RecurrentLinearAttention(nn.Module):
    """

    """
