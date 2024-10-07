import torch

from typing import List, Dict, Any
from torch import nn

def batched_index_select(input: torch.Tensor,
                         dim: int,
                         index: torch.Tensor) -> torch.Tensor:
    """
    A specialized function designed to handle batched
    index selects. I seem to be needing to do this a lot

    :param input: The input indices to select.
        - Shape (...1, ..., options, ...)
    :param dim: The dimension to select from
    :param index: Has matching dimensions to input from the front.
        - Shape( ...1, selections)
    :return: An index select, which has been batched.
    """
    # Internally, we will construct a special matrix
    # and perform matrix multiplication to move
    # selections into options.

    dim_size = input.size(dim)

    # Create matrix multiplication one hot
    transfer_matrix = torch.arange(dim_size, device=input.device, dtype=index.dtype) #(options)
    transfer_matrix = index.unsqueeze(-1) == transfer_matrix #(..., selections, options)

    # Reformat transfer matrix, and input.
    input = input.movedim(dim, -1) #(...1, ..., options)
    input = input.unsqueeze(-2) #(...1, ..., 1, options)