"""
A memory varient, uses a form of linear attention directly in order
to directly track, gather, and return information on the attention
block.
"""
import torch
from torch import nn
from typing import Callable, Tuple, Dict



class LinearAttention(nn.Module):
    """
    Performs linear attention, without heads, using a query,
    a key, and a value. While the query and key must be the same
    shape, the value can differ beyond certain primary dimensions.

    This greatly eases how to handle the matrix and normalizer, as
    we will later see.
    """

    def __init__(self,
                 activation: Callable[[torch.Tensor], torch.Tensor],
                 ):
        super().__init__()
        self.activation = activation

    def read_from_kernel(self,
                         query: torch.Tensor,
                         matrix: torch.Tensor,
                         normalizer: torch.Tensor
                         ) -> torch.Tensor:
        """
        Performs linear attention, using an existing attention kernel.
        Returns the attention result. d_model arbitrary
        :param query: Something of shape (...,queries,  d_address)
        :param matrix: Something of shape (..., d_address, d_memory)
        :param normalizer: Something of shape (..., d_address)
        :return: Something of shape (..., queries, d_address)
        """
        query = self.activation(query)
        numerator = torch.matmul(query, matrix)
        denominator = torch.matmul(query, normalizer.unsqueeze(-1)) + 1e-5
        return numerator / denominator

    def make_kernel(self,
                    key: torch.Tensor,
                    value: torch.Tensor,
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Creates a linear attention kernel from the set of
        keys and values.
        :param key: Shape (..., items, d_address)
        :param value: Shape (..., items, d_memories)
        :return:
            -  matrix: Something of shape (..., d_address, d_memories)
            -  normalizer: Something of shape (..., d_address)
        """
        # Activate the key
        key = self.activation(key)

        # Rearrange in preparation for matmul and sum
        key = key.movedim(-2, -1)

        # Execute matmul. Execute sum
        matrix = torch.matmul(key, value)
        normalizer = torch.sum(key, dim=-1)

        # Return
        return matrix, normalizer

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor
                ):
        """
        Performs a cross linear attention process.
        :param query: Something of shape (...,queries,  d_address)
        :param key: Shape (..., items, d_address)
        :param value: Shape (..., items, d_memory)
        :return: The attention result. Shape (..., queries, d_address)
        """
        matrix, normalizer = self.make_kernel(key, value)
        return self.read_from_kernel(query, matrix, normalizer)
