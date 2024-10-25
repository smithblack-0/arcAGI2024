"""

"""
from typing import Optional, Tuple

import torch
from torch import nn

from src.main.model.long_term_memories.core import VirtualMemoryManager
from src.main.model.virtual_layers import DropoutLogits, VirtualLinear, SelectionSpec

memory_type = Tuple[torch.Tensor, torch.Tensor]
class LinearKernelMemoryBanks(VirtualMemoryManager[memory_type]):
    """
    A Linear kernel memory bank

    Memories are stored away in linear attention kernels, as in
    the fast transformers implementation. Accessing these memories
    performs an attention read, while updating them produces an
    attention update. Performing a memory update always also results
    in some quantity of decay happening
    """
    def dot_product_focus(self,
                          query: torch.Tensor,
                          key: torch.Tensor
                          ) -> torch.Tensor:
        """
        Performs only the focus portion of attention, getting the nxm matrix
        :param query: The query. Shape (..., query, d_memory)
        :param key: The key. Shape (..., key, d_memory)
        :return: Activated probabilities. Shape (..., query, key)
        """

        logits = torch.matmul(query, key.T)
        logits = self.dropout_logits(logits)
        if self.activation == "softmax":
            probabilities = torch.softmax(logits, dim=-1)
        else:
            probabilities = torch.sigmoid(logits)
        return probabilities

    @staticmethod
    def dot_product_attention(weights: torch.Tensor,
                              values: torch.Tensor,
                              )-> torch.Tensor:
        """
        Performs the rest of dot product attention. Will
        also handle weirdly shaped tensors by flattening and unflattening everything
        beyond attn dim.
        :param weights: The attention weights. Shape (..., query, key)
        :param values: The values. Shape (..., key, ...d_model)
        :return: The result of attention. Shape (..., query, ...d_model)
        """

        # Set aside shape. Flatten after key in values

        key_location = weights.dim() - 1 # From the front, the key will be located where the query is on weights
        restore_shape = values.shape[key_location+1:]
        values = values.flatten(key_location+1, -1) #(..., key, d_all)

        # Perform attention
        result = torch.matmul(weights, values)

        # Restore original shape. Return
        return result.unflatten(-1, restore_shape)
    def __init__(self,
                 d_memory: int,
                 num_memories: int,
                 bank_size: int,
                 activation: str,
                 dropout: Optional[float] = None,
                 ):
        assert activation in ["sigmoid", "softmax"]
        super().__init__(d_memory, num_memories, bank_size)

        # Create default memories
        self.default_matrix = nn.Parameter(torch.empty([num_memories, d_memory, d_memory]).uniform_(-0.1, 0.1))
        self.default_normalizer = nn.Parameter(torch.empty([num_memories, d_memory])).uniform_(-0.1, 0.1)

        # Create memory addresses, and other important features

        self.memory_addresses = nn.Parameter(torch.empty([d_memory, num_memories])).uniform_(-0.1, 0.1)

        self.dropout_logits = DropoutLogits(dropout)
        self.activation = activation

        # And the decay logits. Decay rates are defined per memory element
        # and the model can decide how much to make short vs long term.
        # We initialize the logits based on what decay factor they will
        # activate to.

        decay_factors = torch.zeros([d_memory])
        decay_factors.uniform_(0.0001, 0.2)
        decay_logits = -torch.log((1 / decay_factors) - 1)
        self.decay_logits = nn.Parameter(decay_logits, requires_grad=True)



    def express_memory(self,
                       query: torch.Tensor,
                       memories: memory_type,
                       selection: SelectionSpec
                       ) -> torch.Tensor:
        """

        :param query: The headed query. Shape (..., heads, d_memory
        :param memories: Tuple of
            - matrix: (...,num_memories, d_memory, d_memory)
            - normalizer: (...,num_memories, d_memory, )
        :param selection: Unused, but required for contract.
        :return: The expressed memory. Shape (..., heads, d_memory)
        """
        # Unpack
        matrix, normalizer = memories

        # Perform attention focusing situation. This narrows
        # us down to a single relevant collection.

        attn_weights = self.dot_product_focus(query, normalizer + self.memory_addresses)
        matrix = self.dot_product_attention(attn_weights, matrix)
        normalizer = self.dot_product_attention(attn_weights, normalizer)

        # Perform attention search

