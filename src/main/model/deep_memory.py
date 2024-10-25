"""
The deep memory unit is responsible for extremely
long term storage of memory information and lookup
"""


import torch
from torch import nn
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Any, TypeVar, Generic
from src.main.model import registry
from src.main.model.virtual_layers import (VirtualLayer, SelectionSpec,
                                           DropoutLogits,
                                           VirtualMakeHeads, VirtualMergeHeads)
from fast_transformers import builders


memory_state = Tuple[torch.Tensor, torch.Tensor]

class MemoryManager(VirtualLayer):
    """
    Manages the creation and access of memories using
    a linear attention kernel. Converting these into reads
    and writes is not this layer's responsibility. It only
    figures out what could be read from or written to.

    Internally, the memory actually consists of an extremely
    large number of heads.
    """
    def __init__(self,
                 d_model: int,
                 d_address: int,
                 num_heads: int,
                 num_memories: int,
                 num_banks: int,
                 dropout: float
                 ):
        super().__init__(num_banks)

        self.d_model = d_model
        self.d_address = d_address
        self.num_memories = num_memories
        self.num_banks = num_banks
        self.dropout = DropoutLogits(dropout)

        # Create head mechanisms of relevance
        self.make_query_heads = VirtualMakeHeads(d_model, d_address, num_heads, num_banks)
        self.make_key_heads = VirtualMakeHeads(d_model, d_address, num_heads, num_banks)
        self.make_value_heads = VirtualMakeHeads(d_model, d_address, num_heads, num_banks)

        # Create memory addresses and recurrent linear self attention action.

        self.memory_addresses = nn.Parameter(torch.empty([num_memories, d_address])).uniform_(-0.1, 0.1)
        self.rla = builders.RecurrentAttentionBuilder.from_kwargs(query_dimensions=d_address)

        # And the decay logits. Decay rates are defined per memory element
        # and the model can decide how much to make short vs long term.
        # We initialize the logits based on what decay factor they will
        # activate to.

        decay_factors = torch.zeros([d_model])
        decay_factors.uniform_(0.0001, 0.2)
        decay_logits = -torch.log((1 / decay_factors) - 1)
        self.decay_logits = nn.Parameter(decay_logits, requires_grad=True)

    def create_memory(self, shape: torch.Tensor) -> memory_state:
        """
        Creates a new memory container.
        :param shape: The shape to repeat.
        :return: The set up memory container
        """
        matrix = torch.zeros([self.num_memories, self.d_address, self.d_address])
        normalizer = torch.zeros([self.num_memories, self.d_address])

        for _ in shape:
            matrix = matrix.unsqueeze(-1)
            normalizer = normalizer.unsqueeze(-1)

        matrix = matrix.broadcast_to([*shape, *self.default_matrix.shape])
        normalizer = normalizer.broadcast_to([*shape, *self.default_normalizer.shape])
        return matrix, normalizer

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                selection: SelectionSpec,
                state: Optional[memory_state] = None
                )->Tuple[torch.Tensor, memory_state]:
        """
        :param query: Shape (batch_size, d_model)
        :param key: Shape (batch_size, d_model)
        :param value: Shape (batch_size, d_model)
        :param state: Memory state. Consists of
            - matrix: (batch_size, memories, d_memory, d_model)
            - normalizer: (batch_size, memories, d_memory, d_model)
        :return:
        """

        # Standardize to setup memory
        if state is None:
            state = self.create_memory(query.shape[:-1])

        # Create heads
        query = self.make_query_heads(query, selection)  # (batch_size, heads, d_address)
        key = self.make_key_heads(key, selection)  # (batch_size, heads, d_address)
        value = self.make_value_heads(value, selection) # (batch_size, heads, d_address)

        # Run attention.
        response, revised_state = self.rla(query, key, value, state)

        # Break apart revised state. Get update state, indicating the difference
        # from original to revised state.

        update_state = []
        for original, revised in zip(state, revised_state):
            update_state.append(revised-original)

        # 
        response # (batch_size, head, d_address)


