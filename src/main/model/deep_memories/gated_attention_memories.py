"""
Implementation file for the large gated attention memory
"""

import torch
from torch import nn
from src.main.model.deep_memories.abstract import DeepMemoryUnit
from src.main.model.virtual_layers import SelectionSpec, VirtualLinear, VirtualState
from typing import Optional, Tuple


class GatedAttentionMemories(DeepMemoryUnit):
    """
    GatedAttentionMemories is an extension of the RecurrentMemoryAttention mechanism
    that employs a gated write operation and memory decay. This model allows selective
    writing to memories, where memories decay over time, and new information is added
    through a gate mechanism.

    This approach facilitates storing long-term information in a differentiable memory
    structure, where old information is slowly overwritten by new inputs, controlled by
    a decaying write gate. It also enables reading from memory using attention mechanisms.
    It is designed with the idea in mind of write rarely, read frequently.

    This is both recurrent and banked. In particular, the write gate is banked, ensuring
    that different virtual layers may write to long term memory in different ways.
    """

    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 num_memories: int,
                 num_banks: int,
                 dropout: float
                 ):
        """
        :param d_model: The dimensionality of the input features (embedding size).
        :param num_heads: Number of attention heads for memory reading.
        :param num_memories: The number of memories (memory slots) to allocate.
        :param num_banks: The number of layer banks available for virtual layers.
        :param dropout: The dropout rate applied during memory reading.
        """
        super().__init__(d_model, num_memories, num_banks, dropout)

        # Setup a default memory configuration
        self.memory_default = nn.Parameter(torch.zeros([num_memories, d_model]), requires_grad=True)
        nn.init.normal_(self.memory_default)



        # Setup the attention transfer systems. Particularly the write
        # projections. The write gate is banked.
        self.memory_write_transfer = nn.Linear(d_model, d_model * num_memories)
        self.memory_write_gate = VirtualLinear(d_model, num_memories, num_banks)

        # The memory read is not banked. I do not have a banked MHA ready yet.
        self.make_memory_query = VirtualLinear(d_model, d_model, num_banks)
        self.memory_read_transfer = nn.MultiheadAttention(d_model, num_heads, dropout, batch_first=True)

    def write_new_memories(self,
                           tensor: torch.Tensor,
                           state: torch.Tensor,
                           bank_selection: SelectionSpec
                           ) -> torch.Tensor:
        """
        Creates the new memories based on the provided tensor and handles the write gating mechanism.
        :param tensor: Input tensor to process. Shape: (..., d_model)
        :param state: The current state of memory. Shape: (..., num_memories, d_model)
        :param bank_selection: The bank selection object. Tells us what bank state we are in.
        :return: Updated state of memory. Shape: (..., num_memories, d_model)
        """

        # Create the write possibilities. This information may end up written
        # to memory, but only if the write gate cooperates. Then define the
        # write gate.

        write_info = self.memory_write_transfer(tensor).reshape(tensor.shape[:-1],
                                                                self.num_memories,
                                                                self.d_model)  # (..., num_memories, d_model)

        write_gate = self.memory_write_gate(write_info, bank_selection)
        write_gate = torch.sigmoid(write_gate).unsqueeze(-1)  # (..., num_memories, 1)

        # At least one units worth of probability WILL be written somewhere. If the model
        # tries to shut off all writing, we renormalize back to a total of one. This
        # will, hopefully, greatly improves the long term gradient flow by ensuring the model
        # cannot shut off writes that it needs to train with. T

        write_gate_needs_renormalization = write_gate.sum(dim=-2, keepdim=True)
        renormalized_write_gate = torch.linalg.vector_norm(write_gate, ord=1, dim=-2)
        write_gate = torch.where(write_gate_needs_renormalization, renormalized_write_gate, write_gate)

        # The final write probability is a combination of the decay rates
        # (how much memory CAN be updated) and the write gate (how much the model WANTS to update it).

        decay_rates = torch.sigmoid(self.write_decay_logits)
        write_probability = decay_rates * write_gate

        # Handle the writing and decay. Return results
        state = state * (1 - write_probability) + write_info * write_probability
        return state

    def read_from_memories(self,
                           tensor: torch.Tensor,
                           state: torch.Tensor,
                           bank: SelectionSpec) -> torch.Tensor:
        """
        Reads something important from memories, if there is any such thing.
        :param tensor: The tensor to use as a read query. (..., d_model)
        :param state: The memories. (..., num_memories, d_model)
        :param bank_selection: The bank selection object. Tells us what bank state we are in.
        :return: The result of reading from memory. (..., d_model)
        """
        # Read from memory using multi-head attention, where the tensor acts as the query
        # and the memory state serves as both the key and value.
        query = self.make_memory_query(tensor, bank)
        read = self.memory_read_transfer(query.unsqueeze(-2), state, state)
        return read.squeeze(-2)

    def forward(self,
                tensor: torch.Tensor,
                bank_selection: SelectionSpec,
                state: Optional[torch.Tensor] = None,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process the input tensor through the memory mechanism. This includes reading from the memory
        and updating the memory with the new tensor.

        :param tensor: The input tensor to process. Shape: (..., d_model)
        :param state: The current state of memory, or None to initialize a new memory.
                      Shape: (..., num_memories, d_model)
        :param bank_selection: The selected banks. Used for writing only. Invokes given virtual layer.
        :return:
            - The result from reading the memory. Shape: (..., d_model)
            - The updated memory state. Shape: (..., num_memories, d_model)
        """
        # Standardization. If the state is none, set it up
        if state is None:
            state = self.memory_default + torch.zeros_like(tensor).unsqueeze(-2)

        # Read, write

        memory_read = self.read_from_memories(tensor, state)
        state = self.write_new_memories(tensor, state, bank_selection)
        return memory_read, state
