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
from typing import Tuple, Optional, Any, TypeVar, Generic
from src.main.model import registry
from src.main.model.virtual_layers import (VirtualLayer, SelectionSpec, VirtualMakeHeads, VirtualMergeHeads)


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


# Setup virtual memory manager, along with
# registry.
MemoryType = TypeVar('MemoryType')


class VirtualMemoryManager(VirtualLayer, Generic[MemoryType]):
    """
    Allows the access and update of a virtual
    memory bank. Generally, this would be intended
    to be used through methods other than forward.
    An abstract implementation that can be implemented
    to better sync with concrete virtual memory implementations.

    Generally, the memory type is going to contain tensors that
    have shapes that are some variation of (..., banks, d_model)
    """

    def __init__(self,
                 d_memory: int,
                 bank_size: int,
                 memory_size: int,
                 ):
        super().__init__(bank_size)
        self.d_memory = d_memory
        self.memory_size = memory_size

    @abstractmethod
    def create_memory(self, shape: torch.Tensor) -> MemoryType:
        """
        Creates the virtual memory, if it is needed.
        :param shape: The batch shape to build the memories around
        :return: The memory type. Implementation dependent
        """

    @abstractmethod
    def express_memory(self,
                       query: torch.Tensor,
                       memories: MemoryType,
                       selection: SelectionSpec
                       ) -> torch.Tensor:
        """
        Express an interpretable memory-based response to the query, like in
        attention, based on the recorded memories.
        :param query: The query to respond to. Shape (...batch, heads, d_model)
        :param memories: The memories that have been expressed previously. Implementation dependent
        :param selection: The selection specification.
        :return: The resulting tensor. Shape (..., heads, d_model)
        """

    @abstractmethod
    def store_memory(self,
                     query: torch.Tensor,
                     statement: torch.Tensor,
                     memories: MemoryType,
                     selection: SelectionSpec
                     ) -> MemoryType:
        """
        Uses the query to decide how to store the statement into the memory.
        Highly implementation dependent.
        :param query: The query to use for storage. Shape (..., heads, d_model)
        :param statement: The statement to store into memory. Shape (..., heads, d_model)
        :param memories: The memory type. Implementation dependent
        :param selection: The selection specification.
        :return: The new memories.
        """


virtual_memory_manager = registry.TorchLayerRegistry[VirtualMemoryManager]("VirtualMemoryManager",
                                                                           VirtualMemoryManager)


# Setup memory reader. Setup registry, then immediately register
# the used implementation. Use registry indirection.
class MemoryReader(nn.Module):
    """
    A main memory reading layer, capable of performing banked
    setup, access, and collapse.

    Memory reads consist of using headed banks to construct queries
    that are used to respond to information, and the heads are
    then later recombined. Must be provided a memory manager
    instance to function.
    """

    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 bank_size: int,
                 vmm: VirtualMemoryManager
                 ):
        super().__init__()
        d_memory = vmm.d_memory

        # Store
        self.d_model = d_model
        self.d_memory = d_memory
        self.num_heads = num_heads
        self.vmm = vmm

        # Setup head projectors and removers. These
        # portions are banked

        self.make_heads = VirtualMakeHeads(d_model, d_memory, num_heads, bank_size)
        self.merge_heads = VirtualMergeHeads(d_model, d_memory, num_heads, bank_size)

    def forward(self,
                query: torch.Tensor,
                memories: Any,
                selection: SelectionSpec
                ) -> torch.Tensor:
        """
        Performs a read of the long term memory, returning the responses,
        based on the banked state.
        :param query: A query to inquire after. Shape (..., d_model)
        :param memories: The memories. Implementation dependent
        :param selection: The selection. A selection spec. Selects banks
        :return: The tensor after reading.
        """
        headed_queries = self.make_heads(query, selection)
        headed_responses = self.vmm.express_memory(headed_queries, memories, selection)
        responses = self.merge_heads(headed_responses, selection)
        return responses


class MemoryWriter(nn.Module):
    """
    Exactly what it says. Attempts to write to the memory.
    In practice, it produces heads and access entries of the right
    shape.
    """

    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 bank_size: int,
                 vmm: VirtualMemoryManager
                 ):
        super().__init__()
        d_memory = vmm.d_memory

        # Store
        self.d_model = d_model
        self.d_memory = d_memory
        self.num_heads = num_heads
        self.vmm = vmm

        # Setup head projectors and removers. These
        # portions are banked

        self.make_query_heads = VirtualMakeHeads(d_model, d_memory, num_heads, bank_size)
        self.make_statement_heads = VirtualMakeHeads(d_model, d_memory, num_heads, bank_size)

    def forward(self,
                query: torch.Tensor,
                statement: torch.Tensor,
                memories: Any,
                selection: SelectionSpec
                ) -> Any:
        """
        Accesses the memory bank with intention to place statement into
        memories based on the responses from queries.
        :param query: The query to select with. Shape (..., d_model)
        :param statement: The statement to integrate. Shape (..., d_model)
        :param memories: The memories. implementation dependent
        :param selection: The selection. A selection spec. Selects banks to use.
        :return: The new memories
        """
        query = self.make_query_heads(query, selection)
        statement = self.make_statement_heads(statement, selection)
        return self.vmm.store_memory(query, statement, memories, selection)


class DeepMemoryUnit(nn.Module):
    """
    Puts everything together. A deep memory unit
    contains the create, read, and write action
    all in one central location.
    """

    def __init__(self,
                 vmm: VirtualMemoryManager,
                 d_model: int,
                 num_heads: int,
                 bank_size: int,
                 dropout: Optional[float] = None
                 ):
        super().__init__()

        # Store raw.
        self.num_heads = num_heads
        self.d_memory = vmm.d_memory
        self.d_model = d_model

        # Standardize dropout

        if dropout is None:
            dropout = 0.0

        # Create and store layers
        self.vmm = vmm
        self.reader = MemoryReader(d_model, num_heads, bank_size, vmm)
        self.writer = MemoryWriter(d_model, num_heads, bank_size, vmm)
        self.layernorm = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                query: torch.Tensor,
                selection: SelectionSpec,
                memories: Optional[Any] = None,
                ) -> Tuple[torch.Tensor, Any]:
        """
        Runs the deep memory process designed to store long
        term memories.

        :param query: The query to select with. Shape (... d_model)
        :param selection: The selection spec. Selects banks
        :param memories: The memories to get from. Implementation dependent
        :return:
            - The result of probing the memories
            - The resulting long term memories.
        """
        # Create memories if needed
        if memories is None:
            shape = [*query.shape[:-1], self.num_heads, self.d_memory]
            memories = self.vmm.create_memory(shape)

        # Read from the memories, and incorporate this into a response
        read = self.reader(query, memories, selection)
        response = self.layernorm(query + self.dropout(read))
        memories = self.writer(query, response, memories, selection)

        # Return the results.
        return response, memories


deep_memory_unit_registry = registry.TorchLayerRegistry[DeepMemoryUnit]("DeepMemoryUnit", DeepMemoryUnit,
                                                                        vmm=virtual_memory_manager,
                                                                        )
deep_memory_unit_registry.register_class("Default", DeepMemoryUnit)

###
# Some implementations
##

