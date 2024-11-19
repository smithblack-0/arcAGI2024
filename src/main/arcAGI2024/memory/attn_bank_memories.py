import math
from typing import Optional, Tuple, List, Callable, Any, Dict

import torch
from dataclasses import dataclass
from torch import nn
from torch.nn import functional as F
from torch.autograd.function import Function, FunctionCtx
from ..base import TensorTree, DeviceDtypeWatch, PytreeState, DropoutLogits, parallel_pytree_map
from .base import (AbstractMemoryUnit, MemoryState, AbstractMemoryConfig,
                   AbstractReadMemory, AbstractWriteMemory, AbstractCreateState)

##
# Config requirements
##

class BankMemoryConfig(AbstractMemoryConfig):
    """
    Specifies the memory configuration for
    a bank memory state. This will include
    things like num memories
    """
    d_memory: int
    num_memories: int
    num_read_heads: int
    num_write_heads: int
    linear_kernel_activation: Callable[[torch.Tensor], torch.Tensor] = torch.relu



##
# State creation mechanism
##

class CreateState(AbstractCreateState):
    """
    Creates the default memory state on
    demand when requested
    """

    def __init__(self, device: torch.device, dtype: torch.dtype, config: BankMemoryConfig):
        super().__init__(dtype, device)
        self.config = config
        self.addresses = nn.Parameter(torch.randn([config.num_memories, config.d_memory],
                                                   dtype=dtype, device=device),
                                      requires_grad=True)


    def forward(self, batch_shape: torch.Size) -> MemoryState:
        """
        Sets up the state.
        :param batch_shape: The batch shape that is correlated with the memories
        :return: The setup memory state.
        """
        # Setup the state dicts
        persistent_state: Dict[str, torch.Tensor] = {}
        interpolation_state: Dict[str, torch.Tensor] = {}

        # Setup the memories as blank tensors. This is the only
        # needed interpolation state.
        memories =  torch.zeros([*batch_shape, self.config.num_memories, self.config.d_memory],
                             dtype=self.dtype, device=self.device, requires_grad=True)
        interpolation_state["memories"] = memories

        # Setup the persistent state. We need two of them.

        persistent_state["cum_write_factors"] = torch.zeros([*batch_shape, self.num_memories],
                                                dtype=self.dtype, device=self.device, requires_grad=True)
        persistent_state["attn_addresses"] = self.addresses

        return MemoryState(persistent_state, interpolation_state)


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


class ReadMemory(AbstractReadMemory):
    """
    Reads the memory using the provided query.
    Memory is encoded in terms of two levels.
    First, there is the num_memories feature.
    Second, there are the matrix and normalizer we
    might actually be able to perform attention with.
    Both levels must be dealt with to perform a read.

    So, we basically use two attention steps. Step 1 is
    a linear attention step of the normalizer and matrix
    across the memories dimension, reducing the kernel
    down to something without a memory dimension. Step 2
    then reads the contents of that kernel.
    """
    def __init__(self,
                 d_model: int,
                 dtype: torch.dtype,
                 device: torch.device,
                 config: BankMemoryConfig,
                 ):
        """
        :param d_model: The size of the core arcAGI2024 dimensions
        :param dtype: The dtype
        :param device: The device
        :param config: The BankMemoryConfig. Contains everything else.
                       Look to the class for an explanation of the parameters.
        """
        super().__init__(dtype, device)

        self.config = config
        self.linear_attn = LinearAttention(config.linear_kernel_activation)
        self.query_projector = nn.Linear(d_model, config.d_memory * config.num_read_heads,
                                         dtype=dtype, device=device)
        self.merge_head_projector = nn.Linear(config.d_memory * config.num_read_heads, d_model,
                                              dtype=dtype, device=device)

    def forward(self,
                query: torch.Tensor,
                memory: MemoryState,
                ) -> torch.Tensor:
        """
        :param query: The query to read with. Shape (..., d_model)
        :param memory: The memory state. Contains memory tensor shaped (..., num_memories, d_memory)
        :return: The memory read. Shape (..., d_model)
        """

        # Unpack the memory

        memory_bank = memory.interpolation_state["memories"]
        addresses = memory.persistent_state["addresses"]

        # Create the read heads. These will be used
        # to attend to the memory using a linear attention
        # mechanism

        query = self.query_projector(query)
        query = query.unflatten(dim=-1,
                                sizes=[self.num_read_heads, self.d_memory]
                                ) # (..., num_heads, d_memory)

        # Perform the read process on each head.
        mem_key = memory_bank + addresses # (..., num_memories, d_memory)
        attn_results = self.linear_attn(query, mem_key, memory) # (..., num_heads, d_memory)

        # Collapse the heads, and return the result
        return self.merge_head_projector(attn_results.flatten(-2, -1))


class WriteMemory(AbstractWriteMemory):
    """
    Performs a very efficient, reversible memory
    writing process.

    Attention is used to transfer write heads
    into the memory domain, where they are integrated
    and utilized.

    A bank of address parameters are used to perform this
    transfer.

    During the commit process, a write probability is used
    to gauge how much of the available interpolation
    process to apply.

    """

    def __init__(self,
                 d_model: int,
                 dropout_rate: float,
                 dtype: torch.dtype,
                 device: torch.device,
                 config: BankMemoryConfig,
                 ):
        """
        :param d_model: The size of the core arcAGI2024 dimensions
        :param dtype: The dtype
        :param device: The device
        :param config: The BankMemoryConfig. Contains everything else.
                       Look to the class for an explanation of the parameters.
        """

        super().__init__(dtype, device, config)
        self.d_model = d_model
        self.config = config

        # Create the linear attn mechanism.

        self.linear_attn = LinearAttention(config.linear_kernel_activation)

        # Create the addresses projection mechanisms. Also, create
        # the attention addresses. Note the value projector is
        # twice as wide as you might expect. This provides fodder for both the update
        # and the write gate logits

        self.key_head_projector = nn.Linear(d_model, config.d_memory * config.num_write_heads, dtype=dtype, device=device)
        self.value_head_projector = nn.Linear(d_model, 2*config.d_memory * config.num_write_heads, dtype=dtype, device=device)
        self.dropout_logits = DropoutLogits(dropout_rate)

    def _compute_common(self,
                        query: torch.Tensor,
                        persistent_state: Dict[str, torch.Tensor],
                        ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        The implementation. We compute the updated state, consisting
        solely of the proposed new memories, alongside the write
        probability.

        :param query: The query tensor. Shape (..., d_model), presumably
        :param persistent_state: The persistent state of the memory.
        :return:
        - update: An implementation-specific update, consisting of a dict of
                  tensortrees corrolating with the interpolation memory content.
        - write_probability: A write probability that tells us how strongly to write to the memory slots.
            - Must cleanly multiply the interpolation factor shape.
        """
        # Two things must be computed here.
        #
        # These will be the memory update, and the
        # write probability for each update option.


        # Place write heads on the key, value features
        key = self.key_head_projector(query)
        key = key.unflatten(dim=-1, sizes=[self.num_write_heads, self.d_memory])  # (..., num_heads, d_memory)

        values = self.value_head_projector(query)
        values = values.unflatten(dim=-1, sizes=[self.num_write_heads, 2*self.d_memory])  # ( ..., num_heads, 2*d_memory)

        # Transfer values onto addresses to produce memory updates.
        #
        # Also create the write logits at the same time.
        addresses = persistent_state["addresses"]
        results = self.linear_attn(addresses, key, values) # (..., num_memories, 2*d_memory)
        update = results[..., :self.d_memory]
        write_logits = results[..., self.d_memory:]

        # We drop out some percentage of the write logits
        # forcing exploration during training. Then we
        # activate producing write probabilities. Activation
        # is sigmoid.

        write_logits = self.dropout_logits(write_logits)
        write_probability = torch.sigmoid(write_logits)

        # Return common
        return update, write_probability

class FastLinearMemory(nn.Module):
    """
    A linear transformer fast memory system. It is designed to contain
    a very large, fast to address, memory collection optimized
    for very long term memory purposes.

    The class is designed to allow a reversable training process,
    though not the most efficient one in the world. The prior memory
    state can be computed from the current memory state, and the inputs.
    This allows for trading computation time for memory, and allows
    handling really long sequences.

    A linear attention mechanism is used to ensure all of this
    happens very fast, and with few parameters.
    """

    @property
    def device(self) -> torch.device:
        return self.__metainfo.device

    @property
    def dtype(self) -> torch.dtype:
        return self.__metainfo.dtype

    def __init__(self,
                 d_model: int,
                 d_memory: int,
                 num_read_heads: int,
                 num_write_heads: int,
                 num_memories: int,
                 dropout_rate: int,
                 linear_kernel_activation: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                 max_write_factor: float = None,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None,
                 ):
        """
        :param d_model: The arcAGI2024 dimensionality
        :param d_memory: The memory storage space. Can be quite a bit larger.
        :param num_read_heads: The number of read heads. 4-8 is fine.
        :param num_write_heads: The number of write heads. 4-8 is fine.
        :param num_memories: The number of memories under consideration. Should be large
        :param linear_kernel_activation: The activation kernel for linear attention. Defaults to elu
        :param max_write_factor: The maximum probability, between 0 and 1, that can be committed at once
                                 to a memory position
                                 - It is required to be less than 1.0 for numeric reasons.
                                 - Default is 0.6
                                 - If numeric divergence is an issue, try lowering it a bit.
        :param dtype: defaults to float32
        :param device: defaults to cpu
        """

        # Defaults
        if linear_kernel_activation is None:
            linear_kernel_activation = F.softplus
        if max_write_factor is None:
            max_write_factor = 0.6

        super().__init__()

        # Define address creation
        self.addresses = nn.Parameter(torch.randn([num_memories, d_memory], dtype=dtype, device=device),
                                      requires_grad=True)

        # Create the various other control features
        self.state_creator = CreateState(d_memory, num_memories, dtype=dtype, device=device)
        self.memory_reader = ReadMemory(d_model, d_memory, num_read_heads, linear_kernel_activation,
                                        device=device, dtype=dtype)
        self.memory_writer = WriteMemory(d_model, d_memory, num_write_heads, num_memories,
                                         dropout_rate, max_write_factor, linear_kernel_activation,
                                         dtype=dtype, device=device)

        # Create the device metainfo watch.
        self.__metainfo = DeviceDtypeWatch(device=device, dtype=dtype)

    def create_state(self,
                     batch_shape: torch.Size
                     ) -> MemoryState:
        """
        Creates and returns the blank memory state
        :param batch_shape: the batch shape to match
        :return: The concrete memory state.
        """
        return self.state_creator(batch_shape)

    def reverse(self,
                tensor: torch.Tensor,
                batch_mask: torch.Tensor,
                next_memory: MemoryState,
                ) -> Tuple[Tuple[torch.Tensor, MemoryState], MemoryState]:
        """
        The reverse implementation. Training is actually
        intended to occur in this one. A memory state consisting
        of the next memory is provided, the original memory
        is calculated, then that is used to continue the
        computation.

        Some sophisticated work needs to be done later running
        backwards passes through the various final memories
        to keep the gradients alive. This must be handled
        manually. It is intended that these gradients will
        be fed in through the "next_memory" parameters.

        :param tensor: The original tensor input
        :param batch_mask: Shape (...). Indicates whether memory can be updated. True means No.
        :param next_memory: The next memory state
        :return:
        - The originally produced output, usable to continue the computation
        - The original memory state. Setup to accumulate gradients.
        """
        # Compute the write components.
        update, write_factor = self.memory_writer.compute_common(tensor, tensor, self.addresses)

        # Get the original memory state.
        #
        # Make it retain grads so we can get
        # our gradients off it later for the next
        # backwards pass.
        with torch.no_grad():
            original_memory = self.memory_writer.reverse_memory(update, write_factor, batch_mask, next_memory)

        def setup_grads(tensor: torch.Tensor)->torch.Tensor:
            tensor = nn.Parameter(tensor, requires_grad=True)
            tensor.retain_grad()
            return tensor

        original_memory = parallel_pytree_map(setup_grads, original_memory)

        # Manually complete the read
        next_memory = self.memory_writer.advance_memory(update, write_factor, batch_mask, original_memory)
        read = self.memory_reader(tensor, self.addresses, next_memory)
        return (read, next_memory), original_memory
    def forward(self,
                tensor: torch.Tensor,
                batch_mask: torch.Tensor,
                memory: MemoryState,
                ) -> Tuple[torch.Tensor, MemoryState]:
        """
        Forward pass for the fast linear memory unit.
        :param tensor: The tensor to use to access and update the mem state. Shape (..., d_model)
        :param batch_mask: Shape (...). Indicates whether memory can be updated. True means No.
        :param memory: The memory state. Contains
        - matrix: shape (..., num_memories, d_addresses, d_memory)
        - normalizer: shape (..., num_memories, d_addresses)
        - write_probability_mass
        :return:
        - The response tensor. Shape (..., d_model)
        - The new memory state
        """


        update, write_factor = self.memory_writer.compute_common(tensor, tensor, self.addresses)
        memory = self.memory_writer.advance_memory(update, write_factor, batch_mask, memory)
        read = self.memory_reader(tensor, self.addresses, memory)
        return read, memory

