import math
from typing import Optional, Tuple, List, Callable, Any, Dict

import torch
from dataclasses import dataclass
from torch import nn
from torch.nn import functional as F
from torch.autograd.function import Function, FunctionCtx
from src.main.arcAGI2024.base import TensorTree, DeviceDtypeWatch, SavableState, DropoutLogits, parallel_pytree_map


# Define the memory state
class FastMemoryState(SavableState):
    """
    The memory state. Contains within it
    the linear kernel attention matrix
    and normalizer.

    This contains the internal state and state tracking
    mechanisms to allow the memory process to proceed.

    It is sometimes updated in place, and sometimes
    a new variation is created. The difference is
    that during the forward pass we reuse the container
    to allow passing of recomputed memories during the
    backwards pass in checkpoints.

    ----- fields ----

    Linear attention features:

    memories:
    - The memories tensors.
    - Shape: (..., num_memories, d_memory)
    - Where the model actually stores persistent information.
    write_probability_mass:
        - Keeps an eye on the probability mass distribution for
          each memory state
    """

    def __init__(self,
                 memories: torch.Tensor,
                 write_probability_mass: torch.Tensor,
                 ):

        self.memories = memories
        self.write_probability_mass = write_probability_mass
    def get_statistics(self) -> Dict[str, torch.Tensor]:
        """
        Returns relevant memory access metrics
        """
        statistics = {}
        statistics["write_probability_mass"] = self.write_probability_mass
        return statistics

    def get(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.memories, self.write_probability_mass

    # Setting up saving and loading allows parallel pytree map
    # to work its way down to the tensor leafs.
    def save_state(self) -> Tuple[TensorTree, Optional[Any]]:
        return (self.memories, self.write_probability_mass), None
    @classmethod
    def load_state(cls, pytree: TensorTree, bypass: Any) -> 'SavableState':
        return cls(*pytree)
##
# State creation mechanism
##

class CreateState(nn.Module):
    """
    Creates the default attention state when requested
    """

    @property
    def device(self) -> torch.device:
        return self.__metainfo.device

    @property
    def dtype(self) -> torch.dtype:
        return self.__metainfo.dtype

    def __init__(self,
                 d_memory: int,
                 num_memories: int,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None,
                 ):
        super().__init__()
        self.d_memory = d_memory
        self.num_memories = num_memories
        self.__metainfo = DeviceDtypeWatch(device=device, dtype=dtype)

    def forward(self, batch_shape: torch.Size) -> FastMemoryState:
        """
        Sets up the state.
        :param batch_shape: The batch shape that is correlated with the memories
        :return: The setup memory state.
        """
        # Setup the memories as blank tensors
        memories =  torch.zeros([*batch_shape, self.num_memories, self.d_memory],
                             dtype=self.dtype, device=self.device, requires_grad=True)

        # Setup the statistics containers
        write_probability_mass = torch.zeros([*batch_shape, self.num_memories],
                                             dtype=self.dtype, device=self.device, requires_grad=True)

        return FastMemoryState(memories, write_probability_mass)


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


class ReadMemory(nn.Module):
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
                 linear_kernel_activation: Callable[[torch.Tensor], torch.Tensor],
                 dtype: torch.dtype,
                 device: torch.device
                 ):
        """
        :param d_model: The size of the core arcAGI2024 dimensions
        :param d_memory: The storage available in each memory slot.
        :param num_read_heads: The number of heads to read with at once
        :param linear_kernel_activation: The linear kernel activation
        :param dtype: The dtype
        :param device: The device
        """
        super().__init__()

        self.d_model = d_model
        self.d_memory = d_memory
        self.num_read_heads = num_read_heads
        self.linear_kernel_activation = linear_kernel_activation
        self.__metainfo = DeviceDtypeWatch(device=device, dtype=dtype)

        # Define linear attention mechanism.
        self.linear_attn = LinearAttention(linear_kernel_activation)
        self.query_projector = nn.Linear(d_model, d_memory * num_read_heads, dtype=dtype, device=device)
        self.merge_head_projector = nn.Linear(d_memory * num_read_heads, d_model, dtype=dtype, device=device)

    def forward(self,
                query: torch.Tensor,
                addresses: torch.Tensor,
                memory: FastMemoryState
                ) -> torch.Tensor:
        """
        :param query:
            - The query to read with. Shape (..., d_model)
        :param addresses: The memory addresses. Shape ( num_memory, d_memory).
        :param memory: The memory state. Contains memory tensor shaped (..., num_memories, d_memory)
        :return: The memory read. Shape (..., d_model)
        """

        # Unpack the memory

        memory, _ = memory.get() # (..., num_memories, d_memory)

        # Create the read heads. These will be used
        # to attend to the memory using a linear attention
        # mechanism

        query = self.query_projector(query)
        query = query.unflatten(dim=-1,sizes=[self.num_read_heads, self.d_memory]) # (..., num_heads, d_memory)

        # Perform the read process on each head.
        mem_key = memory + addresses # (..., num_memories, d_memory)
        attn_results = self.linear_attn(query, mem_key, memory) # (..., num_heads, d_memory)

        # Collapse the heads, and return the result
        return self.merge_head_projector(attn_results.flatten(-2, -1))


class WriteMemory(nn.Module):
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

    @property
    def device(self) -> torch.device:
        return self.__metainfo.device

    @property
    def dtype(self) -> torch.dtype:
        return self.__metainfo.dtype

    def __init__(self,
                 d_model: int,
                 d_memory: int,
                 num_write_heads: int,
                 num_memories: int,
                 dropout_rate: float,
                 max_write_factor: float,
                 linear_kernel_activation: Callable[[torch.Tensor], torch.Tensor],
                 dtype: torch.dtype,
                 device: torch.device
                 ):
        """
        :param d_model: The model dim
        :param d_address: The address dim
        :param d_memory: the memory dim
        :param num_write_heads: num write heads
        :param num_memories: num memores
        :param dropout_rate: write selection logit dropout
        :param max_write_factor: Maximum rate we can write to a memory with
        - Note: While max is 1.0, for numeric stability you should keep it lower like 0.7
        :param linear_kernel_activation: The kernel activation
        :param dtype: The dtype
        :param device: The device.
        """

        super().__init__()
        self.d_model = d_model
        self.d_memory = d_memory
        self.num_write_heads = num_write_heads
        self.linear_kernel_activation = linear_kernel_activation
        self.max_write_factor = max_write_factor
        self.__metainfo = DeviceDtypeWatch(device=device, dtype=dtype)

        # Create the linear attn mechanism.

        self.linear_attn = LinearAttention(linear_kernel_activation)

        # Create the addresses projection mechanisms. Also, create
        # the attention addresses. Note the value projector is
        # twice as wide as you might expect. This provides fodder for both the update
        # and the write gate logits

        self.key_head_projector = nn.Linear(d_model, d_memory * num_write_heads, dtype=dtype, device=device)
        self.value_head_projector = nn.Linear(d_model, 2*d_memory * num_write_heads, dtype=dtype, device=device)

        # Create the interpolation rate logits.
        def initialize_based_on_half_lives(shape: torch.Size,
                                           low: float,
                                           high: float
                                           ):
            ##
            # Basically, we uniformly initialize
            # to lie within a certain half life band.
            #
            # We solved 0.5 = sigmoid(A)^t to do it.
            #
            # This will govern how many ticks it
            # takes to decay a certain amount.
            ##

            half_lives = torch.zeros(shape, device=device, dtype=dtype)
            half_lives.uniform_(low, high)

            x = (0.5) ** (1 / half_lives)
            logits = -torch.log(1 / x - 1)
            logits = nn.Parameter(logits, requires_grad=True)
            return logits

        # Create the probability projectors and interpolation logits
        self.interpolation_logits = initialize_based_on_half_lives([num_memories, d_memory],
                                                                   1,
                                                                   30)
        self.write_logits = nn.Linear(d_memory, 1, dtype=dtype, device=device)
        self.dropout_logits = DropoutLogits(dropout_rate)

    @staticmethod
    def broadcasted_where(batch_mask: torch.Tensor,
                          masked_case: torch.Tensor,
                          update_case: torch.Tensor
                          )-> torch.Tensor:
        """
        Basically, a where statement where the batch mask can broadcast from the left.
        :param batch_mask: The batch mask. Shape (...). True indicates masked and should not be updated.
        :param masked_case: What to do when we want to mask. Shape (..., ...more)
        :param update_case: What to do when we do not want to mask. Shape (..., ...more)
        :return: Something of shape (..., ...more)
        """
        while batch_mask.dim() < masked_case.dim():
            batch_mask = batch_mask.unsqueeze(-1)
        return torch.where(batch_mask, masked_case, update_case)

    def compute_common(self,
                       key: torch.Tensor,
                       values: torch.Tensor,
                       addresses: torch.Tensor,
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the matrix and normalizer update, along with the write
        factor for a particular write step. These are returned. How they
        are utilized depends on if we are operating in a forward or backwards
        mode.

        This can then be used to either run an update in forward mode, or
        reverse an update in backward mode.

        :param key: A tensor of keys, shaped (..., d_model)
        :param values: A tensor of values, shaped (..., d_model)
        :param addresses: The memory addresses. Shape (num_memory, d_memory).
        :return:
            - The updated memories. (..., num_memories, d_memory)
            - The write factor. (..., num_memories, d_memory)
        """
        # Two things must be computed here.
        #
        # These will be the memory update, and the
        # write probability for each update option.


        # Place write heads on the key, value features
        key = self.key_head_projector(key)
        key = key.unflatten(dim=-1, sizes=[self.num_write_heads, self.d_memory])  # (..., num_heads, d_memory)

        values = self.value_head_projector(values)
        values = values.unflatten(dim=-1, sizes=[self.num_write_heads, 2*self.d_memory])  # ( ..., num_heads, 2*d_memory)

        # Transfer values onto addresses to produce memory updates.
        #
        # Also create the write logits at the same time.

        results = self.linear_attn(addresses, key, values) # (..., num_memories, 2*d_memory)
        update = results[..., :self.d_memory]
        write_logits = results[..., self.d_memory:]

        # Create the write factor.
        #
        # The write probability governs how strongly we want to write,
        # while the interpolation factor governs how fast we CAN write.
        #
        # Memory is maintained as a weighted average over the writes.
        #
        # This will later be used for interpolation. The write probability
        # ends up determining how much of the interpolation factor to apply.

        write_probability = torch.sigmoid(write_logits)  # (..., num_memories, d_memories)
        interpolation_factor = torch.sigmoid(self.interpolation_logits)  # (num_memories, d_memories)
        write_factor = write_probability * interpolation_factor # (..., num_memories, d_memories)

        # In order to prevent numeric explosion, we must limit the maximum available write factor
        # Otherwise, the reverse pass can divide by zero

        write_factor = self.max_write_factor*write_factor

        # Return common
        return update, write_factor

    def reverse_memory(self,
                       update: Tuple[torch.Tensor, torch.Tensor],
                       write_factor: torch.Tensor,
                       batch_mask: torch.Tensor,
                       memory: FastMemoryState,
                       ) -> FastMemoryState:
        """
        Figures out the previous memory state from the
        current and the various update factors.
        :param update: The matrx, normalizer update pair
        :param write_factor: The write factor
        :param batch_mask: Shape (...). Indicates whether memory can be updated. True means No.
        :param memory: The current memory state
        :return: The last memory state
        """
        # Unpack the memory
        next_memory, next_cum_prob = memory.get()

        # Run the inverse of the original update factor.

        memory = (next_memory - update*write_factor)/ (1-write_factor)
        cum_prob = next_cum_prob - write_factor.mean(dim=-1)
        # Mask out anything that was not able to be updated

        memory = self.broadcasted_where(batch_mask, next_memory, memory)
        cum_prob = self.broadcasted_where(batch_mask, next_cum_prob, cum_prob)

        # Return the original memory state
        return FastMemoryState(memory, cum_prob)

    def advance_memory(self,
                       update: torch.Tensor,
                       write_factor: torch.Tensor,
                       batch_mask: torch.Tensor,
                       memory: FastMemoryState,
                       ) -> FastMemoryState:
        """
        Commits the computed update into the long term memory.
        Commits using interpolation.
        :param update: The update to integrate.
        :param write_factor: The write factor to go with it
        :param batch_mask: Shape (...). Indicates whether memory can be updated. True means No.
        :param memory: The current memory
        :return: The updated memory
        """
        # Unpack the memory
        last_memory, last_cum_prob= memory.get()

        # Get the key common information
        memory = last_memory*(1-write_factor) + update*write_factor
        cum_prob = last_cum_prob + write_factor.mean(dim=-1)

        # Mask out anything that was not able to be updated

        memory = self.broadcasted_where(batch_mask, last_memory, memory)
        cum_prob = self.broadcasted_where(batch_mask, last_cum_prob, cum_prob)

        # Return the new memory
        return FastMemoryState(memory, cum_prob)

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
                     ) -> FastMemoryState:
        """
        Creates and returns the blank memory state
        :param batch_shape: the batch shape to match
        :return: The concrete memory state.
        """
        return self.state_creator(batch_shape)

    def reverse(self,
                tensor: torch.Tensor,
                batch_mask: torch.Tensor,
                next_memory: FastMemoryState,
                ) -> Tuple[torch.Tensor, FastMemoryState]:
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
                memory: FastMemoryState,
                ) -> Tuple[torch.Tensor, FastMemoryState]:
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

