import math
from typing import Optional, Tuple, List, Callable, Any, Dict

import torch
from dataclasses import dataclass
from torch import nn
from torch.nn import functional as F

from src.main.model.base import TensorTree, DeviceDtypeWatch
from src.main.model.deep_memories.abstract import DeepMemoryUnit, deep_memory_registry, AbstractMemoryState
from src.main.model.virtual_layers import (DropoutLogits, VirtualLinear, VirtualAdvancedLinear,
                                           VirtualMakeHeads, VirtualMergeHeads, AbstractBankSelector,
                                           virtual_state_select, virtual_state_scatter, VirtualParameter,
                                           SelectionSpec)


# Define the memory state
class MemoryState(AbstractMemoryState):
    """
    The memory state. Contains within it
    the linear kernel attention matrix
    and normalizer.

    This contains the internal state and state tracking
    mechanisms to allow the memory process to proceed.


    ----- fields ----

    Linear attention features:

    matrix:
        - Part of the linear attention process. See papers for details
        - Stores memories as heads, of shape (..., num_memories, d_address, d_memory)
    normalizer:
        - Part of the linear attention process. See papers for details
        - Stores memories as heads, of shape (..., num_memories, d_address)

    Metrics:

    write_probability_mass:
        - Total probability mass associated with the write actions
        - Shape (..., num_memories, d_address)
    erase_probability_mass: Total probability mass associated with the erase actions
        - Total erasure and decay probability that has been executed
        - Shape (..., num_memories, d_address

    """
    def __init__(self,
                 matrix: torch.Tensor,
                 normalizer: torch.Tensor,
                 write_probability_mass: torch.Tensor,
                 erase_probability_mass: torch.Tensor,
                 ):
        self.matrix = matrix
        self.normalizer = normalizer
        self.write_probability_mass = write_probability_mass
        self.erase_probability_mass = erase_probability_mass
    def get_statistics(self)->Dict[str, torch.Tensor]:
        """
        Returns relevant memory access metrics
        """
        statistics = {}
        statistics["write_probability_mass"] = self.write_probability_mass
        statistics["erase_probability_mass"] = self.erase_probability_mass
        return statistics

    # Save and load contracts
    def save_state(self) -> Tuple[TensorTree, None]:
        """
        Saves the state into a pytree
        :return:
        """
        save_state = (
            self.matrix,
            self.normalizer,
            self.write_probability_mass,
            self.erase_probability_mass,
        )

        return save_state, None
    @classmethod
    def load_state(cls, pytree: TensorTree, bypass: None) -> 'MemoryState':
        """
        Loads the class from the pytree
        """
        return cls(*pytree)

    # Update and get contracts
    def update_(self,
                matrix: torch.Tensor,
                normalizer: torch.Tensor,
                write_probabilities: torch.tensor,
                erase_probabilities: torch.tensor,
                ):

        self.matrix = matrix
        self.normalizer = normalizer
        self.write_probability_mass = self.write_probability_mass + write_probabilities
        self.erase_probability_mass = self.erase_probability_mass + erase_probabilities

    def get(self) ->Tuple[torch.Tensor, torch.Tensor]:
        return self.matrix, self.normalizer



##
# State creation mechanism
##

class CreateState(nn.Module):
    """
    Creates the default attention state when requested
    """
    @property
    def device(self)->torch.device:
        return self.__metainfo.device

    @property
    def dtype(self)->torch.dtype:
        return self.__metainfo.dtype

    def __init__(self,
                 d_address: int,
                 d_memory: int,
                 num_memories: int,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None,
                 ):
        super().__init__()
        self.d_address = d_address
        self.d_memory = d_memory
        self.num_memories = num_memories
        self.__metainfo = DeviceDtypeWatch(device=device, dtype=dtype)


    def forward(self, batch_shape: torch.Size) -> MemoryState:
        """
        Sets up the state.
        :param batch_shape: The batch shape that is correlated with the memories
        :return: The setup memory state.
        """
        # Setup the matrix, normalizer, and default memory logits
        matrix = torch.zeros([*batch_shape, self.num_memories, self.d_address, self.d_memory],
                             dtype=self.dtype, device=self.device)
        normalizer = torch.zeros([*batch_shape, self.num_memories, self.d_address],
                             dtype=self.dtype, device=self.device)

        # Setup the statistics containers
        write_probability_mass = torch.zeros([*batch_shape, self.num_memories, self.d_address],
                                             dtype=self.dtype, device=self.device)
        erase_probability_mass = torch.zeros([*batch_shape, self.num_memories, self.d_address],
                                             dtype=self.dtype, device=self.device)


        return MemoryState(matrix, normalizer, write_probability_mass, erase_probability_mass)

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
                        )->torch.Tensor:
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
        denominator = torch.matmul(query, normalizer.unsqueeze(-1)) + 1e-9
        return numerator / denominator

    def make_kernel(self,
                    key: torch.Tensor,
                    value: torch.Tensor,
                    )->Tuple[torch.Tensor, torch.Tensor]:
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
    def device(self)->torch.device:
        return self.__metainfo.device

    @property
    def dtype(self)->torch.dtype:
        return self.__metainfo.dtype
    def __init__(self,
                 d_model: int,
                 d_address: int,
                 d_memory: int,
                 num_read_heads: int,
                 linear_kernel_activation: Callable[[torch.Tensor], torch.Tensor],
                 dtype: torch.dtype,
                 device: torch.device
                 ):
        """
        :param d_model: The size of the core model dimensions
        :param d_address: The size of the address logits to assign to each memory slot
        :param d_memory: The storage available in each memory slot.
        :param num_read_heads: The number of heads to read with at once
        :param linear_kernel_activation: The linear kernel activation
        :param dtype: The dtype
        :param device: The device
        """
        super().__init__()

        self.d_model = d_model
        self.d_address = d_address
        self.d_memory = d_memory
        self.num_read_heads = num_read_heads
        self.linear_kernel_activation = linear_kernel_activation
        self.__metainfo = DeviceDtypeWatch(device=device, dtype=dtype)

        # Define linear attention mechanism.
        self.linear_attn = LinearAttention(linear_kernel_activation)

        # Define access and read queries.
        self.access_query_projector = nn.Linear(d_model, d_address*num_read_heads, dtype=dtype, device=device)
        self.read_query_projector = nn.Linear(d_model, d_address*num_read_heads, dtype=dtype, device=device)

        # Define the head merging projector
        self.merge_head_projector = nn.Linear(d_memory*num_read_heads, d_model, dtype=dtype, device=device)
    def forward(self,
                query: torch.Tensor,
                addresses: torch.Tensor,
                memory: MemoryState
                )->torch.Tensor:
        """
        :param query:
            - The query to read with. Shape (..., d_model)
        :param addresses: A specialized varient of the normalizer, with addresses integrated.
            - Shape (..., num_memories, d_addresses)
        :param memory: The memory state. Contains
        - matrix: shape (..., num_memories, d_addresses, d_memory)
        - normalizer: shape (..., num_memories, d_addresses)
        :return: The memory read. Shape (..., d_model)
        """

        # Unpack the memory

        matrix, normalizer = memory.get()


        # Create the heads. The objective
        # from this point forward is to make the
        # memories interact sanely with each head.
        #
        # access query will be: (..., num_heads, d_address)
        # read query will be: (..., num_heads, d_address)

        access_query = self.access_query_projector(query)
        access_query = access_query.unflatten(dim=-1, sizes=[self.num_read_heads, self.d_address])

        read_query = self.read_query_projector(query)
        read_query = read_query.unflatten(dim=-1, sizes=[self.num_read_heads, self.d_address])

        # Flatten the matrix, and set aside it's shape. This makes
        # it much easier to run linear attention

        matrix = matrix.flatten(-2, -1) # (..., num_memories, d_address*d_memory)

        # Run attention against the matrix and the normalizer. By doing so,
        # combine them into a single kernel, which we can attend with.
        matrix = self.linear_attn(access_query, addresses, matrix) # (..., num_heads, d_address*d_memory)
        normalizer = self.linear_attn(access_query, addresses, normalizer) # (..., num_heads, d_address)

        # Unflatten the matrix
        # (..., num_heads, d_address, d_memory)
        matrix = matrix.unflatten(dim=-1, sizes=[self.d_address, self.d_memory])

        # Run read against the kernel. We perform a linear attn process,
        #
        # (..., num_heads, d_memory)
        response = self.linear_attn.read_from_kernel(read_query.unsqueeze(-2), matrix, normalizer).squeeze(-2)

        # Merge the heads
        response = self.merge_head_projector(response.flatten(-2, -1))

        # Return
        return response

class WriteMemory(nn.Module):
    """
    Performs an in-place update of the memory.

    This is done by:

    1: Perform domain transfer. Linear attention transfers the write heads
       into the domain of the memories, replacing the head dim with memories.
       We end up with keys, values in the memory dim.
    2: Kernel updates. Kernel updates for each memory position are calculated. In theory,
       we could directly add these updates the existing memory state to produce a new memory
       state.

       However, we do something more advanced.
    3: Relevancy filtering and gatekeeping. Based on the key, we create write and erase probabilities.
       Based on some additional clever code, we place limits on how strong an update can be, and force
       a little bit of erasure to happen whenever writing. Together these allow the model to flexibly store
       information while not overflowing the memory.
    4: Interpolation. Using computed probabilities, a weighed combination of the original and update state are
       stored.


    """

    @property
    def device(self) -> torch.device:
        return self.__metainfo.device

    @property
    def dtype(self) -> torch.dtype:
        return self.__metainfo.dtype
    def __init__(self,
                 d_model: int,
                 d_address: int,
                 d_memory: int,
                 num_write_heads: int,
                 num_memories: int,
                 linear_kernel_activation: Callable[[torch.Tensor], torch.Tensor],
                 dtype: torch.dtype,
                 device: torch.device

                 ):
        super().__init__()
        self.d_model = d_model
        self.d_address = d_address
        self.d_memory = d_memory
        self.num_write_heads = num_write_heads
        self.linear_kernel_activation = linear_kernel_activation
        self.__metainfo = DeviceDtypeWatch(device=device, dtype=dtype)

        # Create the linear attn mechanism.

        self.linear_attn = LinearAttention(linear_kernel_activation)

        # Create the addresses projection mechanisms

        self.key_head_projector = nn.Linear(d_model, d_address*num_write_heads, dtype=dtype, device=device)
        self.value_head_projector = nn.Linear(d_model, d_memory*num_write_heads, dtype=dtype, device=device)

        # Create the interpolation logits.
        def initialize_decay_logits(shape: torch.Size,
                                    low: float,
                                    high: float) -> nn.Parameter:
            decay_factors = torch.zeros(shape, device=device, dtype=dtype)
            decay_factors.uniform_(low, high)
            decay_logits = torch.log(decay_factors / (1 - decay_factors))
            decay_logits = nn.Parameter(decay_logits)
            return decay_logits

        self.interpolation_logits = initialize_decay_logits([num_memories, d_address], 0.001, 0.2)

        # Create the probability projectors

        self.write_logits = nn.Linear(d_address, 1, dtype=dtype, device=device)
        self.erase_logits = nn.Linear(d_address, d_address, dtype=dtype, device=device)
    def create_raw_probabilities(self,
                                           key: torch.Tensor
                                           )->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Creates the raw probabilities that will be used by the model
        :param key: The moved key. Shape (..., num_memories, d_address)
        :return:
        - The write probability. Shape (..., num_memories, 1). This being zero means no change
        - The erase probabilities. Shape (..., num_memories, d_address). This being zero tries to avoid erasing
        - The address interpolation rates. Shape (..., num_memories, d_address). Updates cannot happen faster than this,
                                   and this will force a certain amount of decay when writing.
        """

        # Create the write probability, and the erase probabilities
        write_probability = torch.sigmoid(self.write_logits(key)) # (..., num_memories, 1)
        erase_probability = torch.sigmoid(self.erase_logits(key)) # (..., num_memories, d_address)
        interpolation_factor = torch.sigmoid(self.interpolation_logits) # (num_memories, d_address)

        # The write probability must sum up to something greater than or equal
        # to one. If you are not committing at least one unit of probability among
        # all memories, we normalize.
        #
        # This is designed to prevent dead gradients by ensuring each write has to end
        # up somewhere.

        cumulative_probability = write_probability.sum(dim=-2, keepdim=True) + 1e-6
        needs_normalization = cumulative_probability < 1.0
        write_probability = torch.where(needs_normalization,
                                        write_probability/cumulative_probability,
                                        write_probability)


        return write_probability, erase_probability, interpolation_factor



    def forward(self,
                key: torch.Tensor,
                values: torch.Tensor,
                addresses: torch.Tensor,
                memory: MemoryState,
                ):
        """
        :param key: A tensor of keys, shaped (..., d_model)
        :param values: A tensor of values, shaped (..., d_model)
        :param addresses: The address feature. Shape (..., num_memories, d_addresses)
        :param memory: The memory state. Contains
        - matrix: shape (..., num_memories, d_addresses, d_memory)
        - normalizer: shape (..., num_memories, d_addresses)
        NO RETURN. Update is in place.
        """
        # Unpack the memory
        matrix, normalizer = memory.get()

        # Place write heads on the key, value features
        key = self.key_head_projector(key)
        key = key.unflatten(dim=-1, sizes=[self.num_write_heads, self.d_address]) # (..., num_heads, d_address)

        values = self.value_head_projector(values)
        values = values.unflatten(dim=-1, sizes=[self.num_write_heads, self.d_memory]) # ( ..., num_heads, d_memory)

        # Transfer using linear attention. Bind to addresses
        value = self.linear_attn(addresses, key, values) # (..., num_memories, d_memory)
        key = self.linear_attn(addresses, key, key) # (..., num_memories, d_addresses)

        # Create kernel update. Unsqueeze to fit linear attn mechanism format.
        #
        # In theory, if we added these updates to the existing matrix and normalizer,
        # we could be done. But, that is not a sophisticated enough memory system.
        #
        # matrix_update: (..., num_memories, d_addresses, d_memory)
        # normalizer_update: (..., num_memories, d_addresses)
        matrix_update, normalizer_update = self.linear_attn.make_kernel(key.unsqueeze(-2), value.unsqueeze(-2))

        # Create the various probabilities.
        write_probabilities, erase_probability, interpolation_factor = self.create_raw_probabilities(key)

        # Create the erase factor. One part of the update intepolation
        # Observe that with minimal erase probability, the
        # max ensures that at 100% write, we cannot go below
        # a certain level of erasure. This is intentional,
        # and ensures the model will accumulate a weighted average.
        # (..., num_memories, d_address)

        erase_factor = write_probabilities*torch.max(erase_probability, interpolation_factor) #
        # Create the write factor. This is the other part of the
        # update interpolation. The write probabilities, times
        # the interpolation factor, determines how quickly or strongly
        # we can update
        # (..., num_memories, d_address)

        write_factor = write_probabilities*interpolation_factor

        # Perform the actual interpolation, producing the updated matrix
        # and normalizer

        normalizer = normalizer*(1-erase_factor) + normalizer_update*write_factor
        matrix = matrix*(1-erase_factor.unsqueeze(-1)) + matrix_update*write_factor.unsqueeze(-1)

        # Integrate the updates.
        memory.update_(matrix, normalizer, write_factor, erase_factor)

@deep_memory_registry.register("FastLinearMemory")
class FastLinearMemory(DeepMemoryUnit):
    """
    A linear transformer fast memory system. It is designed to contain
    a very large, fast to address, memory collection optimized
    for very long term memory purposes.

    Gates are used to control when and how updates happen. However,
    some degree of update is forced to happen somewhere.

    Internally, d_addresses gives the model things
    it can attempt to find using address dot product
    attention, while d_memory is what can be stored in
    that space. Generally, d_address should be much less that
    d_memory, and similar to d_model//num_heads.

    The memory itself is maintained by transformer attention
    Read and write heads are separate, and determine how
    many actions occur at a time.

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
                d_address: int,
                d_memory: int,
                num_read_heads: int,
                num_write_heads: int,
                num_memories: int,
                bank_size: int,
                linear_kernel_activation: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                dtype: Optional[torch.dtype] = None,
                device: Optional[torch.device] = None,
                ):
        """
        :param d_model: The model dimensionality
        :param d_address: The address dimensionality. Should generally be much smaller. Think d_head
        :param d_memory: The memory storage space. Can be quite a bit larger.
        :param num_read_heads: The number of read heads. 4-8 is fine.
        :param num_write_heads: The number of write heads. 4-8 is fine.
        :param num_memories: The number of memories under consideration. Should be large
        :param bank_size: The virtual layer bank size
        :param linear_kernel_activation: The activation kernel for linear attention. Defaults to elu
        :param dtype: defaults to float32
        :param device: defaults to cpu
        """

        # Defaults
        if linear_kernel_activation is None:
            linear_kernel_activation = F.elu


        super().__init__(bank_size, d_model)

        # Setup
        self.addresses = VirtualParameter.create(bank_size, [num_memories, d_address],
                                                 device=device, dtype=dtype, init = nn.init.uniform_
                                                 )
        self.state_creator = CreateState(d_address, d_memory, num_memories, dtype=dtype, device=device)
        self.memory_reader = ReadMemory(d_model, d_address, d_memory, num_read_heads, linear_kernel_activation,
                                        device=device, dtype=dtype)
        self.memory_writer = WriteMemory(d_model, d_address, d_memory, num_write_heads, num_memories,
                                         linear_kernel_activation, dtype=dtype, device=device)
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
    def forward(self,
                tensor: torch.Tensor,
                selection: SelectionSpec,
                memories: MemoryState,
                ) -> torch.Tensor:
        """
        Implementation for the deep memory unit. Based on the tensor, and the
        memstate, produces a new tensor and a new memstate.
        :param tensor: The tensor to use to access and update the mem state. Shape (..., d_model)
        :param selection: The linear kernel selection spec.
        :param memory: The memory state. Contains
        - matrix: shape (..., num_memories, d_addresses, d_memory)
        - normalizer: shape (..., num_memories, d_addresses)
        :return:
        - The response tensor. Shape (..., d_model)
        - Memory is updated indirectly.
        """

        # Create the addresses.
        #
        # The adresses are used to isolate what pieces of memory to
        # care about, under the assumption that the normalizer is a sum of
        # attn activity which can represent the situation, and mixing in
        # per position parameters will help things along.
        #
        # This is also the only virtual layer on the class, and it
        # makes sense to exist here since different layers will
        # need to be remembering different things

        addresses = memories.normalizer + self.addresses(selection)

        # Perform the memory read, and the memory write
        read = self.memory_reader(tensor, addresses, memories)
        self.memory_writer(tensor, tensor, addresses, memories)

        # Return the read value
        return read