from typing import Optional, Tuple, List, Callable, Any, Dict

import torch
from dataclasses import dataclass
from torch import nn
from ..base import (TensorTree,
                    DeviceDtypeWatch,
                    PytreeState,
                    DropoutLogits,
                    parallel_pytree_map,
                    )
from .base import (AbstractMemoryUnit,
                   MemoryState,
                   AbstractMemoryConfig,
                   register_concrete_implementation,
                   AbstractReadMemory,
                   AbstractWriteMemory,
                   AbstractCreateState)


##
# Config requirements
##
@dataclass
class BankMemoryConfig(AbstractMemoryConfig):
    """
    Specifies the memory configuration for
    a bank memory state. This will include
    things like num memories

    Detail on what each config entry does follows

    **Attn memories config**

    The attention memories are a vector of num_memories x d_memory
    blocks where:

    d_memory: The width of each memory unit
    num_memories: The number of memories
    num_read_heads: Number of latent vectors to create out of the input to transfer out of the memory.
                    We then bind the memories onto it
    num_write_heads: Number of latent vectors to create out of the input to transfer into the memory.
                     We then bind the memories onto it.
    write_dropout_factor: A specialized dropout piece. It drops out attempts to write to some
                          memory location. Default is 0.1
    linear_kernel_activation_fn: The layer to use to activate the linear kernel. Default is nn.ReLU

    ** Abstract config ***
    These generally have pretty good defaults, but can be modified if needed.
    The write factor used to commit updates into memories has a lot
    of math associated with it. They do the following. You can modify
    them by passing in an optional kwarg.

    max_interpolation_factor: The maximum probabilty that can be written in a single step.
                              This is needed in order to prevent division by zero. Set to
                              0.999 as default, but lower might help with numeric stability

    The following two control how the write interpolation rates are initialized. Those factors
    are initialized uniformly between these, and can then be trained. These are basically
    decay factors between 0 and 1, that control how fast the running interpolation decays away
    when the model chooses to write to the memory in every step.

    min_write_half_life_init: Controls the minimum half life that the write interpolation rates
                              can be set on initialization. Must be >= 0.
    max_write_half_life_init: Controls the maximum half life that the write interpolation rates
                              can be set on initialization. Must be > min_write_half_life_init.
    """

    @property
    def interpolation_factor_shapes(self) -> torch.Size:
        return torch.Size([self.num_memories, self.d_memory])

    d_memory: int
    num_memories: int
    num_read_heads: int
    num_write_heads: int
    write_dropout_factor: float
    linear_kernel_activation_fn: nn.Module = nn.ReLU()
    max_interpolation_factor: float = 0.999,
    min_write_half_life_init: float = 0.1,
    max_write_half_life_init: float = 1000,

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
        self.num_memories = config.num_memories
        self.d_memory = config.d_memory
        self.addresses = nn.Parameter(torch.randn([config.num_memories, config.d_memory],
                                                  dtype=dtype, device=device),

                                      requires_grad=True)

    def forward(self, batch_shape: List[int]) -> MemoryState:
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
        memories = torch.zeros(batch_shape + [self.num_memories, self.d_memory],
                               dtype=self.dtype, device=self.device)
        memories.requires_grad_(True)

        interpolation_state["memories"] = memories

        # Setup the persistent state. We need two of them.

        persistent_state["cum_write_factors"] = torch.zeros(batch_shape + [self.num_memories],
                                                            dtype=self.dtype, device=self.device,
                                                            )
        persistent_state["cum_write_factors"].requires_grad_(True)

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

    def __init__(self, config: BankMemoryConfig):
        super().__init__()
        self.activation = config.linear_kernel_activation_fn

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

        self.num_read_heads = config.num_memories
        self.d_memory = config.d_memory
        self.linear_attn = LinearAttention(config)
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

        memory_bank = memory.memory_tensors["memories"]
        addresses = memory.metric_tensors["addresses"]

        # Create the read heads. These will be used
        # to attend to the memory using a linear attention
        # mechanism

        query = self.query_projector(query)
        query = query.unflatten(dim=-1,
                                sizes=[self.num_read_heads, self.d_memory]
                                )  # (..., num_heads, d_memory)

        # Perform the read process on each head.
        mem_key = memory_bank + addresses  # (..., num_memories, d_memory)
        attn_results = self.linear_attn(query, mem_key, memory_bank)  # (..., num_heads, d_memory)

        # Collapse the heads, and return the result
        return self.merge_head_projector(attn_results.flatten(-2, -1))


class WriteMemory(AbstractWriteMemory):
    """
    Performs a very efficient, reversible memory
    writing process.

    As per the contract, we implement the compute
    common method, allowing the super class to
    implement the forward and reverse mechanism.


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

        super().__init__(dtype, device, config)
        self.d_model = d_model
        self.num_write_heads = config.num_write_heads
        self.d_memory = config.d_memory

        # Create the linear attn mechanism.

        self.linear_attn = LinearAttention(config)

        # Create the addresses projection mechanisms. Also, create
        # the attention addresses. Note the value projector is
        # twice as wide as you might expect. This provides fodder for both the update
        # and the write gate logits

        self.key_head_projector = nn.Linear(d_model, config.d_memory * config.num_write_heads, dtype=dtype,
                                            device=device)
        self.value_head_projector = nn.Linear(d_model, 2 * config.d_memory * config.num_write_heads, dtype=dtype,
                                              device=device)
        self.dropout_logits = DropoutLogits(config.write_dropout_factor)

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
        values = values.unflatten(dim=-1,
                                  sizes=[self.num_write_heads, 2 * self.d_memory])  # ( ..., num_heads, 2*d_memory)

        # Transfer values onto addresses to produce memory updates.
        #
        # Also create the write logits at the same time.
        addresses = persistent_state["addresses"]
        results = self.linear_attn(addresses, key, values)  # (..., num_memories, 2*d_memory)
        update = results[..., :self.d_memory]

        # We drop out some percentage of the write logits
        # forcing exploration during training. Then we
        # activate producing write probabilities. Activation
        # is sigmoid.

        write_logits = results[..., self.d_memory:]  # (..., num_memories, d_memory)
        write_logits = self.dropout_logits(write_logits)
        write_probability = torch.sigmoid(write_logits)

        # Output
        output = {"memories" : update}

        # Return common
        return output, write_probability


class AttnBankMemory(AbstractMemoryUnit):
    """
    An attention based memory bank
    unit.

    Implements the needed logic in order to
    create an instance of the attn memory
    bank from a config, in a way that is
    compatible with the factories.
    """
    def __init__(self,
                 d_model: int,
                 dtype: torch.dtype,
                 device: torch.device,
                 config: BankMemoryConfig
                 ):
        state_creator = CreateState(device, dtype, config)
        state_reader = ReadMemory(d_model, dtype, device, config)
        state_writer = WriteMemory(d_model, dtype, device, config)
        super().__init__(state_creator, state_reader, state_writer)


register_concrete_implementation(BankMemoryConfig, AttnBankMemory)

