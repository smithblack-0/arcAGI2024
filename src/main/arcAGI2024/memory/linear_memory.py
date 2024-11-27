"""
A memory varient, uses a form of linear attention directly in order
to directly track, gather, and return information on the attention
block.
"""
import torch
from torch import nn
from typing import Callable, Tuple, Dict, List
from .base import (AbstractMemoryConfig,
                   GradientTimeLossConfig,
                   MemRegularizationLossConfig,
                   AbstractWriteMemory,
                   AbstractCreateState,
                   AbstractReadMemory,
                   AbstractMemoryUnit,
                   MemoryState)


class LinearMemoryConfig(AbstractMemoryConfig):
    """
    Configuration object for the linear
    attention memory system.

    Basically, we keep around a linear attention
    kernel and can commit writes to the kernel bits
    as desired.

    Each head operates independently of the others,
    and the best way to add memory capacity is likely
    to increase the number of heads vs increase d_memory,
    as this has a lesser impact on the amount of memory
    that must be retained
    """
    num_heads: int
    d_address: int
    d_memory: int
    gradient_loss: GradientTimeLossConfig
    mem_regularization_loss: MemRegularizationLossConfig
    min_write_half_life_init: float = 1.0
    max_write_half_life_init: float = 100.0
    erase_epsilon_factor: float = 0.0001
    linear_activation_kernel: Callable[[torch.Tensor], torch.Tensor] = torch.relu
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

class CreateState(AbstractCreateState):
    """
    Creates a functional linear attention
    memory state for the model. This will
    setup all the cumulative tensors with the
    correct dtype, device, shape, etc
    """
    def __init__(self,
                 d_model: int,
                 device: torch.device,
                 dtype: torch.dtype,
                 config: LinearMemoryConfig
                 ):
        super().__init__(dtype, device)
        self.d_model = d_model
        self.num_heads = config.num_heads
        self.d_address = config.d_address
        self.d_memory = config.d_memory

    def initialize_with_shape(self, shape: List[int])->torch.Tensor:
        return torch.zeros(shape, dtype=self.dtype, device=self.device)
    def forward(self, batch_shape: List[int])->MemoryState:
        """
        Creates the state. This includes initializing all
        the various important metric tensors to their
        defaults.
        :param batch_shape:
        :return:
        """

        # Setup the metrics tensor.
        #
        # The write factor operates at the head level.
        # The erase factor operates one detail at the elements level.
        # The effective write mass must be at the deepest of the above two
        # The timestep only depends on the batch level.
        # And the average timestep distance depends on write and erase, so same depth.

        metrics = {
            "cum_write_probability" : self.initialize_with_shape(batch_shape + [self.num_heads]),
            "cum_erase_probability" : self.initialize_with_shape(batch_shape + [self.num_heads, self.d_address]),
            "effective_write_mass" : self.initialize_with_shape(batch_shape + [self.num_heads, self.d_address]),
            "timestep" : self.initialize_with_shape(batch_shape),
            "average_timestep_distance" : self.initialize_with_shape(batch_shape + [self.num_heads, self.d_address]),
        }

        # Setup the normalizer and matrix. These are the actual memory features
        memories = {
            "matrix" : self.initialize_with_shape(batch_shape + [self.num_heads, self.d_address, self.d_memory]),
            "normalizer" : self.initialize_with_shape(batch_shape + [self.num_heads, self.d_address]),
        }

        # Return the memory state. We do not end up needing the persistent tensors at all
        # for this architecture.
        return MemoryState(metrics, memories, {})

class ReadMemory(AbstractReadMemory):
    """
    Performs the read memory action. This basically
    just consists of performing linear attention
    using the kernel out of the provided
    input query
    """
    def __init__(self,
                 d_model: int,
                 device: torch.device,
                 dtype: torch.dtype,
                 config: LinearMemoryConfig
                 ):
        super().__init__(dtype, device)
        self.d_model = d_model
        self.num_heads = config.num_heads
        self.d_address = config.d_address
        self.d_memory = config.d_memory

        # Make the linear attention mechanism

        self.linear_attention = LinearAttention(config.linear_activation_kernel)

        # Designate the head projectors
        self.make_headed_query_projection = nn.Linear(d_model, self.d_address*self.num_heads, bias=False)
        self.merge_heads_projector = nn.Linear(self.num_heads*self.d_memory, d_model, bias=False)
    def read_memory(self,
                    query: torch.Tensor,
                    memories: Dict[str, torch.Tensor],
                    persistent: Dict[str, torch.Tensor]
                    ) -> torch.Tensor:
        """
        Performs the memory read operation. This is done by
        taking the query, turning it into a headed query, then
        performing linear attention against the relevant
        memory kernels.

        :param query: The query to read with. Shape (..., d_model). Recurrent, so no items dim
        :param memories: The memory tensors. Same as in create.
        :param persistent: Unused in this architecture
        :return: The result of reading. Shape (..., d_model)
        """

        # Create the headed question
        query = self.make_headed_query_projection(query) # (..., num_heads*d_address)
        query = query.unflatten(dim=-1, sizes=[self.num_heads, self.d_address]) # (..., num_heads, d_address)

        # Perform the attention mechanism.
        #        Response will be (..., num_heads, d_memory)
        response = self.linear_attention.read_from_kernel(query, memories["matrix"], memories["normalizer"])

        # Finish and return
        response = response.flatten(-2, -1)
        response = self.merge_heads_projector(response)
        return response



