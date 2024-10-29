import math
from typing import Optional, Tuple, List, Callable, Any

import torch
from dataclasses import dataclass
from torch import nn
from torch.nn import functional as F

from src.main.model.base import TensorTree
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
    and normalizer

    matrix: (...batch_shape, num_memories, num_heads, d_head, d_head)
    normalizer: (...batch_shape, num_memories, num_heads,  d_head)
    """
    def __init__(self,
                 matrix: torch.Tensor,
                 normalizer: torch.Tensor,
                 ):
        self.matrix = matrix
        self.normalizer = normalizer

    # Save and load contracts
    def save_state(self) -> Tuple[TensorTree, Optional[Any]]:
        return (self.matrix, self.normalizer), None
    @classmethod
    def load_state(cls, pytree: TensorTree, bypass: Any) -> 'MemoryState':
        return cls(*pytree)

    # Update and get contracts
    def update_(self, matrix: torch.Tensor, normalizer: torch.Tensor):
        self.matrix = matrix
        self.normalizer = normalizer

    def get(self) ->Tuple[torch.Tensor, torch.Tensor]:
        return self.matrix, self.normalizer



##
# State creation mechanism
##

class CreateState(nn.Module):
    """
    Creates the default attention state when requested
    """

    def __init__(self,
                 d_head: int,
                 d_model: int,
                 num_heads: int,
                 num_memories
                 ):
        super().__init__()
        self.d_head = d_head
        self.d_model = d_model
        self.num_memories = num_memories
        self.num_heads = num_heads

    def forward(self, batch_shape: torch.Size) -> MemoryState:
        """
        Sets up the state.
        :param batch_shape: The batch shape that is corrolated with the memories
        :return: The setup memory state.
        """
        matrix = torch.zeros([*batch_shape, self.num_memories, self.num_heads, self.d_head, self.d_head])
        normalizer = torch.zeros([*batch_shape, self.num_memories, self.num_heads])
        return MemoryState(matrix, normalizer)

class SelectExperts(AbstractBankSelector):
    """
    Performs the expert selection process.

    We implement a varient of abstract bank selector here,
    since it saves us the trouble of coding a significant
    portion of the logic.

    Internally, what is going on is we are creating a selection
    spec that indicates a small subset of the memory we wish
    to work on.
    """
    def __init__(self,
                 d_head: int,
                 num_memories: int,
                 num_heads: int,
                 mem_top_k: int,
                 mem_random: int,
                 control_dropout: float,
                 dtype: torch.dtype,
                 device: torch.device,
                 ):
        super().__init__(top_k=mem_top_k,
                         rand=mem_random,
                         control_dropout=control_dropout)

        # Create address parameter
        self.addresses = nn.Parameter(torch.rand([num_memories, num_heads, d_head],
                                                 dtype=dtype, device=device
                                                 ))

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                )->SelectionSpec:
        """
        Selects the experts by looking at the query,
        the addresses, then considering the attention
        logits.

        :param query: (...,num_heads, d_heads)
        :param key:  (..., num_memories, num_heads, d_head)
        :return: A headed selection spec. (..., num_head, selected_memories) inside.
        """

        # Ready for attention scoring
        query = query.unsqueeze(-2) #(..., num_heads, 1, d_head)
        addresses = key + self.addresses
        addresses = addresses.movedim(-2, -3) # (..., num_head, num_memories, d_head)

        # Create logits.

        logits = torch.matmul(query, addresses.swapdims(-1, -2)) # (..., num_head, 1, num_memories)
        logits = logits / math.sqrt(self.d_head)
        logits = logits.squeeze(-2) # (..., num_head, num_memories)

        # Select key memories from them.
        return self.select_logits(logits)

class MemorySubsets(nn.Module):
    """
    A class for managing the sparse experts system.

    It is responsible, more or less, for ensuring the
    rest of the code does not have to worry about things
    being sparse. It has two important helper methods
    attached, to give you a subset of the memory to
    perform primary computation with, and later on
    to integrate those results into the broader memory.
    """
    def __init__(self,
                 d_head: int,
                 num_heads: int,
                 num_memories: int,
                 mem_top_k: int,
                 mem_random: int,
                 control_dropout: float,
                 device: torch.device,
                 dtype: torch.dtype
                 ):
        super().__init__()

        self.d_head = d_head
        self.num_memories = num_memories
        self.mem_top_k = mem_top_k
        self.mem_random = mem_random
        self.control_dropout = control_dropout

        self.device = device
        self.dtype = dtype


        # Initialize the decay logits across the entire memory.
        # Create the decay rates to associate with each key, and
        # with each value within a key. The inverse sigmoid function
        # is used to ensure we start with logits activating to factors
        # between 0 and 0.2

        def initialize_decay_logits(tensor: torch.Tensor,
                                    low: float,
                                    high: float) -> nn.Parameter:
            decay_factors = tensor
            decay_factors.uniform_(low, high)
            decay_logits = torch.log(decay_factors / (1 - decay_factors))
            decay_logits = nn.Parameter(decay_logits)
            return decay_logits

        initialize_decay_logits = lambda  x : initialize_decay_logits(x, 0.001, 0.2)

        self.main_decay_logits = VirtualParameter.create(num_memories, [num_heads, d_head],
                                                device=device, dtype=dtype, init=initialize_decay_logits)
        self.auxiliary_decay_logits = VirtualParameter.create(num_memories, [1, 1, d_head],
                                                device=device, dtype=dtype, init=initialize_decay_logits)
    def select_experts(self,
                       memory: MemoryState,
                       selection_spec: SelectionSpec
                       )->Tuple[torch.Tensor, torch.Tensor]:
        """
        Selects a subset of the memory to be computed with.
        :param query: The query to use when decoding the memory. Shape (..., num_heads, d_head)
        :param memory: The memory unit. Contains:
           - matrix: (...batch_shape, num_memories, num_heads, d_head, d_head)
           - normalizer: (...batch_shape, num_memories, num_heads, d_head)
        :param selection_spec: A set of probabilities and indexes.
            - Shapes (..., num_heads, selected_memories)
            - Presumably from select experts
        :return:
            - Tuple:
                - matrix: (..., num_selected, num_heads, d_head, d_head)
                - normalizer: (..., num_selected, num_heads, d_head)
        """

        # Unpack the memory
        matrix, normalizer = memory.get()

        # Figure out the selection
        selection = selection_spec

        # Get a subset of the matrix and normalizer arrays to return.
        # To do this, we have to rearrange the matrox and normalizer
        # to have the head dimension before the memories, due to the
        # way the virtual select function works
        #
        # The virtual select function is just a special kind of sparse
        # select.
        #
        # We then fix it again

        matrix = matrix.movedim(-4, -3) # ( ..., num_heads, num_mem, d_head, d_head)
        matrix = virtual_state_select(matrix, selection, dim=-3, superposition=False)
            # Now (... , num_heads, selected, d_head, d_head)

        normalizer = normalizer.movedim(-2, -3) # (..., num_heads, num_memories, d_head)
        normalizer = virtual_state_select(normalizer, selection, dim=-2, superposition=False)
            # Now (..., num_heads, selected, d_head)

        matrix = matrix.movedim(-3, -4) # (..., selected_mems, num_heads, d_head, d_head)
        normalizer = normalizer.movedim(-3, -4) # (..., selected_mems, num_heads d_head)

        # Now we can return the selected features for further attention processes
        return matrix, normalizer

    def update_memories(self,
                        selected_memories: Tuple[torch.Tensor, torch.Tensor],
                        update: Tuple[torch.Tensor, torch.Tensor],
                        memory: MemoryState,
                        selection_spec: SelectionSpec
                        ):
        """
        Updates the long term memories based on the processed matrix
        update and normalizer update. This will include both applying
        the decay factors, and scattering back into the memories

        As a reminder, updates are done in place!
        :param selected_memories: The memories that were previously selected.
        - matrix: (..., num_selected, num_heads, d_head, d_head)
        - normalizer: (..., num_selected, num_heads, d_head)
        :param update: Tuple. Consists of
        - matrix: (..., num_selected, num_heads, d_head, d_head)
        - normalizer: (..., num_selected, num_heads, d_head)
        :param memory: the memory state unit
        :param selection_spec: The selection spec.
        """

        # Unpack the update and the memory
        matrix_update, normalizer_update = update
        selected_matrix, selected_normalizer = selected_memories

        # Figure out the matrix and normalizer decay
        # factors. This will consist of getting instances
        # of the virtual parameters, activating them
        # in a reasonable manner, and arranging
        # the dimensions for the challenge

        normalizer_decay_logits = self.main_decay_logits(selection_spec, sum=False) # (num_heads, d_head, selected)
        auxiliary_decay_logits = self.auxiliary_decay_logits(selection_spec, sum=False)
        matrix_decay_logits = normalizer_decay_logits.unsqueeze(-1) + auxiliary_decay_logits
            # matrix decay logits : (num_heads, d_head, selected, d_head)

        normalizer_decay_factors = torch.sigmoid(normalizer_decay_logits)
        matrix_decay_factors = torch.sigmoid(matrix_decay_logits)

        normalizer_decay_factors = normalizer_decay_factors.movedim(-1, -3)
        # normalizer_decay_factors : (selected, num_heads, d_head, d_head)
        matrix_decay_factors = matrix_decay_factors.movedim(-2, -4) # (selected, num_heads, d_head)

        # Mix the updates in by interpolation based on the strength of the decay factors
        selected_matrix = selected_matrix * (1 - matrix_decay_factors) + matrix_update * matrix_decay_factors
        selected_normalizer = (selected_normalizer * (1 - normalizer_decay_factors) +
                               normalizer_update * normalizer_decay_factors) # (..., selected, num_heads, d_head)

        # Now integrate it into the broader memory
        original_matrix, original_normalizer = memory.get()
        updated_matrix = virtual_state_scatter(original_matrix, selected_matrix, selection_spec, -4)
        updated_normalizer = virtual_state_scatter(original_normalizer, selected_normalizer, selection_spec, -3)

        # and store
        memory.update_(updated_matrix, updated_normalizer)

class LinearExpertBanks(DeepMemoryUnit):
    """
    A sparse interaction linear memory system.
    A sparse subset of the memories z
    """
