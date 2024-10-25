"""
The linear expert memories mechanism is a sparse
long term memory mechanism designed to provide both deep
and efficient access by means of banked memory collections.
"""

import torch
from torch import nn
from core import RecurrentMemoryAttention, recurrent_long_term_memory_registry
from src.main.model.virtual_layers import (SelectionSpec, DropoutLogits, VirtualLinear,
                                           virtual_state_select, virtual_state_scatter)
from fast_transformers import builders
from typing import Optional, Tuple


class MakeHeads(nn.Module):
    """
    Exactly what it says on the tin. Makes attention heads.
    Notably, they can be more than just reshapes of the
    query.

    This layer is banked. This means how the heads are created
    may depend on the virtual layer configuration. The banks are
    however both created and consumed within the layer, resulting
    in a superposition.
    """

    def __init__(self,
                 d_model: int,
                 d_head: int,
                 num_heads: int,
                 num_banks: int
                 ):
        super().__init__()

        self.d_model = d_model
        self.d_head = d_head
        self.num_heads = num_heads
        self.num_banks = num_banks

        self.projection = VirtualLinear(d_model, d_head * num_heads, num_banks)

    def forward(self,
                tensor: torch.Tensor,
                bank_selection: SelectionSpec
                ) -> torch.Tensor:
        """
        Creates the heads on the tensor.
        :param tensor: The tensor to make heads on. Shape (..., d_model)
        :param bank_selection: The virtual layer bank selection.
        :return: The headed tensor. Shape (..., num_heads, d_head)
        """
        tensor = self.projection(tensor, bank_selection)  #(..., d_head*num_heads)
        tensor = tensor.reshape(tensor.shape[:-2], self.num_heads, self.d_head)
        return tensor


class MergeHeads(nn.Module):
    """
    Exactly what it says on the tin. Merges attention heads.

    This layer is banked. This means how the heads are merged
    may depend on the virtual layer configuration. The banks are
    however both created and consumed within the layer, resulting
    in a superposition.
    """

    def __init__(self,
                 d_model: int,
                 d_head: int,
                 num_heads: int,
                 num_banks: int
                 ):
        super().__init__()

        self.d_model = d_model
        self.d_head = d_head
        self.num_heads = num_heads
        self.num_banks = num_banks

        self.projection = VirtualLinear(d_head * num_heads, d_model, num_banks)

    def forward(self,
                tensor: torch.Tensor,
                bank_selection: SelectionSpec
                ) -> torch.Tensor:
        """
        Removes  the heads on the tensor. Gets back to something of shape d_model
        :param tensor: The tensor to merge heads on. Shape (..., num_heads, d_heads)
        :param bank_selection: The virtual layer bank selection.
        :return: The deheaded tensor. Shape (..., d_model)
        """
        tensor = self.projection(tensor, bank_selection)  #(..., d_head*num_heads)
        tensor = tensor.reshape(tensor.shape[:-2], self.num_heads, self.d_head)
        return tensor


class Memory:
    """
    A container containing the recurrent memories.
    Includes attention addresses, normalizer, and matrix.
    Used to update information on the fly.
    ---- fields ----

    addresses:
        - Specialized parameter used to identify what information will go where
        - Shape (num_memories, d_memory).
    normalizer:
        - The normalizer sum for linear kernel attention.
        - shape (..., num_memories, d_memory)
    matrix:
        - The linear kernel attention matrix.
        - Shape (..., num_memories, d_memory, d_model)
        - Contains the actual output features.
    memory_change_rates:
        - The rate at which change in memory can happen.
        - Shape (..., num_memories, d_memory).
        - Sigmoids.

    """

    def __init__(self,
                 addresses: torch.Tensor,
                 normalizer: torch.Tensor,
                 matrix: torch.Tensor,
                 memory_change_rates: torch.Tensor
                 ):
        self._addresses = addresses
        self._normalizer = normalizer
        self._matrix = matrix
        self._memory_change_rates = memory_change_rates

    def get_addresses(self) -> torch.Tensor:
        """
        Gets the memory addresses that are used for attention selection
        :return: The addresses. Shape (..., num_memories, d_memory)
        """
        return self._addresses + self._normalizer

    def get_state(self, selection: SelectionSpec) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gets the kernel state.
        In the superimposed format, based on the indicated selection.
        This can then be used directly with fast transformers
        :return: The memories tuple.
            - Matrix: Shape (..., d_memory, d_model)
            - Normalizer: Shape (..., d_memory)
        """

        # Get bank selections
        matrix_selections = virtual_state_select(self._matrix, selection, dim=-3, superposition=True)
        normalizer_selections = virtual_state_select(self._normalizer, selection, dim=-2, superposition=True)

        return matrix_selections, normalizer_selections

    def update_state(self,
                     matrix_update: torch.Tensor,
                     normalizer_update: torch.Tensor,
                     selection: SelectionSpec
                     ):
        """
        Updates the memory state with the given matrix updates
        and normalizer updates. The selection probabilities, which were used
        to compose the read state, are combined with the decay factors to
        produce write probabilities. Then a probabilitistic scatter into
        memory is performed.

        :param matrix_update: The matrix update. Shape (..., d_memory, d_model)
        :param normalizer_update: The normalizer update. Shape (..., d_memory)
        :param selection: The selection parameters. For the entire memory
        """

        # Get the memory change rates for the selection. This is the maximum
        # rate is is POSSIBLE for each element to change at, in terms of interpolation
        # probability.

        relevant_decay_rates = virtual_state_select(self._memory_change_rates, selection,
                                                    dim=-2, superposition=False
                                                    )  #(..., selected_memories, d_memory)

        # Combine it with the selection probabilities. We get the scatter selection

        selection_probabilities = selection.selection_probabilities  #(..., selected_memories)
        write_probability = selection_probabilities.unsqueeze(-1) * relevant_decay_rates
        update_directive = SelectionSpec(selection.selection_index, write_probability)

        # Run the updates

        self._matrix = virtual_state_scatter(self._matrix, matrix_update,
                                             update_directive, dim=-3)
        self._normalizer = virtual_state_scatter(self._normalizer, normalizer_update,
                                                 update_directive, dim=-2)


class CreateMemories(nn.Module):
    """
    Creates the linear attention kernels, or "memories",
    used to perform linear expert memory attention. Holds
    the default parameters and initializes properly when
    you tell me the query tensor.
    """

    def __init__(self, num_memories: int, d_memory: int, d_model: int):
        super().__init__()

        # store constants
        self.num_memories = num_memories
        self.d_memory = d_memory
        self.d_model = d_model

        # Set up the initial write strengths. We want decay strengths that corrolate
        # with various decay factors of between 0.2 and 0.0001, distributed about
        # evenly. We go backwards through a sigmoid function to find what these are

        decay_factors = torch.zeros([num_memories, d_memory])
        decay_factors.uniform_(0.0001, 0.2)
        decay_logits = -torch.log((1 / decay_factors) - 1)
        self.write_rate_logits = nn.Parameter(decay_logits, requires_grad=True)

        # Set up the addresses parameter

        self.addresses = nn.Parameter(torch.zeros([num_memories, d_model]), requires_grad=True)
        nn.init.uniform_(self.addresses)

    def forward(self,
                batch_shape: Tuple[int, ...],
                device: torch.device,
                dtype: torch.dtype) -> Memory:
        """
        Creates the memory object, based on the given batch shape.
        :param batch_shape: The batch shape to create the memory out of
        :param device: The device to be on
        :param dtype: The dtype to be on.
        :return: The memory object
        """

        # To create the memories object, we create the normalizer and
        # matrix tensors.

        normalizer = torch.zeros([*batch_shape, self.d_memory], device=device, dtype=dtype)
        matrix = torch.zeros([*batch_shape, self.d_memory, self.d_model], device=device, dtype=dtype)

        # Now make the memory and return it
        return Memory(
            self.addresses,
            normalizer,
            matrix,
            torch.sigmoid(self.write_rate_logits)
        )


class SelectExperts(nn.Module):
    """
    A specialized layer that performs an
    expert selection against the possible memories.
    We get back a selection spec.

    Selection can occur using num_top or num_rand.
    Generally, num_top should always be active to
    some degree. Having a num_rand of 2 or 3
    will ensure that gradients that should connect
    eventually do, though it could take awhile.
    """

    def __init__(self,
                 num_top: int,
                 num_rand: int,
                 dropout: float
                 ):
        super().__init__()

        self.num_top = num_top
        self.num_rand = num_rand
        self.dropout_logits = DropoutLogits(dropout)

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor
                ) -> SelectionSpec:
        """
        Computes the selection spec based on the given query
        and keys.
        :param query: The headed query to use for selection. Shape is (..., num_heads, d_memory)
        :param key: The key of heads to compare against. Shape is (..., num_memory, d_memory)
        :return:
            - The selection spec. Shape is (..., num_heads, retained_memory)
            - Idea is the various retained memories will be superimposed by their probability,
              providing gradients to train off of.
        """

        # Create attention logits. Apply dropout on some of them
        # to promote exploration diversity.
        #
        # Note that this is basically a varient of dot product attention,
        # except we are only using the focus portion of the attention mechanism.

        selection_logits = torch.matmul(query, key.T)  #(..., num_heads, num_memory)
        selection_logits = self.dropout_logits(selection_logits)

        # Narrow down our selection. We select a certain portion
        # by top-k, ensuring rapid training convergance.

        top_logits, top_indexes = selection_logits.topk(self.num_top, dim=-1)

        # Exclude what has already been selected.

        memory_indices = torch.arange(key.shape[-2], device=query.device)  #(num_memory)
        exclusion_mask = ~torch.isin(memory_indices, top_indexes)  #(..., num_memory)
        memory_indices = memory_indices[exclusion_mask]  #(..., reduced_num_memory)

        # Select a certain number randomly from what remains

        randomized_indices = torch.argsort(torch.rand_like(memory_indices), dim=-1)  # (..., reduced_memory)
        randomized_indices = randomized_indices[..., :self.num_rand]  #(..., very_reduced)
        randomized_indices = memory_indices[randomized_indices]  #(..., very_reduced)
        random_logits = selection_logits[..., randomized_indices]

        # Combine both selection mechanisms. Then finally active
        # into probabilities. Return result

        composite_indices = torch.concat([top_indexes, randomized_indices], dim=-1)
        composite_logits = torch.concat([top_logits, random_logits], dim=-1)
        composite_probs = torch.softmax(composite_logits, dim=-1)

        return SelectionSpec(composite_indices, composite_probs)


@recurrent_long_term_memory_registry.register("LinearExpertMemories")
class LinearExpertMemories(RecurrentMemoryAttention):
    """
    The linear expert attention mechanism. A collection of
    experts are consumed each iteration in order to both update the
    memories and get meaningful results. Importantly, the memories
    consist of a large collection of linear kernels which can be
    updated in superposition and sparsely, allowing room for a
    LOT of memories.
    """

    def __init__(self,
                 d_model: int,
                 d_memory: int,
                 num_memories: int,
                 num_heads: int,
                 num_banks: int,
                 num_top_memories: int,
                 num_rand_memories: int,
                 dropout: float
                 ):
        super().__init__(d_model, num_memories, num_banks, dropout)

        # Create the recurrent linear attention unit

        # Create the head generation, and elimination mechanism
        #
        # Notice that the value projections end up wider than memories.

        self.create_query_heads = MakeHeads(d_model, d_memory, num_heads, num_banks)
        self.create_key_heads = MakeHeads(d_model, d_memory, num_heads, num_banks)
        self.create_value_heads = MakeHeads(d_model, d_model, num_heads, num_banks)

        self.merge_heads = MergeHeads(d_model, d_memory, num_heads, num_banks)

        # Create the memory bank selection and creation mechanisms

        self.make_memories = CreateMemories(num_memories, d_memory, d_model)
        self.select_memories = SelectExperts(num_top_memories, num_rand_memories, dropout)

    def forward(self,
                tensor: torch.Tensor,
                bank_selection: SelectionSpec,
                state: Optional[Memory] = None
                ) -> Tuple[torch.Tensor, Memory]:
        """
        Runs the linear expert memory mechanism, including
        random updates and more.

        :param tensor: The tensor to process. Shape (..., d_model)
        :param bank_selection: The bank configuration. Shape is (..., banks)
        :param state: The memory state. May be none
        :return:
            - The tensor response. Same shape
            - The new memory state
        """

        # Standardize shape
        if state is None:
            state = self.make_memories(tensor.shape[:-1], device=tensor.device, dtype=tensor.dtype)

        # Create the projections

        queries = self.create_query_heads(tensor, bank_selection)  # (..., heads, d_memory)
        keys = self.create_key_heads(tensor, bank_selection)  # (..., heads, d_memory)
        values = self.create_value_heads(tensor, bank_selection)  # (..., heads, d_model)

        # Make the selection.

        memory_selection = self.select_memories(queries, state.get_addresses())

        # Now perform attention.
        #
        # Due to the linear kernels involved, we can
        # easily find what the update was by taking the difference
        # between the original and new state

        attention_state = state.get_state(memory_selection)
        output, new_state = self.rla(queries, keys, values, attention_state)
        update_state = map(lambda x, y: x - y, new_state, attention_state)

        # We need to integrate the update. Do so
        #
        # The

        matrix_update, normalizer_update = update_state
        state.update_state(matrix_update, normalizer_update, memory_selection)

        # We are done. Return

        return output, state
