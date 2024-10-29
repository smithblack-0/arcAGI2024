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
                       query: torch.Tensor,
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


# Define the straightforward
# read process.
#
# This does linear kernel read.
class ReadState(nn.Module):
    """
    Performs read from memory state.
    This is fairly straightforward. We create the
    queries, then use them to access the state.
    Queries are run through the kernel function before
    usage. It follows the ideas of linear kernel
    attention.

    This does not perform any sort of memory update,
    as that is both done elsewhere and by different
    mechanisms.

    Internally, the memories are actually stored in
    a vast collection of heads.
    """

    def __init__(self,
                 d_model: int,
                 d_head: int,
                 num_heads: int,
                 bank_size: int,
                 dropout: float,
                 kernel_activation: Callable[[torch.Tensor], torch.Tensor],
                 addresses: nn.Parameter,
                 device: torch.device,
                 dtype: torch.dtype,
                 ):
        # Addresses: (..., num_memories, num_heads, d_head)
        super().__init__()

        self.d_model = d_model
        self.d_head = d_head
        self.bank_size = bank_size

        # Create the query creator, and also
        # the head merger.

        self.kernel_activation = kernel_activation
        self.head_creator = VirtualAdvancedLinear(d_model, [num_heads, d_head], bank_size,
                                                   device=device, dtype=dtype)
        self.head_merger = VirtualAdvancedLinear([num_heads, d_head], d_model, bank_size)
        self.dropout_logits = DropoutLogits(dropout)

        # Store the addresses
        self.addresses = addresses
    def dot_product_weights(self,
                            queries: torch.Tensor,
                            keys: torch.Tensor,
                            )->torch.Tensor:
        # Create read focus. This is an attention weighted subset of the memories.
        #
        # So we are essentially performing attention twice. There is no speedup
        # to be gained, though, as we cannot reuse any state here, so just normal
        # dot product attn


        queries = queries.movedim(-2, -3) # (..., num_heads, 1, d_head)
        keys = keys.movedim(-2, -3) # (..., num_heads, num_memories, d_head)

        weights = torch.matmul(queries, keys.swapdims(-2, -1)) # (..., num_heads, 1, num_memories)
        weights = weights/torch.sqrt(torch.tensor(self.d_head))
        weights = self.dropout_logits(weights)
        weights = torch.softmax(weights, dim=-1)
        weights = weights.movedim(-1, -3) # (..., num_memories, num_heads, 1)
        return weights
    def forward(self,
                tensor: torch.Tensor,
                state: MemoryState,
                selection: SelectionSpec
                ) -> torch.Tensor:
        """
        Performs only the read action. This consists of
        :param tensor: The tensor to run the read with.
        :param state: The state of the memory to read from.
            - matrix: (..., num_memories, d_key, d_value)
            - normalizer: (..., num_memories, d_key)
        :param selection: The virtual layer selection
        :return:
            - results: The result of reading from memory, using linear kernels.
        """
        # Unpack state
        matrix, normalizer = state.get()
        # matrix: (..., num_memories, num_heads, d_key, d_value)
        # normalizer: (..., num_memories, num_heads, d_key)

        # Create read queries, and attention addresses.
        #
        # The read queries are a projection, with an extra dimension for num_memories
        # The addresses comes from adding the normalizer state to the addresses parameter,
        # giving us a unique idea of what each piece is doing.

        queries = self.head_creator(tensor, selection).unsqueeze(-3)  # (..., 1, num_heads, d_head)
        addresses = normalizer + self.addresses.unsqueeze(-2) #(..., num_memories, num_heads, d_head)

        # Run linear attention kernel

        numerator = torch.matmul(matrix.swapdims(-1, -2), queries.unsqueeze(-1)).squeeze(
            -1)  # (..., num_memories, num_heads, d_head)
        denominator = torch.sum(queries * normalizer, dim=-1, keepdim=True) + 1e-9
        result = numerator / denominator # (..., num_memories, num_heads, d_head)

        # Combine results, then get rid of the heads
        result = torch.sum(weights * result, dim=-3) # (..., num_heads, d_head)
        result = self.head_merger(result)

        # Return
        return result


##
# Define all the features used in the write
# process. This involves quite a few gates
##




class WriteState(nn.Module):
    """
    Performs the specialized linear kernel attention
    write process, with decay, write gates, and
    erase gates.

    Writing is by its nature a destructive process. In
    particular, it is the case that to write, there
    is a minimum amount of memory that must decay. This
    keeps the memory state from getting arbitrarily
    large.
    """
    def dot_product_weights(self,
                            queries: torch.Tensor,
                            keys: torch.Tensor,
                            activation: str
                            )->torch.Tensor:
        # Create read focus. This is an attention weighted subset of the memories.
        #
        # So we are essentially performing attention twice. There is no speedup
        # to be gained, though, as we cannot reuse any state here, so just normal
        # dot product attn


        queries = queries.movedim(-2, -3) # (..., num_heads, 1, d_head)
        keys = keys.movedim(-2, -3) # (..., num_heads, num_memories, d_head)

        weights = torch.matmul(queries, keys.swapdims(-2, -1)) # (..., num_heads, 1, num_memories)
        weights = weights/torch.sqrt(torch.tensor(self.d_head))
        weights = self.dropout_logits(weights)
        if activation == "softmax":
            weights = torch.softmax(weights, dim=-1)
        elif activation == "sigmoid":
            weights = torch.sigmoid(weights)
        else:
            raise RuntimeError()
        weights = weights.movedim(-1, -3) #(..., num_memories, num_heads, 1)
        return weights
    def __init__(self,
                 d_model: int,
                 d_head: int,
                 num_heads: int,
                 num_memories: int,
                 bank_size: int,
                 dropout: float,
                 kernel_activation: Callable[[torch.Tensor], torch.Tensor],
                 ):
        super().__init__()

        self.d_model = d_model
        self.d_head = d_head
        self.num_memories = num_memories
        self.bank_size = bank_size
        self.kernel_activation = kernel_activation

        # Create the key and value creators.

        self.make_key_heads = VirtualAdvancedLinear(d_model, [num_heads, d_head], bank_size)
        self.make_value_head = VirtualAdvancedLinear(d_model, [num_heads, d_head], bank_size)

        # Create the write and erase gates.

        self.write_gate_projector = VirtualAdvancedLinear(d_head,


        # Create the decay rates to associate with each key, and
        # with each value within a key. The inverse sigmoid function
        # is used to ensure we start with logits activating to factors
        # between 0 and 0.2

        def initialize_decay_logits(shape: torch.Size,
                                    low: float,
                                    high: float) -> nn.Parameter:
            decay_factors = torch.zeros(shape)
            decay_factors.uniform_(low, high)
            decay_logits = torch.log(decay_factors / (1 - decay_factors))
            decay_logits = nn.Parameter(decay_logits)
            return decay_logits

        self.key_decay_logits = initialize_decay_logits([d_key], 0.0, 0.2)
        self.value_decay_logits = initialize_decay_logits([d_key, d_value], 0.0, 1.0)

    def forward(self,
                tensor: torch.Tensor,
                state: MemoryState,
                selection: SelectionSpec
                ):
        """
        Performs the update process for the linear attention mechanism,
        consisting of creating the updates then passing them through
        the gates.

        :param tensor: The tensor. Shape (..., d_model)
        :param state: The state of the memory to read from.
            - matrix: (..., num_memories, d_key, d_value)
            - normalizer: (..., num_memories, d_key)
        :param selection: The virtual layer selection.
        """
        # Basically, when it comes to writing, we compute the updates
        # for all cases. Then we use attention with write and erase
        # systems to determine what to update, where, and by how much.



        # Create headed features, and activate them by running it through
        # the kernels.

        headed_keys = self.make_key_heads(tensor, selection)  # (..., 1, d_head,  d_key)
        headed_values = self.make_value_heads(tensor, selection)  # (..., memories, d_value)
        headed_keys = self.kernel_activation(headed_keys)

        # Create the two decay probability variations. In keeping with the linear kernel,
        # we define separate decay probabilities for the normalizer and the matrix. However,
        # decay in the matrix is only possible if it is being accessed as a key through the
        # normalizer.
        #
        # This produces behavior where the model, if writing to a key, has a chance to both
        # be influenced by the overall decay rate for that key slot, then by the decay
        # rates for the values in the key slot.

        normalizer_decay_probability = torch.sigmoid(self.key_decay_logits)  # (...., memories, d_key)
        matrix_decay_probability = torch.sigmoid(self.value_decay_logits)  # (..., memories, d_value
        matrix_decay_probability = matrix_decay_probability * normalizer_decay_probability.unsqueeze(-1)

        # Create write and erase probabilities. These are now being defined per d_key slot.
        # Notice that we MUST give an erase probability above a certain point due to the
        # decay factors, but we may choose to erase more if we wish.

        normalizer_write_probabilities = self.write_gate(tensor, selection)  # ( ..., memories, d_key)
        matrix_write_probabilities = normalizer_write_probabilities.unsqueeze(-1)  # (..., memories, d_key, 1)

        normalizer_erase_probabilities = self.erase_gate(tensor, selection)  # ( ..., memories, d_key)
        matrix_erase_probabilities = normalizer_erase_probabilities.unsqueeze(-1)

        normalizer_erase_probabilities = torch.max(normalizer_erase_probabilities, normalizer_decay_probability)
        matrix_erase_probabilities = torch.max(matrix_erase_probabilities, matrix_decay_probability)
        # Create the actual update to be integrated. This is the normal kernel attention process.
        # Also, get the current matrix and normalizer to integrate with. We also scale
        # down the updates by the strength of the decay probabilities, which lets it act
        # as a running average over the writes.

        matrix, normalizer = state.get()

        matrix_update = headed_keys.unsqueeze(-1) * headed_values.unsqueeze(-2)  #(..., memories, d_key, d_value)
        normalizer_update = headed_keys  # (..., memories, d_key)

        matrix_update = matrix_update * matrix_decay_probability
        normalizer_update = normalizer_update * normalizer_decay_probability

        # Perform the update process. We push an update based on the strength of the write
        # probabilities. We also erase, but we can only erase when we were otherwise
        # already in the process of writing

        matrix = matrix * (1 - matrix_write_probabilities * matrix_erase_probabilities) + \
                 matrix_update * matrix_write_probabilities
        normalizer = normalizer * (1 - normalizer_write_probabilities * normalizer_erase_probabilities) + \
                     normalizer_update * normalizer_write_probabilities

        # Store the new spec
        state.update_(matrix, normalizer)


@deep_memory_registry.register("LinearKernelMemoryBank")
class LinearKernelMemoryBank(DeepMemoryUnit):
    """
    The linear kernel memory bank is designed to provide
    extremely long term memory building capability and
    dynamic adaptation of the model.

    Think of the idea of allowing the model to read
    a database of languages, and it learns how to
    interpret them, and you are not too far off.

    ---- linear kernel bank ----

    The memories are, basically, stored in linear kernel
    heads. Since this is a virtual layer, we can configure
    the method of access to attempt to access or write
    to different heads at different strengths on the fly.

    The linear kernel attention process is being expressed
    in a recurrent self attention format, in which an
    accumulated sum of certain key and value entries
    can be used to skip explicitly attending to everything.

    ---- reading ----

    Reading from the memory can be done without penalty,
    and follows the normal process. See Read State for more details

    ---- writing ----

    To support the objective, writing is SIGNIFICANTLY modified.
    In particular, writing now requires going through a write gate,
    which will integrate an update into the accumulators, meaning
    not every update will actually make it into the memory to be
    integrated.

    This write gate also pairs with an erase gate to nicely mean
    that the model can choose to erase a memory unit, and that
    writing has a minumum amount of erasure that must happen.
    This prevents the cells from accumulating exceptionally
    large values.

    The memory cells actually contain, basically, a weighed average.
    What is going to be written is scaled down before write to
    match this average.
    """

    def __init__(self,
                 d_model: int,
                 mem_d_value: int,
                 num_memories: int,
                 bank_size: int,
                 submodule_dropout: Optional[float] = None,
                 kernel_activation: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                 ):
        """
        :param d_model: The size of the incoming tensors to build and access memories from
        :param mem_d_value: The size of the internal value dimension. Larger here will mean more memory slots
        :param num_memories: The number of memories to store. Heads.
        :param bank_size: The size of the virtual layer bank
        :param submodule_dropout: The dropout strength
        :param kernel_activation: The activation kernel. Defaults to relu
        """
        super().__init__(bank_size)

        if kernel_activation is None:
            kernel_activation = F.elu
        if submodule_dropout is None:
            submodule_dropout = 0.0

        self.state_creator = CreateState(d_model, mem_d_value, num_memories)
        self.read_state = ReadState(d_model, d_model, mem_d_value, num_memories,
                                    bank_size, submodule_dropout, kernel_activation)
        self.write_state = WriteState(d_model, d_model, mem_d_value, num_memories,
                                      bank_size, submodule_dropout, kernel_activation)

    def create_state(self,
                     batch_shape: torch.Size
                     ) -> MemoryState:
        """
        Creates the default memory state
        :param batch_shape: The batch shape to build it around
        :return: The memory state container.
        """
        return self.state_creator(batch_shape)

    def forward(self,
                tensor: torch.Tensor,
                selection: SelectionSpec,
                state: MemoryState
                ) -> torch.Tensor:
        """
        Runs the linear kernel memory attention process
        :param tensor: The tensor to access and update with. Shape (..., d_model)
        :param selection: The virtual layer selection
        :param state: The setup memory state. Need one? Use create state
        :return:
            - The response. (..., d_model)
            - The new memory state.
        """
        self.write_state(tensor, state, selection)
        response = self.read_state(tensor, state, selection)
        return response
