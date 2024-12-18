from typing import Optional, Tuple, List, Callable, Any

import torch
from dataclasses import dataclass
from torch import nn
from torch.nn import functional as F

from ..base import TensorTree
from ..deep_memories.abstract import DeepMemoryUnit, deep_memory_registry, AbstractMemoryState
from ..virtual_layers import (DropoutLogits, VirtualLinear,
                                                VirtualMakeHeads, VirtualMergeHeads,

                                                SelectionSpec)


# Define the memory state

class MemoryState(AbstractMemoryState):
    """
    The memory state. Contains within it
    the linear kernel attention matrix
    and normalizer

    matrix: (...batch_shape, d_key, d_value)
    normalizer: (...batch_shape, d_key)
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

    def get(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.matrix, self.normalizer


##
# State creation mechanism
##

class CreateState(nn.Module):
    """
    Creates the default attention state when requested
    """

    def __init__(self,
                 d_key: int,
                 d_value: int,
                 num_memories
                 ):
        super().__init__()
        self.d_key = d_key
        self.d_value = d_value
        self.num_memories = num_memories

    def forward(self, batch_shape: torch.Size) -> MemoryState:
        """
        Sets up the state.
        :param batch_shape: The batch shape that is corrolated with the memories
        :return: The setup memory state.
        """
        matrix = torch.zeros([*batch_shape, self.num_memories, self.d_key, self.d_value])
        normalizer = torch.zeros([*batch_shape, self.num_memories, self.d_key])
        return MemoryState(matrix, normalizer)


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
                 d_key: int,
                 d_value: int,
                 num_memories: int,
                 bank_size: int,
                 dropout: float,
                 kernel_activation: Callable[[torch.Tensor], torch.Tensor],
                 ):
        super().__init__()

        self.d_model = d_model
        self.d_key = d_key
        self.num_memories = num_memories
        self.bank_size = bank_size

        # Create the query creator, and also
        # the head merger.

        self.kernel_activation = kernel_activation
        self.query_creator = VirtualMakeHeads(d_model, d_key, num_memories, bank_size)
        self.merge_heads = VirtualMergeHeads(d_model, d_value, num_memories, bank_size)
        self.dropout_logits = DropoutLogits(dropout)

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
        # matrix: (..., num_memories, d_key, d_value)
        # normalizer: (..., num_memories, d_key)

        # Create queries
        queries = self.query_creator(tensor, selection)  # (..., num_memories, d_key)
        queries = self.dropout_logits(queries.movedim(-2, -1)).movedim(-1, -2)
        queries = self.kernel_activation(queries)

        # Run attention
        print(matrix.swapdims(-1, -2).shape)
        print(queries.unsqueeze(-1).shape)
        numerator = torch.matmul(matrix.swapdims(-1, -2), queries.unsqueeze(-1)).squeeze(
            -1)  # (..., num_memories, d_value)
        denominator = torch.sum(queries * normalizer, dim=-1, keepdim=True) + 1e-9
        result = numerator / denominator

        # Remove heads

        result = self.merge_heads(result, selection)

        # Return
        return result


##
# Define all the features used in the write
# process. This involves quite a few gates
##

class ControlGate(nn.Module):
    """
    Creates probabilities associated with some sort of control action,
    such as write probabilities or erase probabilities. Functions by
    accepting an unheaded tensor, and creating headed probabilities
    per indicated element indicating whether or not to write in
    that location.

    A "shape" is provided to return as, which can be just one
    dimension, like [d_key], or multiple dimensions, like
    [d_key, d_value]. Only the last dimension will be activated
    or dropped out when using activations that care about that.

    Supports two methods of activation. Sigmoid will indicate
    per individual element whether you need to be modified,
    while softmax will ensure something is interacted with.
    """

    def __init__(self,
                 d_model: int,
                 shape: torch.Size,
                 num_heads: int,
                 bank_size: int,
                 activation: str,
                 dropout: Optional[float] = None,
                 ):

        # Standardize and sanitize
        if dropout is None:
            dropout = 0.0
        assert activation in ["sigmoid", "softmax"]

        super().__init__()

        # Store features
        self.d_model = d_model
        self.output_shape = shape
        self.num_heads = num_heads
        self.activation = activation

        # Store primary worker
        self.dropout_logits = DropoutLogits(dropout)
        self.control_gate = VirtualMakeHeads(d_model, torch.prod(torch.tensor(shape)), num_heads, bank_size)

    def forward(self,
                tensor: torch.Tensor,
                virtual_layer_selection: SelectionSpec) -> torch.Tensor:
        """
        Runs the gate construction process. This means
        turning our incoming tensor into logits per element,
        then activating into probabilities.
        :param tensor: The tensor to use for construction. Shape (..., d_model)
        :param virtual_layer_selection: The virtual layer selection.
        :return: The gate probabilities. Shape (..., num_heads, d_head)
        """
        logits = self.control_gate(tensor, virtual_layer_selection)
        logits = logits.unflatten(-1, self.output_shape)
        logits = self.dropout_logits(logits)
        if self.activation == "sigmoid":
            probabilities = torch.sigmoid(logits)
        else:
            probabilities = torch.softmax(logits, dim=-1)
        return probabilities


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

    def __init__(self,
                 d_model: int,
                 d_key: int,
                 d_value: int,
                 num_memories: int,
                 bank_size: int,
                 dropout: float,
                 kernel_activation: Callable[[torch.Tensor], torch.Tensor],
                 ):
        super().__init__()

        self.d_model = d_model
        self.d_key = d_key
        self.num_memories = num_memories
        self.bank_size = bank_size
        self.kernel_activation = kernel_activation

        # Create the key and value creators.

        self.make_key_heads = VirtualMakeHeads(d_model, d_key, num_memories, bank_size)
        self.make_value_heads = VirtualMakeHeads(d_model, d_value, num_memories, bank_size)

        # Create the erase and  write control gates.
        self.write_gate = ControlGate(d_model, [d_key],
                                      num_memories, bank_size,
                                      "sigmoid", dropout)
        self.erase_gate = ControlGate(d_model, [d_key],
                                      num_memories, bank_size,
                                      "sigmoid", dropout)

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

        # Create headed features, and activate them by running it through
        # the kernels.

        headed_keys = self.make_key_heads(tensor, selection)  # (..., memories, d_key)
        headed_values = self.make_value_heads(tensor, selection)  # (..., memories, d_value)
        headed_keys = self.kernel_activation(headed_keys)

        # Create the two decay probability variations. In keeping with the linear kernel,
        # we define separate decay probabilities for the normalizer and the matrix. However,
        # decay in the matrix is only possible if it is being accessed as a key through the
        # normalizer.
        #
        # This produces behavior where the arcAGI2024, if writing to a key, has a chance to both
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

        matrix_update = headed_keys.unsqueeze(-1) * headed_values.unsqueeze(-2)  # (..., memories, d_key, d_value)
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


@deep_memory_registry.register("LinearKernelMemoryBankOld")
class LinearKernelMemoryBank(DeepMemoryUnit):
    """
    The linear kernel memory bank is designed to provide
    extremely long term memory building capability and
    dynamic adaptation of the arcAGI2024.

    Think of the idea of allowing the arcAGI2024 to read
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
    that the arcAGI2024 can choose to erase a memory unit, and that
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
                memories: MemoryState
                ) -> torch.Tensor:
        """
        Runs the linear kernel memory attention process
        :param tensor: The tensor to access and update with. Shape (..., d_model)
        :param selection: The virtual layer selection
        :param memories: The setup memory state. Need one? Use create state
        :return:
            - The response. (..., d_model)
            - The new memory state.
        """
        self.write_state(tensor, memories, selection)
        response = self.read_state(tensor, memories, selection)
        return response
