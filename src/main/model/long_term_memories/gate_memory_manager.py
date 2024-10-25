from typing import Optional, Tuple

import torch
from torch import nn

from src.main.model.long_term_memories.core import VirtualMemoryManager, MemoryType
from src.main.model.virtual_layers import DropoutLogits, VirtualLinear, SelectionSpec

memory_type = torch.Tensor


class ControlGate(nn.Module):
    """
    Produces a gate when provided a headed query.
    This will address individual memories that may be
    under consideration, and fetch information about them.

    There is both a focus along the different memory
    banks, and a focus along the elements in the memory.
    The latter allows for focus to be placed on memory
    elements that may decay at different rates, if you
    need to consider short vs long term memories.

    Two activation methods are supported. These
    are "sigmoid" and "softmax".

    This gate can be used to produce read probabilities,
    and also write probabilities.
    """

    def __init__(self,
                 d_memory: int,
                 num_memories: int,
                 bank_size: int,
                 activation: str,
                 dropout: Optional[float] = None,
                 ):
        """
        :param d_memory: The size of the memory elements
        :param num_memories: The number of memores
        :param activation: One of "sigmoid", "softmax"
        """
        if dropout is None:
            dropout = 0.0

        assert activation in ["sigmoid", "softmax"]
        super().__init__()

        self.d_memory = d_memory
        self.num_memories = num_memories
        self.activation = activation

        # Two gates.
        self.dropout_logits = DropoutLogits(dropout)
        self.memory_focus = VirtualLinear(d_memory, num_memories, bank_size)
        self.element_focus = VirtualLinear(d_memory, d_memory, bank_size)

    def forward(self,
                query: torch.Tensor,
                selection: SelectionSpec,
                ) -> torch.Tensor:
        """
        Runs the read gate process, producing the read
        probabilities.
        :param query: The query under consideration. Shape (..., heads, d_memory)
        :param selection: The bank selection.
        :return: The read probabilities. Shape (..., heads, memory_size, d_memory)
        """

        # Create focus logits
        memory_focus = self.memory_focus(query, selection)  # (..., heads, memory_size)
        element_focus = self.element_focus(query, selection)  # (..., heads, d_memory)

        # Dropout
        memory_focus = self.dropout_logits(memory_focus)

        # Activate
        if self.activation == "sigmoid":
            memory_focus = torch.sigmoid(memory_focus)
            element_focus = torch.sigmoid(element_focus)
        else:
            memory_focus = torch.softmax(memory_focus, dim=-1)
            element_focus = torch.softmax(element_focus, dim=-1)

        # Merge. Return
        focus_probabilities = memory_focus.unsqueeze(-1) * element_focus.unsqueeze(-2)
        return focus_probabilities


class EraseGate(nn.Module):
    """
    The erase gate, as the name seems to imply,
    is responsible for providing information
    on how to erase memories.

    Erasure of memories depends on two things. First,
    for it to happen the write gate must be active,
    and there is always some amount of probability
    that decays away when it is so.

    Second, the model can decide to deliberately
    erase memories it deems irrelevant to the
    current task. It also adjusts the write to
    account for the relevant decay constants
    """

    def __init__(self,
                 d_memory: int,
                 num_memories: int,
                 bank_size: int,
                 activation: str,
                 dropout: Optional[float]
                 ):
        super().__init__()

        self.d_memory = d_memory
        self.num_memories = num_memories
        self.activation = activation

        # Erase gate

        self.erase_gate = ControlGate(d_memory, num_memories, bank_size, activation, dropout)

        # And the decay logits. Decay rates are defined per memory element
        # and the model can decide how much to make short vs long term.
        # We initialize the logits based on what decay factor they will
        # activate to.

        decay_factors = torch.zeros([d_memory])
        decay_factors.uniform_(0.0001, 0.2)
        decay_logits = -torch.log((1 / decay_factors) - 1)
        self.decay_logits = nn.Parameter(decay_logits, requires_grad=True)

    def forward(self,
                query: torch.Tensor,
                statement: torch.Tensor,
                write_probabilities: torch.Tensor,
                selection: SelectionSpec,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Figures out the erase probability and decayed write
        tensor based on the tensor and internally stored
        write probabilities
        :param query: The tensor input. Shape (..., heads, d_memory)
        :param statement: The statement to write. Shape (..., heads, d_memory)
        :param write_probabilities: The write probabilities. Shape (..., heads, memory_size, d_memory)
        :param selection: The bank selection.
        :return:
            - erase_probabilities. Shape (..., heads, memory_size, d_memory)
            - decayed tensor. Shape (..., heads, d_memory). Will be written.
        """

        # Create focus and decay probabilities

        erase_focus = self.erase_gate(query, selection)  # (..., heads, memory_size, d_memory)
        decay_probabilities = torch.sigmoid(self.decay_logits)  # (d_memory)

        # The erase probability is proportional to the erase focus times the write probability. However,
        # the erase probability is at least equal to the decay probability. That is, there is a minimum
        # amount of erasure that must happen just because we are writing. This prevents memory
        # saturation.

        erase_focus = torch.max(erase_focus, decay_probabilities)
        erase_probabilities = erase_focus * write_probabilities

        # Finally, the tensor that was originally provided needs to be scaled according to the
        # decay probability. This prevents greatly differing element magnitudes.

        statement = statement * decay_probabilities
        return erase_probabilities, statement


class GateMemoryManager(VirtualMemoryManager[memory_type]):
    """
    Memories are managed using access and write gates.
    Behavior is not too dissimilar to how the NTM
    works, though without the state loop.

    ---- heads ----
    queries are recieved in terms of headed information.
    it is assumed we want to do an operation for each recieved head.

    --- read gate ---

    * reading is a weighted sum based on the read gate
    * weight based on activation

    ---- writing ---

    * writing is controlled by the write gate, the erase gate, and the decay gate
    * We write to a slot by adding into it, according to the write gate probability
    * We can also choose to erase when both write and erase gates are true
    *
    """

    def __init__(self,
                 d_memory: int,
                 num_memories: int,
                 bank_size: int,
                 activation: str,
                 dropout: Optional[float] = None,
                 ):
        assert activation in ["sigmoid", "softmax"]
        super().__init__(d_memory, num_memories, bank_size)

        # Create control gates
        self.read_gate = ControlGate(d_memory, num_memories, bank_size, activation, dropout)
        self.write_gate = ControlGate(d_memory, num_memories, bank_size, activation, dropout)
        self.erase_gate = EraseGate(d_memory, num_memories, bank_size, activation, dropout)
        self.merge_layer = VirtualLinear(num_memories, 1, bank_size)

        # Create default memories
        self.default_memories = nn.Parameter(torch.empty([num_memories, d_memory]).uniform_(-0.1, 0.1))

    def create_memory(self, shape: torch.Tensor) -> torch.Tensor:
        """
        Creates the memories to be in shape, based on the
        default memory parameters
        :param shape: The shape it should be
        :return: The memory. Shape (...shape, num_memories, d_memory)
        """
        expansion = [*shape] + [-1, -1]
        memory = self.default_memories
        for _ in shape:
            memory = memory.unsqueeze(0)
        return memory.expand(*expansion)

    def express_memory(self,
                       query: torch.Tensor,
                       memories: MemoryType,
                       selection: SelectionSpec
                       ) -> torch.Tensor:
        """
        Expresses a memory for each head using the read gates
        :param query: The query. Shape (..., heads, d_memory)
        :param memories: The memories. Shape (..., num_memories, d_memory)
        :param selection: The virtual layer selection
        :return: The expressed memorry. Shape (..., heads, d_memory)
        """

        read_probabilities = self.read_gate(query, selection)  # (..., heads, num_memories, d_memory)
        response = memories.unsqueeze(-3) * read_probabilities  # (..., heads, num_memories, d_memory)
        return response.sum(dim=-2)

    def store_memory(self,
                     query: torch.Tensor,
                     statement: torch.Tensor,
                     memories: MemoryType,
                     selection: SelectionSpec
                     ) -> MemoryType:
        """
        Uses the query to decide how to store the statement into the memory.
        :param query: The query to use for storage. Shape (..., heads, d_memory)
        :param statement: The statement to store into memory. Shape (..., heads, d_memory)
        :param memories: The memory type. Shape (..., memories, d_memory)
        :param selection: The selection specification.
        :return: The new memories.
        """
        write_probabilities = self.write_gate(query, selection)  # (..., heads, memory_size, d_memory)
        erase_probabilities, write_statement = self.erase_gate(query, statement, selection)  # (..., heads, mem, dmem)
        memory_update = memories.unsqueeze(-3) * (1 - erase_probabilities) + \
                        write_statement * write_probabilities  #(..., heads, memory_size, d_memory)
        memory_update = memory_update.mean(dim=-3)  #(...., memory_size, d_memory)
        return memory_update
