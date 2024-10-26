"""
A place to put some code that really does not belong elsewhere, that defines primitives used
to make up the tranformer.
"""
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Optional, Any, Tuple

import torch

from src.main.model import registry
from src.main.model.virtual_layers import VirtualLayer, SelectionSpec

MemState = TypeVar('MemState')


class DeepMemoryUnit(VirtualLayer, Generic[MemState], ABC):
    """
    A Deep Memory Unit is a recurrent memory
    device designed to hold memories over an
    extremely long term duration.
    """

    def __init__(self,
                 bank_size: int
                 ):
        super().__init__(bank_size)

    @abstractmethod
    def forward(self,
                tensor: torch.Tensor,
                selection: SelectionSpec,
                state: Optional[Any] = None,
                ) -> Tuple[torch.Tensor, Any]:
        """
        Implementation for the deep memory unit. Based on the tensor, and the
        memstate, produces a new tensor and a new memstate.
        :param tensor: The tensor to use to access and update the mem state. Shape (..., d_model)
        :param selection: The linear kernel selection spec.
        :param state: The memory state. May be none. Implementatin dependent
        :return:
        - The response tensor. Shape (..., d_model)
        - The new mem state. Implementatin dependent
        """


class AbstractComputationStack(ABC):
    """
    Abstract interface for a differentiable computation stack used in adaptive computation.
    This interface defines the core properties and methods that any implementation of
    a computation stack must provide.

    The stack is controlled by three core actions:
    - **Enstack (Push)**: Adds a new element to the top of the stack.
    - **No-op**: Maintains the current stack state without changes.
    - **Destack (Pop)**: Removes an element from the top of the stack.

    Each action is probabilistic, allowing for smooth transitions between states. These actions
    form the basis of adaptive computation for tasks requiring varying levels of recursion or
    step-based processing. The implementations may choose to use the action probabilities, or
    interpret them more concretely, as desired.

    ### Key Properties
    - `normalized_decision_statistics`: Provides normalized statistics on the relative frequency
      of each action (enstack, no-op, destack) at each stack level. This can be used to analyze
      the decision-making pattern over time.
    - `normalized_level_statistics`: Provides normalized statistics on activity across stack levels,
      indicating the relative usage of each level for computational load.
    - `probability_mass`: Returns the total probability mass accumulated over each stack level, useful
      for analyzing the overall computational flow.

    ### Core Methods
    - `adjust_stack`: Adjusts the stack by applying probabilistic actions (enstack, no-op, destack) to
      each level based on given probabilities.
    - `get_expression`: Retrieves the current weighted expression of the stack based on probabilistic pointers.
    - `set_expression`: Sets the stack expression using embeddings and a batch mask.
    - `record_statistics`: Logs statistics on actions and stack transitions based on probabilistic pointers.
    """
    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def normalized_decision_statistics(self) -> torch.Tensor:
        """
        Provides normalized statistics on the actions (enstack, no-op, destack) at each stack level.

        :return: A tensor representing the relative frequency of each action per stack level.
        """
        pass

    @property
    @abstractmethod
    def normalized_level_statistics(self) -> torch.Tensor:
        """
        Provides normalized activity statistics across stack levels, showing the level-wise usage.

        :return: A tensor indicating the relative activity at each stack level.
        """
        pass

    @abstractmethod
    def probability_mass(self) -> torch.Tensor:
        """
        Returns the total accumulated probability mass across all stack levels.

        :return: A tensor with the total probability mass, offering insights into computational flow.
        """
        pass

    @abstractmethod
    def adjust_stack(self, actions_probabilities: torch.Tensor, batch_mask: torch.Tensor):
        """
        Adjusts the stack according to probabilities for enstack, no-op, and destack actions.

        :param actions_probabilities: A tensor of shape (*batch_shape, 3) representing the probabilities for
                                      enstack, no-op, and destack.
        :param batch_mask: A boolean tensor indicating if updates should apply for each batch element.
        """
        pass

    @abstractmethod
    def get_expression(self) -> torch.Tensor:
        """
        Retrieves the current expression of the stack, weighted by probabilistic pointers.

        :return: A tensor of the current weighted stack expression.
        """
        pass

    @abstractmethod
    def set_expression(self, embedding: torch.Tensor, batch_mask: torch.Tensor):
        """
        Sets the stack's expression by integrating the provided embedding using probabilistic pointers.

        :param embedding: A tensor of shape (*batch_shape, d_model) defining the stack's new state.
        :param batch_mask: A boolean tensor controlling which batch elements update the stack.
        """
        pass

    def __call__(self,
                 embedding: torch.Tensor,
                 probabilities: torch.Tensor,
                 batch_mask: torch.Tensor) -> torch.Tensor:
        """
        Executes `set_expression`, `adjust_stack`, and `get_expression` in sequence.

        :param embedding: The embedding to set as the new stack state.
        :param probabilities: Action probabilities for enstack, no-op, and destack.
        :param batch_mask: Mask for batch updates.
        :return: The computed stack expression based on the current state.
        """
        self.set_expression(embedding, batch_mask)
        self.adjust_stack(probabilities, batch_mask)
        return self.get_expression()

# Initialize the various registries
# with their abstract contents.
deep_memory_registry = registry.TorchLayerRegistry[DeepMemoryUnit]("DeepMemoryUnits", DeepMemoryUnit)
