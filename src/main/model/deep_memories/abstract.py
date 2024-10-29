from abc import ABC, abstractmethod
from typing import Generic, Any, Optional, Tuple, TypeVar

import torch
from src.main.model.base import SavableState
from src.main.model.virtual_layers import VirtualLayer, SelectionSpec
from src.main.model.registry import InterfaceRegistry

MemState = TypeVar('MemState')
class AbstractMemoryState(SavableState):
    """
    A container for the memory state.
    It guarantees the implementation of
    saving and loading for the state.
    """
    def update_(self, *args: Any, **kwargs: Any):
        """
        In place update of abstract memory state. Implementation
        specific.
        """
    def get(self)->Any:
        """
        Implementation specific expression of the memory.
        """

class DeepMemoryUnit(VirtualLayer, ABC):
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
    def create_state(self,
                     batch_shape: torch.Size
                     ) -> AbstractMemoryState:
        """
        Creates the state of the memory unit. Uses batch
        shape to do so.
        :param batch_shape: The shape of the batch
        :return: The setup memory state.
        """
    @abstractmethod
    def forward(self,
                tensor: torch.Tensor,
                selection: SelectionSpec,
                state: AbstractMemoryState
                ) -> torch.Tensor:
        """
        Implementation for the deep memory unit. Based on the tensor, and the
        memstate, produces a new tensor and a new memstate.
        :param tensor: The tensor to use to access and update the mem state. Shape (..., d_model)
        :param selection: The linear kernel selection spec.
        :param state: The memory state.  Implementation dependent
        :return:
        - The response tensor. Shape (..., d_model)
        - Memory is updated indirectly.
        """


deep_memory_registry = InterfaceRegistry[DeepMemoryUnit]("DeepMemoryUnits", DeepMemoryUnit)
