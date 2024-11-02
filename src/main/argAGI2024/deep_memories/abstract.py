from abc import ABC, abstractmethod
from typing import Generic, Any, Optional, Tuple, TypeVar, Dict

import torch
from ..base import SavableState, DeviceDtypeWatch
from ..virtual_layers import VirtualLayer, SelectionSpec
from ..registry import InterfaceRegistry

MemState = TypeVar('MemState')
class AbstractMemoryState(SavableState):
    """
    A container for the memory state.
    It guarantees the implementation of
    saving and loading for the state.
    """

    @abstractmethod
    def get_statistics(self)->Dict[str, Any]:
        """
        Gets a set of relevant statistics.
        :return:
        """
    @abstractmethod
    def update_(self, *args: Any, **kwargs: Any):
        """
        In place update of abstract memory state. Implementation
        specific.
        """
    @abstractmethod
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
    @property
    def device(self)->torch.device:
        return self.__metainfo.device

    @property
    def dtype(self)->torch.dtype:
        return self.__metainfo.dtype
    def __init__(self,
                 bank_size: int,
                 d_model: int,
                 device: Optional[torch.device] = None,
                 dtype: Optional[torch.dtype] = None
                 ):
        super().__init__(bank_size)
        self.d_model = d_model
        self.__metainfo = DeviceDtypeWatch(device=device, dtype=dtype)
    @abstractmethod
    def create_state(self,
                     batch_shape: torch.Size
                     ) -> AbstractMemoryState:
        """
        Creates the state of the memory unit. Uses batch
        shape to do so.
        :param batch_shape: The shape of the batch
        :return:
            - The setup memory state.
        """
    @abstractmethod
    def forward(self,
                tensor: torch.Tensor,
                selection: SelectionSpec,
                memories: AbstractMemoryState,
                ) -> torch.Tensor:
        """
        Implementation for the deep memory unit. Based on the tensor, and the
        memstate, produces a new tensor and a new memstate.
        :param tensor: The tensor to use to access and update the mem state. Shape (..., d_model)
        :param selection: The linear kernel selection spec.
        :param memories: The memory state.  Implementation dependent

        :return:
        - The response tensor. Shape (..., d_model)
        - Memory is updated indirectly.
        """


deep_memory_registry = InterfaceRegistry[DeepMemoryUnit]("DeepMemoryUnits", DeepMemoryUnit)
