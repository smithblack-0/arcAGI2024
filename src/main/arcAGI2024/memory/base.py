import torch
from torch import nn
from typing import Tuple
from ..base import SavableState, DeviceDtypeWatch
from abc import ABC, abstractmethod
class AbstractMemoryState(SavableState):
    """
    An abstract, and common, memory state.

    All memory implementation will subclass
    from this. It should be a datastructure
    which implements a save/load contract
    so parallel pytree map can implement it.
    """

    @abstractmethod
    def get(self)->Tuple[torch.Tensor, ...]:
        """
        Gets a relevant memory implementation in the appropriate order.
        """

class AbstractCreateState(nn.Module, ABC):
    """
    Creates a blank memory state based on the
    batch shape. The forward method needs to
    be defined.
    """

    @property
    def device(self) -> torch.device:
        return self.__metainfo.device

    @property
    def dtype(self) -> torch.dtype:
        return self.__metainfo.dtype

    def __init__(self, dtype: torch.dtype, device: torch.device):
        super().__init__()
        self.__metainfo = DeviceDtypeWatch(device=device, dtype=dtype)

    @abstractmethod
    def forward(self, batch_shape: torch.Size) -> AbstractMemoryState:
        """
        When implemented, returns a memory object when seeing a batch shape
        :param batch_shape: The batch shape to consider
        :return: The memory.
        """

class