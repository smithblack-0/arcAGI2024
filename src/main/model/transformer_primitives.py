"""
A place to put some code that really does not belong elsewhere, that defines primitives used
to make up the tranformer.
"""
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Optional, Any, Tuple, Dict
from .base import TensorTree

import torch
from torch import nn
from torch.nn import functional as F

from src.main.model import registry
from src.main.model.virtual_layers import VirtualLayer, SelectionSpec

MemState = TypeVar('MemState')

##
#
# Adaptive computation time contracts. We implement
# information related to adaptive computation time
# here, including implementation factories, layers
# and more.
#
##



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
    def create_state(self, batch_shape: torch.Size) -> Any:
        """
        Creates the state of the memory unit. Uses batch
        shape to do so.
        :param batch_shape:
        :return:
        """
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


class AbstractComputationStackFactory(ABC):
    """
    Abstract interface for a stateful object designed to control
    and manage a computational stack system. It defines
    the core properties and mechanisms that any such
    layer must handle. The stack implementation is
    supposed to be differentiable.

    ---- setup ----

    It is intended an instance of this is always created through
    a factory method.

    ---- compatible tensors ----

    The first invocation will be done without a state feature, and
    will result in the setup of the computation state. Multiple
    features can be placed in a common stack to be accumulated. So
    long as they are all floating, and the same dtype and device,
    this will happily handle them.

    In particular, let the num batch dims be initialized with
    something like (...batch_shape). Any pytrees with
    leaves only consisting of (...batch_shape, ...other), where
    other can vary between leaves, is compatible so long as they
    have same dtype and device.

    ---- stack probabilities ----

    The stack is usually maintained in terms of controlled probabilities.
    Eventually, the computation will likely start discarding information
    off the end of the stack. When this occurs, we must return the
    discarded probability, as it is used in downstream computations,
    and delete that content.

    """
    def __init__(self,
                 stack_depth: int,
                 d_model: int,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None,
                 ):
        super().__init__()
        """
        Sets up the computation stack
        :param num_batch_dims: The number of dimensions reserved for batching.
        :param dtype: The dtype of the tensors
        :param device: The devices of the tensor.
        """
        self.stack_depth = stack_depth
        self.dtype = dtype
        self.device = device
        self.d_model = d_model
    @abstractmethod
    def forward(self,

                state: Any,
                *tensors: TensorTree
                )->Tuple[Tuple[TensorTree],
                         Any,
                         torch.Tensor]:
        """
        :param embedding: The embedding to form stack actions out of.
            - Shape (...batch_shape, d_model)
        :param max_iterations: The maximum number of iterations before forcing us into flush state
        :param min_iterations: The minumum number of iterations before popping off the top is allowed
        :param state: The recurrent state.
        :param tensors: The various features to store. These can be TensorTrees. They had better
                       have leafs with initial shape (...batch_shape, ...)
        :return: A tuple containing
            - The content retrieved from the stack. Has exactly the same shape as tensors
            - The updated state information
            - Deleted stack probability.
        """



# Initialize the various registries
# with their abstract contents.
deep_memory_registry = registry.InterfaceRegistry[DeepMemoryUnit]("DeepMemoryUnits", DeepMemoryUnit)
stack_registry = registry.InterfaceRegistry[AbstractComputationStack]("ComputationStack", AbstractComputationStack)