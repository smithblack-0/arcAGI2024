from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Optional, Any, Tuple, Dict

import torch
from torch import nn

from ..base import TensorTree, SavableState
from ..registry import InterfaceRegistry

class AbstractComputationSupportStack(ABC, SavableState):
    """
    The abstract implementation of a computational support
    stack. It indicates the methods the external world
    must implement in order to work with the stack state,
    including the various adjustment mechanisms and the
    save/load requirements.
    """


