"""
Contains the majority of the types
that will be shared between the various
core classes
"""


import torch
import numpy as np
from typing import Callable, Dict, Tuple, Optional
from ..config import Config
from ..data import ActionRequest


# Data intake and batching
DataCase = Dict[str, torch.Tensor]
DataCaseBuffer = Dict[str, DataCase]

# Response after core processing and batch separation. Also, type
# with error stream mixed in.
CaseResponse = Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
ExceptionAugmentedResponse = CaseResponse | Exception

# Some important callbacks, and an important missed buffer

FutureProcessingCallback = Callable[[ExceptionAugmentedResponse], None]
LoggingCallback = Callable[[str|Exception, int], None]
TerminationCallback = Callable[[Optional[bool]], bool]
FutureProcessingBuffer = Dict[str, FutureProcessingCallback]

# config



SHAPES_NAME = Config.SHAPES_NAME
TARGETS_NAME = Config.TARGETS_NAME
CONTEXT_NAME = Config.CONTEXT_NAME
