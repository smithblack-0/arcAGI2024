"""
Contains the majority of the types
that will be shared between the various
core classes
"""


import torch
from typing import Callable, Dict, Tuple
from ..config import Config
from ..data import ActionRequest

ActionConstructionCallback = Callable[[str, Dict[str, torch.Tensor]], ActionRequest]
BatchEntry = Tuple[torch.Tensor, torch.Tensor] # First tensor, then mask.
RequestBuffer = Dict[str, Tuple[ActionConstructionCallback, ActionRequest]]
BatchCaseBuffer = Dict[str, Dict[str, torch.Tensor]]
MetadataPayload = Tuple[str, ActionConstructionCallback]
LoggingCallback = Callable[[str|Exception, int], None]
TerminationCallback = Callable[[], bool]

SHAPES_NAME = Config.SHAPES_NAME
TARGETS_NAME = Config.TARGETS_NAME
CONTEXT_NAME = Config.CONTEXT_NAME
