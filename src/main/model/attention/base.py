"""
The base recurrent attention class, the interface contract,
and the builder registry all in one place. The outside world
learns how to interface from here.
"""

import torch
from torch import nn
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Union, Type, List, Any


class RecurrentAttention(nn.Module, ABC):
    """
    The base contract for the recurrent linear attention
    mechanism. Includes the three slots for the standard attention
    parameters, then an additional state slot as well.
    """

    @abstractmethod
    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                state: Any) -> Tuple[torch.Tensor, Any]:
        """
        Heavily inspired by the source code from fast transformers, a set of builders
        with adjustable parameters to help me build my layers.
        """
class LinearAttentionRegistry:
    """
    Registers a form of linear attention. Keeps track of the layers
    and the optional or required parameters.
    """

    def __init__(self):
        self._classes: Dict[str, Type] = {}
        self._parameters: Dict[str, Dict[str, Any]] = {}

    def register(self,
                 name: str,
                 class_type: RecurrentAttention,
                 **parameters: Dict[str, Union[str, Any]]
                 ):
        """
        Register a particular class to be associated with a particular
        :param name: name of the class
        :param class_type: The class being registered
        """
        if name in self.classes:
            raise ValueError(f"Class {name} already registered")

        self.classes[name] = class_type
        self.parameters[name] = parameters

    def build(self, name: str, **parameters: Dict[str, Any]):
        """
        Builds the attention mechanism.
        :param name: The name of the class
        :param parameters: The parameters we MIGHT build it with
        :return: The instance
        """
        # Get the parameters and class reference
        class_type = self.classes[name]
        parameter_spec = self.parameters[name]

        # Create a subgroup of parameters to pass into the constructor.
        subparameters = {parameters[name] if name in self.parameter_spec

        # Run the constructor
        return class_type(**parameter_spec)


