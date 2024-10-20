"""
Heavily inspired by the source code from fast transformers, a set of builders
with adjustable parameters to help me build my layers.
"""
import typeguard
from typing import Dict, Tuple, Union, Type, List, Any


class Flexible:
    """
    A small registry wrapper. Indicates the feature is optional.
    """
    def __init__(self, value):
        self.value = value

class LinearAttentionRegistry:
    """
    Registers a form of linear attention. Keeps track of the layers
    and the optional or required parameters.
    """
    def __init__(self):
        self.classes: Dict[str, Type] = {}
        self.parameters: Dict[str, List[Union[str, Flexible]]] = {}

    def register(self,
                 name: str,
                 class_type: Type,
                 **parameters: Dict[str, Union[str, Flexible]]
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


