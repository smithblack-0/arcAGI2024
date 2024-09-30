from src.main.CBTensors.channel_bound_spec import CBTensorSpec
from dataclasses import dataclass
from typing import List, Tuple, Union, Optional
class StringRegistry:
    """
    The string registry is a small utility class designed to allow
    the association of strings with integers. It allows much more nice
    looking definitions
    """
    def __init__(self):
        self.counter = 0
        self.registry = {}
    def register(self, name: str):
        self.registry[self.counter] = name
        self.counter += 1

    def get_mode(self, name: str)->int:
        return self.registry[name]
    def get_name(self, index: int)->str:
        location = list(self.registry.values()).index(index)
        return list(self.registry.keys())[location]

class NumberRegistry:
    """
    Stores a number associated with a string
    """
    def register(self, string: str, number: int):
        self.registry[string] = number
    def __init__(self):
        self.registry = {}
    def get_number(self, string: str) -> int:
        return self.registry[string]

text_vocabulary_size = 100000
intgrid_vocabulary_size = 10
shape_vocabulary_size = 100


data_spec = CBTensorSpec(
    {"state" : 1,
     "mode" : 1,
     "shape" : 2,
     "index" : 2,
     "data" : 1
     }
)




states = StringRegistry()
states.register("mode_select")
states.register("shape_select")
states.register("block_data")
states.register("done")

modes = StringRegistry()
modes.register("text")
modes.register("intgrid")

mode_dims = NumberRegistry()
mode_dims.register("text", 1)
mode_dims.register("intgrid", 2)


# Define the sizes of the vocabularies#
vocabularies = NumberRegistry()
vocabularies.register("mode_select", len(states.registry))
vocabularies.register("shape_select", shape_vocabulary_size)
vocabularies.register("text", text_vocabulary_size)
vocabularies.register("intgrid", intgrid_vocabulary_size)