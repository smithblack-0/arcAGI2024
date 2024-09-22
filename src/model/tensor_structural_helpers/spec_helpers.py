"""
Basically

- There is a tree that can be constructed that can be
  used to process incoming data.
-=


"""
import torch
from src.model.channel_bound_tensors import TensorChannelManager
from typing import List, Dict, Any


class StringSpec:
    """
    An association of a channel and list of
    strings with associated integers, and
    the mechanisms to create, insert, and
    remove.
    """

    def __init__(self,
                 channel: str,
                 channel_spec: TensorChannelManager,
                 options: List[str]
                 ):

        assert channel in channel_spec.channel_allocs
        self.channel = channel
        self.channel_spec = channel_spec
        self.options = options

    def create(self, option: str)->torch.Tensor:
        index = self.options.index(option)
        spec = {name : None for name in self.channel_spec.channel_allocs}
        spec[self.channel] = torch.tensor([index])
        return self.channel_spec.combine(spec)

    def set(self, tensor: torch.Tensor, option: str)->torch.Tensor:
        index = self.options.index(option)
        replacement = torch.tensor([index])
        return self.channel_spec.replace(tensor,
                                         self.channel,
                                         replacement)

class ControlsSpec(StringSpec):
    """

    """


class ControlSpec:
    """
    An association of channel and content with
    integers and strings
    """
    def __init__(self,


                 ):
class ModeNode:
    """

    """



def make_controls()

class ModeBranch:
    def __init__(self,
                 children: Dict[Any, Any]
                 ):
        self.children = children

class ModeLeaf:
    """
    The leaf associated with a particular mode
    of encoding or decoding.
    """

class StringLeaf:
    """
    A string leaf associates a
    particular string with a particular
    integer, and can operate in both a
    forward and reverse mode.
    """

class ShapeLeaf:
    """
    A shape leaf is generally responsible for
    binding onto a shape and inserting
    shape data into the tensor stream
    when requested.

    """

class

    controls:
