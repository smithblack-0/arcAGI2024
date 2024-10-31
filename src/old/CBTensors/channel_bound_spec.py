from functools import cached_property
from types import MappingProxyType
from typing import Dict, List


class CBTensorSpec:
    """
    The CB tensor spec is designed to hold channel binding (CB) data
    for a channel bound tensor. This means tracking both the channels, and the
    widths. We also provide a bunch of informative statistics.

    ---- properties ----

    channels: The channels that are being represented, and in what order.
    channel_width: For each channel name, how many elements wide that channel is
    total_width: The width of all the channels put together. The sum of the individual lengths
    start_index: For each channel, what the start index for the channel is.
    end_index: For each channel, what the end index for the channel is.
    slices: For each channel, what the slice addressing that channel would be.
    """

    spec: Dict[str, int]

    @property
    def channels(self)->List[str]:
        return list(self.spec.keys())

    @property
    def channel_widths(self)->Dict[str, int]:
        return self.spec.copy()

    @property
    def total_width(self)->int:
        return sum(self.spec.values())

    @cached_property
    def start_index(self)->Dict[str, int]:
        position = 0
        output = {}
        for name, length in self.spec.items():
            output[name] = position
            position += length
        return output

    @cached_property
    def end_index(self)->Dict[str, int]:
        position = 0
        output = {}
        for name, length in self.spec.items():
            position += length
            output[name] = position
        return output

    @cached_property
    def slices(self)->Dict[str, slice]:
        output = {}
        for name in self.channels:
            output[name] = slice(self.start_index[name], self.end_index[name])
        return output
    def __init__(self, spec: Dict[str, int]):
        self.spec = MappingProxyType(spec)

    def __contains__(self, item: str)->bool:
        return item in self.spec

    def __eq__(self, other):
        return self.spec == other.spec
