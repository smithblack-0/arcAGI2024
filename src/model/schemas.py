import torch
from typing import List, Any, Dict
from dataclasses import dataclass
from .config import Config







@dataclass
class Header:
    """
    The header class contains the majority
    of header information in an unconverted
    format.
    """
    mode: str
    zone: str
    shape: List[int]
    sequence: int

class Schema:
    """
    A schema is a representation of critical factors
    regarding headers and payload tensors.
    """
    modes: List[str]
    zones: List[str]
    shapes: Dict[str, List[int]]
    def __init__(self,
                 modes: List[str],
                 zones: List[str],
                 shapes: Dict[str, List[int]]):
        for mode in modes:
            assert mode in shapes

        self.modes = modes
        self.zones = zones
        self.shapes = shapes

class SchemaReference:
    """
    A schema reference can be setup from a
    config, and will then proceed to provide
    utility functions mapping information
    from and to tensors.
    """
    def __init__(self, config: Config):
        self.schema = Schema(config.modes, config.zones, config.shapes)
class HeaderReference:
    """
    A small utility class that will let you know
    where each header is located within
    a passed header tensor. Or, let you figure
    out from the position what that header was.
    """
    def get_position(self, header: str) -> int:
        """ Returns the header index for a given header"""
        if header not in self.header_to_position_map:
            raise KeyError(f'"{header}" is not a valid header')
        return self.header_to_position_map[header]

    def get_header(self, position: int)->str:
        """Returns the header associated with a given position"""
        if position not in self.position_to_header_map:
            raise KeyError(f'"{position}" is not a valid position')
        return self.position_to_header_map[position]
    def __init__(self,
                 headers: List["str"]
                 ):
        self.header_to_position_map = {header : i for i, header in enumerate(headers)}
        self.position_to_header_map = {i : header for i, header in enumerate(headers)}

class ModeReference:
    """
    The mode reference. It can turn a mode string
    into an integer value, or
    """
    def __init__(self, config: Config):
        self.modes_int_map = {}


class SchemaBuilder:
    """
    The schema builder class is designed to greatly
    ease the construction of schemas by automatically
    handling construction of schemas and other
    important related functions.

    ---- interface  ----

    * manual definition:
        - User can add a schema to directly be incorporated, as long as they
          provide the shape and embedding mechanism.

    * user can add block shape and embedding mechanism
    * user can add zones they wish to see process
    """
    def __init__(self):
        self.schemas = []

