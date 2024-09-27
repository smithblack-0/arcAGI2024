import torch
import itertools
import functools
import math
from torch import nn
from typing import List, Any, Optional
from src.main.model.core.transfomer_layers import TransformerEncoderLayer, TransformerDecoderLayer
class CycleLayer(nn.Module):
    """
    The cycle layer is designed to allow an arbitrary number of computation layers
    to be formed by reusing parmaeters. Layers can be stored in it, alongside a
    "virtual layer" size, and this will result in those layers being returned in a cycle.
    For instance, if one provides 5 linear layers, and a virtual layer size of 2, using
    gen_layers will get someting that runs the problem through layers [1, 2], then [3, 4],
    then [5, 1], then [2,3], etc.
    """
    def __init__(self,
                 virtual_layer_depth: int,
                 layers: List[nn.Module],
                 layernorms: List[nn.LayerNorm],
                 final_layernorm: nn.LayerNorm,
                 dropout: float
                 ):

        self.virtual_layer_depth = virtual_layer_depth
        self.cycle_length = math.lcm(virtual_layer_depth, len(layers))
        self.layers = nn.ModuleList(layers)
        self.layernorms = nn.ModuleList(layernorms)
        self.layernorm_final = final_layernorm
        self.dropout = nn.Dropout(dropout)


    def virtual_layer_generator(self):
        cycle_iterator = itertools.cycle(range(len(self.layers)))
        while True:
            next_cycle = [next(cycle_iterator) for _ in range(self.virtual_layer_depth)]
            yield functools.partial(self.forward, layers=next_cycle)

    def forward(self, layers: List[int], tensor, *args, **kwargs)->Any:
        skip_connection = tensor
        for index in layers:
            layer = self.layers[index]
            layernorm = self.layernorms[index]
            update = self.dropout(layer(tensor, *args, **kwargs))

            if index != self.virtual_layer_depth - 1:
                # Standard layernorm
                tensor = layernorm(tensor + update)
            else:
                # Cycle end, and multiskip residual.
                tensor = layernorm(tensor + skip_connection + update)

        return tensor



class LogicCore(CycleLayer):
    """
    A logic core consists of a number of transformer encoder
    layers all bound together.
    """

    @classmethod
    def create(cls,
               query_dim: int,
               num_heads: int,
               num_concrete_layers: int,
               virtual_layer_depth: int,
               dim_feedforward: int,
               dropout: float,
               dtype: Optional[torch.dtype] = None,
               device: Optional[torch.device] = None
               ):

        layers = [TransformerEncoderLayer(query_dim,
                                          num_heads,
                                          dim_feedforward,
                                          dropout,
                                          dtype=dtype,
                                          device=device)
                  for _ in range(num_concrete_layers)
                  ]
        layernorms = [nn.LayerNorm(query_dim, dtype=dtype, device=device) for _ in layers]
        cls(virtual_layer_depth, layers, layernorms, dropout)

class ContextCore(CycleLayer):
    """
    Creates context fetching layers
    """
    @classmethod
    def create(cls,
               query_dim: int,
               source_dim: int,
               num_heads: int,
               num_concrete_layers: int,
               virtual_layer_depth: int,
               dim_feedforward: int,
               dropout: float,
               dtype: Optional[torch.dtype] = None,
               device: Optional[torch.device] = None
               ):

        layers = [TransformerDecoderLayer(query_dim,
                                          source_dim,
                                          num_heads,
                                          dim_feedforward,
                                          dropout,
                                          dtype=dtype,
                                          device=device)
                  for _ in range(num_concrete_layers)
                  ]
        layernorms = [nn.LayerNorm(query_dim, dtype=dtype, device=device) for _ in layers]
        cls(virtual_layer_depth, layers, layernorms, dropout)
