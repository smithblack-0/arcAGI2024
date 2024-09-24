from typing import List

import torch
from torch import nn

from src.main.model.core import LogicLayer


# Logic unit layer. This is used in many places throughout the main, and
# has high parameter efficiency


class LogicUnit:
    """
    The logic unit is intended, hopefully, to be applicable to a
    variety of scenarios by means of implementing different context fetching
    layers. It consists of a number of logic layers.
    """
    def __init__(self,
                 d_latents: int,
                 layers: List[LogicLayer],
                 ):
        self.layers = nn.ModuleList(layers)
        self.layernorms = nn.ModuleList([nn.LayerNorm(d_latents) for _ in range(len(layers))])
    def forward(self,
                latents: torch.Tensor
                )->torch.Tensor:
        for layer, layernorm in zip(self.layers, self.layernorms):
            latents = layernorm(latents + layer(latents))
        return latents

class DecodeUnit


class CoreProcessingUnit(nn.Module):
    """
    This is intended to be as close as we are going
    to get to a piece that can be transported between main
    architectures to do any logic-based job.

    It consists of a self attention process designed to work
    on latent embeddings in order to significantly speed up
    computation.
    """
    def __init__(self,

                 ):

class Encoder(nn.Module):

class Encoder:
    """
    The encoder mechanism.
    """
