"""
Cognition cores are long and fast mechanisms designed
to repetitively attend and consider the incoming
latent embeddings. It is the logical heart
of the model.
"""

import torch
from torch import nn
from typing import List, Optional

from src.old.core import LogicLayer
class LogicCore:
    """
    The logic core is a collection
    of a bunch of logic layers, dedicated
    to general logical and computation understanding
    """
    @classmethod
    def create(cls,
               d_latents: int,
               d_feedforward: int,
               num_latent_heads: int,
               num_core_layers: int,
               num_sublayers: int,
               dropout: float,
               dtype: Optional[torch.dtype] = None,
               device: Optional[torch.device] = None
               )->'CognitionCore':
        """
        Creates a cognition core according to the provided
        specification

        :param d_latents: The latent embedding dimensions
        :param d_feedforward: The width of the feedforward actions.
        :param num_latent_heads: The number of transformer heads
        :param num_core_layers: The number of logical layers to make in the core
        :param num_sublayers: The number of sublayers per logical layer. This parameter shares.
        :param dropout: The droput rates
        :param dtype: The device
        :param device: The Dtype
        :return: A cognition core instance
        """

        layers = [LogicLayer.create(d_latents, d_feedforward, num_latent_heads, num_sublayers, dropout,
                                    device=device, dtype=dtype) for _ in range(num_core_layers)]
        return cls(d_latents, layers)
    def __init__(self,
                 d_latents: int,
                 layers: List[LogicLayer],
                 ):

        self.layers = nn.ModuleList(layers)
        self.layernorms = nn.ModuleList([nn.LayerNorm(d_latents) for _ in layers])

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Performs forward pass of the logical layers
        :param latents: The latent representation
            - Shape (..., sequence, d_latents)
        :return: The new latent representation
            - Shape (..., sequence, d_latents)
        """
        for layer, layernorm in zip(self.layers, self.layernorms):
            latents = layernorm(latents + layer(latents))
        return latents