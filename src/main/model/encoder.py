import torch
from torch import nn
from typing import List
from .core import TransformerEncoderLayer, TransformerDecoderLayer
from src.main.CBTensors.channel_bound_tensors import CBTensor


class Encoder:
    """
    The primary encoder for the modeling process. Processes an input
    block into something that can be decoded as an output block. Supports
    parameter sharing, computation length specification, and more. Also
    supports zone restrictions and assignments.
    """

    def __init__(self,
                 num_virtual_layers: int,
                 embedding: nn.Module,
                 latent_embedding: nn.Module,
                 zones_restrictions: torch.Tensor,
                 context_fetch_layers: List[TransformerDecoderLayer],
                 logic_layers: List[TransformerEncoderLayer],
                 ):

        self.num_virtual_layers = num_virtual_layers

        self.embedding = embedding
        self.latent_embedding = latent_embedding
        self.context_layers = nn.ModuleList(context_fetch_layers)
        self.logic_layers = nn.ModuleList(logic_layers)

    def forward(self,
                tensor: CBTensor,
                block_pos: torch.Tensor,
                )->torch.Tensor:

        #





