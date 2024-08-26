"""
Concerns the core transformer model we will be working with,
including builders to make a fresh one.

We will be using a transformer that operates with block multimodal
encoding, and that can take a conventional transformer as input.
"""

import torch
import numpy as np
from abc import ABC, abstractmethod
from torch import nn
from typing import List, Optional

from src.model.schema import SchemaRegistry


## Schema tracker registry


## Data
#
# Data will consist of grids of integers, with, for example, the ability to specify one of
# 256 colors for each channel, or the tokens.
#
# Image data???
# Encoding: SURE
# Decoding: Uh.... data leak?
# Make it entirly pretend to be operating in an encoding mode?
# WTH would the training task look like?x`

class ModeEmbeddingAdapter(nn.Module):
    """
    A class for containing mode-specific embedding logic.
    """
    def __init__(self,
                 schema: str,
                 embedder: nn.Module,
                 encoding_dim: int,
                 patch_shape: Optional[List[int]] = None,
                 ):
        """

        :param schema: The schema to associate this with
        :param embedder: The embedding mechanism. Should handle ND data as appropriate
        :param patch_shape: One can optionally patch sections as in ViT.
        :param encoding_dim: Embeddings must come out of the adapter with this shape
        """

        super().__init__()

        self.schema = schema
        self.embeddings = embedder
        self.adapter = nn.LazyLinear(encoding_dim)

class ModeAdapter(nn.Module, ABC):
    """
    A mode-specific adapter, designed to convert mode-specific
    data into it's more general format.

    We assume in this that
    """
    def __init__(self,
                 schema: str,
                 schema_registry: SchemaRegistry,
                 embedder: nn.Module,
                 decoder: nn.Module,

                 ):
        """
        The setup process for the mode spec

        :param schema: The schema to select for association
        :param schema_registry: The schema registry to draw from
        :param patch_shape: The shape of each patch, as in ViT.
        :param embeddings: The final embedding dim.
        """

        # General behavior and validation
        self.schema = schema
        self.schema_registry = schema_registry

        # Encoding requirements
        self.linear = nn.Linear(embedded_dim*np.prod(patch_shape), encoding_dim)

    @abstractmethod
    def embed(self, int_grid: torch.Tensor)->torch.Tensor:
        """
        This should be capable of embedding an arbitrary
        :param int_grid:
        :return:
        """
    def encode(self, int_grid: torch.Tensor) -> torch.Tensor:


class DataConverter(nn.Module):
    """
    - Embedding and patching control.
    - Associated with a schema - no schema, no converter can be registere.
    - Encode
    - Decode.
    """




class ConverterRegistry(nn.Module):
    """
    A registry system for registering the mechanisms needed

    """
    def __init__(self, logit_slots: int):
        super().__init__()

        self.logit_slots = logit_slots
        self.encoder_registry = nn.ModuleDict()
        self.decoder_registry = nn.ModuleDict()










