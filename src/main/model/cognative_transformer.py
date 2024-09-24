import torch
from torch import nn
from typing import Optional

from .cognitive_layers import LogicCore, LatentEmbedding, ContextFetcher, IncrementalDecoder, FinalDecoder

class CognitiveCore:
    """
    The cognitive core contains three things that are invoked multiple
    times in a row.
    """
    def __init__(self,
                 fetcher: ContextFetcher,
                 logic_core: LogicCore,
                 incremental_decoder: IncrementalDecoder,
                 use_checkpointing: bool = False,
                 ):

        self.context_fetcher = fetcher
        self.logical_core = logic_core
        self.incremental_decoder = incremental_decoder
        self.use_checkpointing = use_checkpointing

    def forward(self,
                inputs: torch.Tensor,
                latents: torch.Tensor,
                ):

class CognativeTransformer: