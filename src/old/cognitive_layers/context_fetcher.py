
import torch
from torch import nn
from typing import Optional

from src.old.core import Feedforward, LogicLayer, MultiheadedAttention


class CrossFetcherLayer(nn.Module):
    """
    Based on a provided latent embeddings collection, the
    context fetcher interfaces with the original collection
    of context in order to gather additional context out of it.

    High parameter efficiency is emphasized, with the idea of
    using checkpointing and reusing existing code. The fetcher
    is intended to go off all in one go, is not intended to
    be too deep. It is also intended to specialize in information
    retrieval over primary processing.
    """

    @classmethod
    def create(cls,
               d_latents: int,
               d_inputs: int,
               num_latent_heads: int,
               num_sublayers: int,
               dropout: float = 0.1,
               device: Optional[torch.device] = None,
               dtype: Optional[torch.dtype] = None
               ):

        final_feedforward_hidden = 4*d_latents
        petite_feedforward_hidden = 2*d_latents
        feedforward = Feedforward(d_latents, final_feedforward_hidden, dropout=dropout, device=device, dtype=dtype)
        cross_attention = MultiheadedAttention.create(d_latents, num_latent_heads, dropout,
                                                      kdim=d_inputs, vdim=d_inputs,
                                                      device=device, dtype=dtype
                                                      )
        core_logic = LogicLayer.create(d_latents,petite_feedforward_hidden, num_latent_heads, num_sublayers,
                                       dropout, device=device, dtype=dtype)

        return cls(d_latents, core_logic, cross_attention, feedforward)

    def __init__(self,
                 # Define parameter reuse abilities
                 d_latents: int,

                 # Define core layers
                 logic_encoder: LogicLayer,
                 cross_attn: MultiheadedAttention,
                 feedforward: Feedforward
                 ):

        super().__init__()

        self.d_latents = d_latents

        self.self_attn_layernorm = nn.LayerNorm(d_latents)
        self.

        self.logic_encoder = logic_encoder
        self.cross_attn = cross_attn
        self.feedforward = feedforward
        self.layernorms = nn.LayerNorm(d_latents)

    def forward(self,
                inputs: torch.Tensor,
                latents: torch.Tensor,
                casual_location: Optional[torch.Tensor] = None,
                intake_mask: Optional[torch.Tensor] = None)->torch.Tensor:
        """
        Performs encoding of the data, using parameter extension
        technologies as is required.

        inputs:
            - The input representation
            - Usually shape (batch, items, embeddings)
            - Possibly (..., input_items, embeddings)
            - Impractical to attend as items is quite large.
        latents:
            - The latent representation
            - The data in the latent representation.
            - Shape (..., latent_sequence, num_latents, embeddings)
            - Usually shape (batch, latent_sequence, num_latents, embedding).
            - This can self attend
        mask:
            - Attention mask.
            - Indicates what key, values cannot be attended to and where when true.
            - Casual encoding is enforced here, and zone enforcement can also be made.
        """
        latents = self.planning_layernorm(latents + self.logic_encoder(latents))
        latents = self.fetching_layernorm(latents + self.cross_attn(latents, inputs, inputs, attn_mask=intake_mask))
        latents = self.feedforward_layernorm(latents + self.feedforward(latents))

        for layernorm in self.layernorms:
            latents = layernorm(latents + original_latents + self.plan_retrieval(latents))
            latents = layernorm(latents + self.cross_attn(latents, inputs, inputs, attn_mask=intake_mask))
            latents = layernorm(latents + self.feedforward(latents))
        return latents


