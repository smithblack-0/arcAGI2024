from typing import Optional

import torch
from torch import nn
from src.main.model.core.feedforward import Feedforward
from src.main.model.core.multiheaded_attention_adapter import MultiheadedAttention

class LogicLayer(nn.Module):
    """
    Used to think about and work on a latent embedding.
    As always, we try to maintain parameter efficiency
    by reusing parameters. There is a cycle length
    associated with this which determines how many self-attn
    feedforward cycles to apply in the layer. It will end up
    used in many other layers
    """
    @classmethod
    def create(cls,
               d_latents: int,
               d_feedforward_hidden: int,
               num_heads: int,
               num_repeats: int,
               dropout: float,
               causal: bool = False,
               device: Optional[torch.device] = None,
               dtype: Optional[torch.dtype] = None,
               )->'LogicLayer':
        """
        Creates a logic layer by creating any needed sublayers.

        :param d_latents: The latent width
        :param num_heads: The number of heads
        :param num_repeats: The number of stages
        :param dropout: The dropout rate
        :return: A logic layer
        """

        self_attn = MultiheadedAttention.create(d_latents, num_heads, dropout, device=device, dtype=dtype)
        feedforward = Feedforward.create(d_latents,d_feedforward_hidden, dropout=dropout, device=device, dtype=dtype)
        return cls(d_latents, num_repeats, self_attn, feedforward, causal)


    def __init__(self,
                 d_latents: int,
                 num_repeats: int,
                 self_attn: nn.MultiheadAttention,
                 feedforward: nn.Module,
                 causal: bool = False
                 ):
        self.d_latents = d_latents
        self.num_repeats = num_repeats

        self.self_attn_layernorms = nn.ModuleList([nn.LayerNorm(d_latents) for _ in range(num_repeats)])
        self.feedforward_layernorms = nn.ModuleList([nn.LayerNorm(d_latents) for _ in range(num_repeats)])

        self.self_attn = self_attn
        self.feedforward = feedforward

        self.causal = causal

    def forward(self,
                latents: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                ) -> torch.Tensor:
        """
        Processes the latents producing further insight into the computation.
        :param latents: The latents to process.
            - Shape (..., items, latents, embeddings)
            - Usually shape (batch, items, latents, embedding).
        :return:
            - Latents
            - Shape (..., items, latents, embeddings)
        """

        for self_attn_norm, feedforward_norm in zip(self.self_attn_layernorms, self.feedforward_layernorms):
            latents, _ = self_attn_norm(latents + self.self_attn(latents, latents, latents, attn_mask=mask,
                                                                 casual=self.causal))
            latents = feedforward_norm(latents + self.feedforward(latents))
        return latents
