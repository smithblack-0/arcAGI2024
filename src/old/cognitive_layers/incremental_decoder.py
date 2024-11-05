import torch

from torch import nn
from typing import List, Optional
from src.old.core import MultiheadedAttention, Feedforward

class IncrementalDecoder:
    """
    An incremental output decoder.

    Output in this arcAGI2024 is incremental, which means adding more computation
    core computation steps will generally increase the performance of the arcAGI2024.
    This increase of performance is also linear. What this layer attempts to do
    is provide the next incremental output. These can later be combined together
    to provide final embeddings for making predictions.

    It uses some tricks to make it easy to encode in a graph, such as using
    lists rather than concatenating tensors. It is designed to run multiple
    times and the results to be combined together, sort of like tree boosting.

    It is called an "adapter" because you could take the same computation core,
    swap in a different version of this and some other features, and be compatible with a completely
    different pipeline architecture.
    """

    @classmethod
    def create(cls,
               d_latents: int,
               d_inputs: int,
               d_feedforward: int,
               num_input_heads: int,
               dropout: float,
               device: Optional[torch.device] = None,
               dtype: Optional[torch.dtype] = None,
               )->'IncrementalOutputDecoder':
        """
        Creates a working incremental output decoder bound
        to the scenario.

        :param d_latents: The latent dimension
        :param d_inputs: The input dimension
        :param d_feedforward: The feedforward width. Usually wider than d_inputs
        :param num_input_heads: Number of heads. Must be compatible with d_inputs
        :param dropout: The dropout rate
        :param device: Torch device
        :param dtype: Torch dtype
        :return: A Incremental output adapter.
        """

        latent_attention = MultiheadedAttention.create(d_inputs,
                                                       num_input_heads,
                                                       dropout,
                                                       kdim=d_latents,
                                                       vdim=d_latents,
                                                       device=device,
                                                       dtype=dtype
                                                       )
        incremental_attention = MultiheadedAttention.create(d_inputs,
                                                            num_input_heads,
                                                            dropout,
                                                            device=device,
                                                            dtype=dtype)
        feedfoward = Feedforward(d_inputs, d_feedforward, dropout, device=device, dtype=dtype)
        return cls(d_inputs, latent_attention, incremental_attention, feedfoward)

    def __init__(self,
                 d_inputs: int,
                 latent_attention: MultiheadedAttention,
                 incremental_attention: MultiheadedAttention,
                 feedforward: MultiheadedAttention
                 ):

        self.incremental_layernorm = nn.LayerNorm(d_inputs)
        self.feedforward_layernorm = nn.LayerNorm(d_inputs)

        self.latent_attention = latent_attention
        self.incremental_attention = incremental_attention
        self.feedforward = feedforward

    def forward(self,
                latents: torch.Tensor,
                block_bindings: torch.Tensor,
                incremental_output: List[torch.Tensor],
                ):
        """
        The forward mechanism. Runs and returns another incremental
        output. Your intuition for those should be that as the number
        of incremental output increase.

        For performance memory reasons, it is better to pass and then
        stack incrementals as part of a list.

        :param inputs:
            - The original embedded inputs, before latent expansion
            - Shape had better be (..., sequence, d_input)
        :param latents:
            - The current latent expression.
            - Shape had better be (..., sequence, num_latents, d_latents)
        :param incremental_decodings:
            - The incremental decodings that have been produced so far
            - Each one had better have shape (..., sequence, d_input)
        :return: The next incremental decoding.
            - Has shape (..., sequence, d_input)
        """
        incremental_tensors = torch.stack(incremental_output, dim=-2) #(..., sequence, incremental, d_input)
        inputs = inputs.unsqueeze(-2) # (..., sequence, 1, d_input)

        output = self.latent_attention(inputs, latents, latents)
        output = output.squeeze(-2)

        temp = self.incremental_attention(output, incremental_tensors, incremental_tensors)
        output = self.incremental_layernorm(output + temp)

        temp = self.feedforward(output)
        output = self.feedforward_layernorm(output + temp)

        return output

