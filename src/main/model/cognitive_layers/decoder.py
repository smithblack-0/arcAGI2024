import torch
from torch import nn
from typing import List, Optional, Tuple
from src.main.model.core import LogicLayer

class DecoderLayer(nn.Module):
    """
    A single decoder layer, this starts from the current
    query, accesses the latents casually, acces
    """

    def forward(self,
                decodings: torch.Tensor,
                mode: str,
                incremental_latents: torch.Tensor,
                ):
        """
        Runs the given decoder layer. The basic idea is that we predict
        the next token in the block sequence given the current token
        and using the latents as context.

        :param decodings: The current decodings
            - Shape (..., latent_sequence, block_sequence, d_input)
            - Shape (..., latent_sequence, 1, d_input) (while decoding)
        :param incremental_latents:
            - Shape (..., latent_sequence, num_latents, d_input)
        :return:
        """


class BlockDecoder(nn.Module):
    """
    The block decoder.

    The block decoder consumes a collection of incremental
    latent embeddings that are the presumable result of
    each cognition cycle. It then uses these latents, which are usually
    associated with blocks, to decode said blocks.
    """
    @classmethod
    def create(cls,
               d_inputs: int,
               d_feedforward: int,
               num_input_heads: int,
               num_decoder_layers: int,
               dtype: Optional[torch.dtype] = None,
               device: Optional[torch.device] = None
               ):

        logic_layer = LogicLayer.create(
            d_inputs,
            d_feedforward,
            num_input_heads,
            num_decoder_layers,
            causal=True,
            dtype= dtype,
            device=device
        )
        return cls(logic_layer)
    def __init__(self,
                 decoder: LogicLayer,
                 ):
        super().__init__()
        assert decoder.causal, "decoder must be causal"
        self.decoder = decoder

    def forward(self,
                seed: torch.Tensor,
                incremental_latents: List[torch.Tensor],
                targets: Optional[torch.Tensor],
                )->torch.Tensor:
        """
        Performs the final causal attention process, producing the predictions for
        each computation section.

        :param incremental_latents: A list of the incremental latent states
            - Each latent had better be shape (..., latent_sequence, num_latents, d_input)
        :return: The output predictions
            - Shape is (..., sequence, computation_step, d_input)
            - All computation steps end up expressed
        """


        incremental_decodings = torch.stack(incremental_decodings, dim=-2) #(..., sequence, compute_steps, d_input)
        return self.decode(incremental_decodings)


