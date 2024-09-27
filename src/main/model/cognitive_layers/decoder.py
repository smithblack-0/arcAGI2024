import torch
from torch import nn
from typing import List, Optional, Tuple
from src.main.model.core import LogicLayer, MultiheadedAttention, Feedforward
from src.main.model.cognitive_layers import ContextFetcher

class DecoderLayer(nn.Module):
    """
    The decoder layer is responsible for turning a latent representation bound
    to a block into the contents of that block. It does this by a varient
    of the same encoding methodolody used to encode the blocks in the
    first place.
    """

    # Fetch context
    # self attention across the latents.
    # feedforward.

    def __init__(self,
                 cross_attn: MultiheadedAttention,
                 self_attn: MultiheadedAttention,
                 feedforward: Feedforward
                 ):

        self.cross_attn = cross_attn
        self.self_attn = self_attn
        self.feedforward = feedforward

    def forward(self,
                decodings: torch.Tensor,
                latents: torch.Tensor,
                )->torch.Tensor:
        """
        Runs the given decoder layer. The basic idea is that we predict
        the next token in the block sequence given the current token
        and using the latents as context.

        :param decodings: The current decodings
            - Shape (..., latent_sequence, block_sequence, d_input)
        :param latents:
            - The latents to decode.
            - Shape (..., latent_sequence, block_sequence, d_input) (during training)
            - Shape (..., latent_sequence, 1, d_input) (during generation)
            - The above shapes control whether operating in training or generative mode.
        :return:
            - The decoded embeddings. Behavior varies depending on if operating in generative mode or not.
              If operating with generative latents, we decode the last term in decodings, while if operating
              in training mode we decode and predict everything.
            - Shape (..., latent_sequence, 1, d_input) (during generation)
            - Shape (..., latent_sequence, block_sequence, d_input) (during training)
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
               num_decoder_sublayers: int,
               dtype: Optional[torch.dtype] = None,
               device: Optional[torch.device] = None
               ):



        self_fetcher = ContextFetcher.create(d_inputs,
                                                d_inputs,
                                                num_input_heads,
                                                num_decoder_layers,
                                                num_decoder_sublayers,
                                                dtype=dtype,
                                                device=device
                                                )

        latent_attn = MultiheadedAttention.create()
    def __init__(self,
                 decoder: LogicLayer,
                 ):
        super().__init__()
        assert decoder.causal, "decoder must be causal"
        self.decoder = decoder

    def forward(self,
                latent_encodings: List[torch.Tensor],
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


