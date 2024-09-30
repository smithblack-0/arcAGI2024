import torch
from torch import nn
from .cycle_layer import ContextCore, LogicCore


class DecoderCore(nn.Module):
    """
    The code decoder class. Used to decode a particular
    instance.
    """

    def create(self,


               ):

    def __init__(self,
                 context_core: ContextCore,
                 logic_core: LogicCore,
                 ):

        super().__init__()

        self.context_core = context_core
        self.logic_core = logic_core

    def forward(self,
                latents: torch.Tensor,
                tensor: torch.Tensor,
                latents_padding: torch.Tensor,
                context_mask: torch.Tensor,
                computation_steps: int
                ):
        """
        Performs a decoder step. Synthesizes and uses virtual layers.

        :param latents: The source embedding tensor
            - The latent embedding tensor.
            - Is what we are actually trying to decode
            - Shape (batch, latent_sequence, num_latents, d_latents) (when training)
            - Shape (batch, 1, num_latents, d_latents) (when generating)
        :param tensor: The tensor we are decoding.
            - Presumably, the latents were created out of elements of this tensor
            - Shape (batch, input_sequence, d_inputs)
        :param mask:
            - A mask, defined in terms of the latent sequence and sequence, that
              lets us know what latent collections can access information about
              what sequence
            - Shape (batch, latent_sequence, input_sequence)
            - True means exclude, false means include.
        :return:
        """


        logic_cycle = self.logic_core.virtual_layer_generator()
        context_fetch_cycle = self.context_core.virtual_layer_generator()
        iterator = zip(range(computation_steps), logic_cycle, context_fetch_cycle)
        outputs = []
        for index, logic_layer, context_layer in iterator:
            skip_latents = latents
            latents = latents + context_fetch_cycle(latents, tensor, attn_mask=context_mask)
            latents = latents + skip_latents + logic_layer(latents, key_padding = latents_padding)
            outputs.append(latents)
        return torch.stack(outputs, dim=-2)



class DecoderTrainer(nn.Module):
    """
    The decoder class. Embedding, finishing, and the decoder core included.
    """
    def __init__(self,
                 encoding: nn.Module,
                 core: DecoderCore,
                 ):

    def forward(self,
                padding_mask: torch.Tensor,
                context: torch.Tensor,
                sequence: torch.Tensor,
                ):


class DecoderGenerator(nn.Module):
    """
    The generator class.
    """
