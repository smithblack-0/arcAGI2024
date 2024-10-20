import torch
from torch import nn


class LatentEmbedding(nn.Module):
    """
    Converts a tensor of embedding inputs into a collection
    of latent representations. Adds an extra dimension in the
    appropriate place to ensure attention occurs on the efficient
    dimension.
    """
    def __init__(self,
                 input_dim: int,
                 latent_dim: int,
                 num_latents: int
                 ):

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_latents = num_latents
        self.projector = nn.Linear(input_dim, num_latents*latent_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Expands a tensor of embedded inputs into a tensor of latent
        representations. Adds an extra dimension for these representations

        :param inputs: The inputs
            - Shape (..., items, input_dim)
            - Should be float containing embeddings
        :return: latent representation
            - Shape (..., items, num_latents, latent_dim)
            - produces by projection
        """
        reshape = list(inputs.shape[:-1]) + [self.num_latents, self.latent_dim]
        latent_embeddings = self.projector(inputs)
        latent_embeddings = latent_embeddings.view(reshape)
        return latent_embeddings
