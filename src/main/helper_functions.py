from typing import List, Callable, Any

import numpy as np
import torch
from torch import nn


## Positional Encoding mechanisms.
#
# This is needed to provide a means of properly encoding information for the
# various modes.

def sinusoidal_positional_encoding(seq_len, model_dim, gen_term=10000.0):
    """
    Computes sinusoidal positional encoding for a transformer.

    Parameters:
    seq_len (int): Length of the sequence.
    model_dim (int): Dimension of the main (d_model).
    gen_term (float): Generation term for frequency calculation (default: 10000.0).

    Returns:
    numpy.ndarray: Positional encoding matrix of shape (seq_len, model_dim).
    """
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, model_dim, 2) * -(np.log(gen_term) / model_dim))
    pos_enc = np.zeros((seq_len, model_dim))
    pos_enc[:, 0::2] = np.sin(position * div_term)
    pos_enc[:, 1::2] = np.cos(position * div_term)
    return pos_enc


def eval_legendre(order, x):
    """
    Evaluates the Legendre polynomial of a given order at points x.

    Parameters:
    order (int): The order of the Legendre polynomial.
    x (numpy.ndarray): The points at which to evaluate the polynomial.

    Returns:
    numpy.ndarray: The evaluated Legendre polynomial values.
    """
    if order == 0:
        return np.ones_like(x)
    elif order == 1:
        return x
    else:
        p0 = np.ones_like(x)
        p1 = x
        for n in range(2, order + 1):
            pn = ((2 * n - 1) * x * p1 - (n - 1) * p0) / n
            p0, p1 = p1, pn
        return pn


def pope_positional_encoding(seq_len, model_dim, gen_term=10):
    """
    Computes PoPE positional encoding using Legendre polynomials.

    Parameters:
    seq_len (int): Length of the sequence.
    model_dim (int): Dimension of the main (d_model).
    order (int): Order of the Legendre polynomial (default: 10).

    Returns:
    numpy.ndarray: PoPE encoding matrix of shape (seq_len, model_dim).
    """
    position = np.linspace(-1, 1, seq_len)
    pos_enc = np.zeros((seq_len, model_dim))
    for i in range(model_dim):
        pos_enc[:, i] = eval_legendre(gen_term, position) * ((i + 1) / model_dim)
    return pos_enc


class PositionalEncodings(nn.Module):
    """
    Produces N-D positional encodings to encode both position and size of input
    space into a single encoding. Precomputes the encodings.
    """

    def __init__(self,
                 embedding_dim: int,
                 max_dim_sizes: int | List[int],
                 gen_function: Callable[[int, int, Any], np.ndarray],
                 gen_term: Any,
                 ):
        """
        Initialize and precompute
        :param embedding_dim: The dimension of the embeddings to make
        :param max_dim_sizes: A list indicating the maximum sizes the dimensions are going to see
        :param gen_function: The function to generate the encodings
        :param gen_term: The term conditioning encoding generation
        """

        super().__init__()
        if isinstance(max_dim_sizes, int):
            max_dim_sizes = [max_dim_sizes]

        # validate
        total_split_factor = 2 * len(max_dim_sizes)
        assert embedding_dim % total_split_factor == 0, f"Embedding dim was not divisible by {total_split_factor}"

        # Store
        self.num_dims = len(max_dim_sizes)
        self.embedding_dim = embedding_dim
        self.intermediate_dim = embedding_dim // total_split_factor
        self.max_dim_sizes = max_dim_sizes
        self.gen_func = gen_function
        self.gen_term = gen_term

        # Pregenerate grids.
        indices = [torch.arange(length) for length in max_dim_sizes]
        index_vectors = torch.meshgrid(*indices, indexing="ij")
        dimension_precomputations = [gen_function(length, self.intermediate_dim, gen_term) for length in max_dim_sizes]
        precomputed_encodings = [precomputed[index] for precomputed, index in
                                 zip(dimension_precomputations, index_vectors)]
        precomputed_encodings = torch.concat([torch.tensor(precomputed) for precomputed in precomputed_encodings],
                                             dim=-1)
        self.precomputed_encodings = precomputed_encodings

    @staticmethod
    def compute_shape_size(mask: torch.Tensor) -> torch.Tensor:
        """
        Computes the maximum size of each dimension per batch.
        :param mask: The mask we were passed in. Shape (batch ...)
        :return: The sizes tensor. Shape (batch x num_dims).
        """

        # We figure this out by assuming that the mask will consist of 1's right up to when we
        # need to stop including new elements, at which point it will become zero. This means summing
        # up along a given dimension will cause each element of the mask to display the number of
        # elements on that dimension.
        sums_per_dimension = [mask.sum(dim=dim, keepdim=True) for dim in range(1, len(mask.shape))]
        sums_per_dimension = [item.flatten(1, -1) for item in sums_per_dimension]
        dim_sizes = torch.stack([item.max(dim=-1)[0].int() for item in sums_per_dimension], dim=-1)
        return dim_sizes

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward sweep.

        :param mask: A floating mask of shape (batch x ...). Note that ... must match the number of dimensions.
        :return: The computed and combined positional encodings.
        """

        assert mask.dim() - 1 == self.num_dims, f"The number of nonmask batch dimensions and positional dimensions were different"

        # Start the positional encoding by slicing and expanding

        slicer = tuple([slice(dim) for dim in mask.shape[1:]] + [slice(None)])
        pos_encoding = self.precomputed_encodings[slicer]  # Shape (..., L)
        pos_encoding = pos_encoding.unsqueeze(0).expand(mask.shape[0],
                                                        *[-1] * pos_encoding.dim())  # Shape (batch x ... x L)

        # compute the size encoding.
        sizes = self.compute_shape_size(mask)  # batch x D
        size_encoding = []
        for index in sizes.unbind(0):
            slicer = tuple([item for item in index] + [slice(None)])
            size_encoding.append(self.precomputed_encodings[slicer])
        size_encoding = torch.stack(size_encoding, dim=0)

        while size_encoding.dim() < pos_encoding.dim():
            size_encoding = size_encoding.unsqueeze(1)
        size_encoding = size_encoding.expand(-1, *pos_encoding.shape[1:-1], -1)

        # Combine and return
        pos_encoding = torch.cat([pos_encoding, size_encoding], dim=-1)
        pos_encoding = pos_encoding * mask.unsqueeze(-1)
        return pos_encoding

