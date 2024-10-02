from typing import Optional, Tuple

import torch
from torch import nn

class MultiheadedAttention(nn.Module):
    """
    An adapter layer. Allows us to use a torch multiheaded attention layer
    with batches that may have multiple batch elements. In particular, promises
    to correctly handle anything with shape

    - (..., sequence, embed)

    Rather than just:
    - (batch, sequence, embed)
    """
    @classmethod
    def create(cls,
               embed_dim: int,
               num_heads: int,
               dropout: float,
               bias: bool = True,
               add_bias_kv: bool = False,
               add_zero_attn: bool = False,
               kdim: Optional[int] = None,
               vdim: Optional[int] = None,
               device: Optional[torch.device] = None,
               dtype: Optional[torch.dtype] = None,
               ):
        """
        Creates and wraps up a multiheaded attention adapter, capable
        of processing batched data. All parameters have the same effects
        as their torch equivalent

        :param embed_dim: Total dimensions of the model
        :param num_heads: Number of heads
        :param dropout: dropout probability
        :param bias: If specified, adds bias to projection layers
        :param add_bias_kv: If specified, add bias to kv projection layers
        :param kdim: The dimensionalities of the keys
        :param vdim: The dimensionalities of the values
        :return: A multibatch multiheaded attention layer
        """
        attn_layer = nn.MultiheadAttention(embed_dim, num_heads,
                                           dropout=dropout,
                                           bias = bias,
                                           add_bias_kv=add_bias_kv,
                                           kdim=kdim,
                                           vdim=vdim,
                                           batch_first=True,
                                           device=device,
                                           dtype = dtype
                                           )
        return cls(attn_layer)
    def __init__(self,
                 attention_layer: nn.MultiheadAttention,
                 ):
        super().__init__()
        self.attention_layer = attention_layer

    def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            need_weights: bool = True,
            attn_mask: Optional[torch.Tensor] = None,
            average_attn_weights: bool = True,
            is_causal: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        See torch.MultiheadedAttention. Changes and inputs are listed here. However, do
        look at attn mask, and notice the significant changes.

        :param query: The queries. May be (..., sequence, embedding).
        :param key: The keys. Same as torch, except may be shape (..., sequences, embeddings)
        :param value: The values. Same as torch, except may be shape (..., sequences, embeddings)
        :param key_padding_mask: Not supported.
        :param need_weights: Same as in torch
        :param attn_mask:
            - !!Significant changes!!
            - You can get torchlike behavior by providing (query, content) mask, in which case it acts
              exactly like in torch.
            - You can provide something in terms of (heads, query, content) in which case you can also
              activate or control heads.
            - You can provide something in terms of (..., heads, query, content) to have a per batch
              mask. If you do not care about addressing heads, you can also pass something as
              (..., 1, query, content)
        :param is_causal: As normal
        :return: Either a tensor or a tuple of tensors. See torch definition.
        """



        # Get the batch shape associated with query and content dimensions
        restoration_shape = list(query.shape[:-2])

        # Flatten primary content. Go from (..., sequence, embeddings) to (flat_batch, sequence, embedding)

        query = query.flatten(0, -3)
        key = key.flatten(0, -3)
        value = value.flatten(0, -3)

        # Flatten the attention mask, if needed.
        if attn_mask is not None and attn_mask.dim() > 2:
            if attn_mask.shape[-3] == 1:
                # Handles expanding in case we are broadcasting the heads.
                # This saves the user from having to do it themselves.
                repeats = [-1]*attn_mask.dim()
                repeats[-3] = self.attention_layer.num_heads
                attn_mask = attn_mask.expand(repeats)

            # Flatten down to (N*heads, query, content)
            attn_mask = attn_mask.flatten(0, -3)

        # Transpose if needed for batch first
        if not self.attention_layer.batch_first:
            query = query.movedim(0, 1)
            key = key.movedim(0, 1)
            value = value.movedim(0, 1)

        # Run attention

        results = self.attention_layer(query,
                                       key,
                                       value,
                                       None,
                                       need_weights,
                                       attn_mask,
                                       average_attn_weights,
                                       is_causal
                                       )


        # Take the results apart, and restore batch behavior
        if need_weights:
            attn_results, attn_weights = results
        else:
            attn_results = results
            attn_weights = None

        # Standardize results to exist in terms of (batch, sequence, embeddings)
        if not self.attention_layer.batch_first:
            attn_results = attn_results.movedim(1, 0)

        # Restore proper shape
        attn_result_shape = restoration_shape + list(attn_results.shape[-2:])
        attn_results = attn_results.view(attn_result_shape)

        if attn_weights is not None:
            if average_attn_weights:
                restore_shape = restoration_shape + list(attn_weights.shape[-3:])
            else:
                restore_shape = restoration_shape + list(attn_weights.shape[-2:])
            attn_weights = attn_weights.view(restore_shape)

        # Finally, return results
        return attn_results, attn_weights

class MultiHeadedAttention(nn.Module):
    """
    Implements a specialized kind of multiheaded attention that is
    capable of accepting multidimensional batches, and that is
    capable of
    """
