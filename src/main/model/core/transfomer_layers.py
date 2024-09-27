import torch
from torch import nn
from typing import Optional
from .feedforward import Feedforward
from .multiheaded_attention_adapter import MultiheadedAttention\

class TransformerEncoderLayer(nn.Module):
    """
    Performs transformer encoding, while supporting
    multidimensional batching.
    """
    def __init__(self,
                 query_dim: int,
                 num_heads: int,
                 dim_feedforward: int,
                 dropout: float,
                 casual: bool = False,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None
                 ):

        self.casual = casual

        # Dropout (shared)
        self.dropout = nn.Dropout(dropout)

        # Self attention implementation
        self.sa_layernorm = nn.LayerNorm(query_dim)
        self.self_attn = MultiheadedAttention.create(query_dim,
                                                     num_heads,
                                                     dropout,
                                                     dtype=dtype,
                                                     device=device)
        # Feedforward implementation


        self.ff_layernorm = nn.LayerNorm(query_dim)
        self.feedforward = Feedforward(query_dim,
                                       dim_feedforward,
                                       dropout=dropout,
                                       device=device,
                                       dtype=dtype)

    def forward(self,
                tensor: torch.Tensor,
                self_mask: Optional[torch.Tensor] = None,
                ) -> torch.Tensor:
        """
        Performs a transformer based attention step

        :param tensor:
            - The data to process
            - Shape (..., sequence, query_dim)
        :param self_mask:
            - An optional mask that corrolates with the self attention step.
            - Core shape is (query_sequence, query_sequence). Additional heads and batch feature alloed.
              batch content allowed.
            - First dimension is target, second is source.
            - See MultiheadedAttention for more details
        :return:
            - The results of performing the transformer process
            - These are unnormalized.
        """
        tensor = self.sa_layernorm(tensor)
        self_attn, _ = self.self_attn(tensor, tensor, tensor, attn_mask=self_mask, casual=self.casual)
        tensor = tensor + self.dropout(self_attn)


        tensor = self.ff_layernorm(tensor)
        feedforward = self.feedforward(tensor)
        tensor = tensor + self.dropout(feedforward)
        return tensor



class TransformerDecoderLayer(nn.Module):
    """
    A transformer decoder layer, capable of being
    """
    def __init__(self,
                 query_dim: int,
                 source_dim: int,
                 num_heads: int,
                 dim_feedforward: int,
                 dropout: float,
                 casual: bool = False,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None
                 ):
        super().__init__()

        if query_dim == source_dim:
            # No need for projection layer when sizes are the same.
            source_dim = None

        self.casual = casual

        # Dropout. Shared.
        self.dropout = nn.Dropout(dropout)

        # Self attention implementation
        self.sa_layernorm = nn.LayerNorm(query_dim)
        self.self_attn = MultiheadedAttention.create(query_dim,
                                                     num_heads,
                                                     dropout,
                                                     dtype=dtype,
                                                     device=device)

        # Cross attention implementation
        self.ca_layernorm = nn.LayerNorm(query_dim)
        self.cross_attention = MultiheadedAttention.create(query_dim,
                                                           num_heads,
                                                           dropout,
                                                           kdim=source_dim,
                                                           vdim=source_dim,
                                                           dtype=dtype,
                                                           device=device)

        # Feedforward implementation
        self.ff_layernorm = nn.LayerNorm(query_dim)
        self.feedforward = Feedforward(query_dim,
                                       dim_feedforward,
                                       dropout=dropout,
                                       device=device,
                                       dtype=dtype)

    def forward(self,
                tensor: torch.Tensor,
                context: torch.Tensor,
                self_mask: Optional[torch.Tensor] = None,
                cross_mask: Optional[torch.Tensor] = None,
                ) -> torch.Tensor:
        """
        Performs a transformer based attention decoding
        process. Supports multidimension batching.

        :param tensor:
            - The data to process
            - Shape (..., sequence, query_dim)
        :param context:
            - The context tensor to decode with
            - Shape (..., context_sequence, source_dim)
        :param self_mask:
            - An optional mask that corrolates with the self attention step.
            - Core shape is (query_sequence, query_sequence). Additional heads and batch feature alloed.
              batch content allowed.
            - First dimension is target, second is source.
            - See MultiheadedAttention for more details
        :param context_mask:
            - An optional mask that corrolates with the cross attention step
            - Restricts queries from accessing content elements.
            - Core shape (query_sequence, context_sequence)
            - See MultiheadedAttention for more details on masking.
        :return: The results of performing the transformer process
        """

        tensor = self.sa_layernorm(tensor)
        self_attn, _ = self.self_attn(tensor, tensor, tensor, attn_mask=self_mask, casual = self.casual)
        tensor = tensor + self.dropout(self_attn)

        tensor = self.ca_layernorm(tensor)
        cross_attn, _ = self.cross_attention(tensor, context, context, attn_mask=cross_mask)
        tensor = tensor + self.dropout(cross_attn)

        tensor = self.ff_layernorm(tensor)
        feedforward = self.feedforward(tensor)
        tensor = tensor + self.dropout(feedforward)

        return tensor

