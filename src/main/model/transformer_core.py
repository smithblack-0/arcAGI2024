import torch
from torch import nn
from typing import Tuple
from src.main.model.base import TensorTree

from src.main.model.attention.linear_routing_attention import LinearRoutingAttention
from src.main.model.routing_feedforward import RoutingFeedforward


class TransformerDecoderCell(nn.Module):
    """
    A recurrent decoder cell.

    The supportive transformer encoder cell.
    This is designed to read in the same manner as
    is seen in transformer encoders.
    """
    def setup_states(self, batch_shape: torch.Size)->TensorTree:
        state = {}
        state["controller"] = self.controller.make_state(batch_shape)
        state["sa_memories"] = self.self_attention.make_state(batch_shape)
        return state

    def __init__(self,
                 d_model: int,
                 controller: RoutingFeedforward,
                 self_attention: LinearRoutingAttention,

                 dropout: float = 0.1,
                 ):
        super().__init__()

        self.self_attention = self_attention
        self.sa_layernorm = nn.LayerNorm(d_model)

        self.controller = controller
        self.controller_layernorm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)


    def forward(self,
                embedding: torch.Tensor,
                state: TensorTree
                )->Tuple[torch.Tensor, TensorTree]:
        """
        Performs the forward pass of this tranformer
        encoder cell.
        :param embedding: The embedding to process. Shape (..., d_model)
        :param state: The existing recurrent state
        :return:
            - The new embeddings
            - The new recurrent state
        """

        # Perform self attention
        sa = self.self_attention(embedding, state["sa_memories"])
        embedding, sa_state = self.sa_layernorm(embedding + self.dropout(sa))

        # Perform feedforward
        ff = self.controller(embedding, state["controller"])
        embedding, ff_state = self.controller_layernorm(embedding + self.dropout(ff))

        # Return results

        return embedding, {"sa_memories" : sa_state, "feedforward" : ff_state}

