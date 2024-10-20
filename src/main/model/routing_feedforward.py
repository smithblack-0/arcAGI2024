from typing import Optional, Tuple

import torch
from torch import nn

from src.main.model import banks


class RoutingFeedforward(nn.Module):
    """
    Basically, the routing version of feedforward.

    * You have a ton of feedforward experts.
    * You only activate a certain number of them every feedforward iteration
    * They specialize in different things.
    * The results are added together and recombined.

    Notably, there is also a built in FSM that has transition probabilities,
    meaning it might not be possible to transition straight from one feedforward
    to another. This is meant to simulate running a complex combination of layers.
    """
    def __init__(self,
                 d_model: int,
                 d_hidden: int,
                 num_experts: int,
                 num_active_experts: int
                 ):
        """
        The initialization

        :param d_model: The incoming model features
        :param d_hidden: The number of hidden layers
        :param num_experts: The number of experts to store
        :param num_active_experts: The number of active experts.
        """
        super().__init__()

        self.d_model = d_model
        self.d_hidden = d_hidden
        self.num_experts = num_experts
        self.num_active_experts = num_active_experts

        # Define routing layers

        self.dynamic_logits = nn.Linear(d_model, num_experts)
        self.routing_bias_kernel = nn.Parameter(torch.Tensor(num_experts, num_experts))
        nn.init.uniform_(self.routing_bias_kernel, -0.1, 0.1)

        # Define feedforward process layers

        self.ff1 = banks.BankedLinear(d_model, d_hidden, num_experts, expand=True, squeeze=False)
        self.ff2 = banks.BankedLinear(d_model, d_hidden, num_experts, expand=False, squeeze=True)
        self.activation = nn.ReLU()

    def make_state(self, batch_shape: torch.Size)->torch.Tensor:
        """
        Makes the default routing state.
        :param batch_shape: The shape of the batch
        :return: The setup state vector
        """
        state = torch.zeros(list(batch_shape) + [self.num_experts])
        state[..., 0] = 1.0 # Start in state 1.0, unless otherwise noted.
        return state
    def forwards(self,
                 embedding: torch.Tensor,
                 state: Optional[torch.Tensor] = None
                 )->Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs the routing based feedforward pass. Returns the new embeddings, and
        the new routing state.

        :param embedding: The embedding to process. Shape (..., d_model)
        :param state: The current expert state. Shape (..., num_experts).
                      Indicate how active an expert was last feedforward.
        :return: The new embeddings and state
            - Embedding: (..., d_model)
            - state: (..., num_experts)
        """

        # Setup state if needed
        if state is None:
            state = self.make_state(embedding.shape[:-1])

        # Compute the transition biases. These are based on how likely
        # it is to transition from a state to another given state

        transition_biases = torch.matmul(self.routing_bias_kernel, state.unsqueeze(-1)).squeeze(-1) #(..., state)

        # Compute model transitions. Figure out routing probabilities. Use
        # that to make a selection. We combine the biases, which can make it more or less
        # likely to choose a transition if we are in a particular state, and the
        # actual dynamic logits

        transition_logits = self.dynamic_logits(embedding) + transition_biases
        transition_logits = self.dropout_logits(transition_logits)
        transition_probabilities = torch.softmax(transition_logits, dim=-1)

        # Make the selection.
        #
        # The selection schema is reversed to topk

        top_probabilities, top_index = transition_probabilities.topk(self.num_active_experts)
        selection = (top_index, top_probabilities)

        # Run the feedforward process

        embedding = self.ff1(embedding, selection)
        embedding = self.activation(embedding)
        embedding = self.ff2(embedding, selection)

        # Return

        return embedding, transition_probabilities
