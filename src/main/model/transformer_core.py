import torch
import math
from torch import nn
from typing import Tuple, Any, Optional
from src.main.model.base import TensorTree, StatefulCore, DropoutLogits
from fast_transformers import builders

import banks

class MakeHeads(nn.Module):
    """
    Exactly what it says on the tin. Designed to make
    heads for usage in the routing transformer. It is
    a banked module which can have the active heads dynamically
    selected. It will generate these head dimensions
    """
    def __init__(self,
                 d_model: int,
                 d_head: int,
                 num_heads: int
                 ):
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.num_heads = num_heads

        # Setup head creator

        self.projector = banks.BankedLinear(d_model, d_head, num_heads, expand=True, squeeze=False)
    def forward(self,
                embedding: torch.Tensor,
                selection: Tuple[torch.Tensor, torch.Tensor]
                )->torch.Tensor:
        """
        Forward pass, to create the headed embeddings
        :param embedding: Shape (..., d_model). The embeddings to expand
        :param the head selection:
            - Shape (..., active_heads). Integers. Indicates what heads to use. We actually care about this
            - Shape (..., active_heads). Probabilities. Do not care
        :return: Shape (..., active_heads, d_head)
        """
        embedding = self.projector(embedding, selection)
        return embedding

class MergeHeads(nn.Module):
    """
    Merges active heads back into a single embedding.
    """
    def __init__(self, d_model: int, d_head: int, num_heads: int):
        super().__init__()

        self.d_model = d_model
        self.d_head = d_head
        self.num_heads = num_heads

        # Setup head merger
        self.projector = banks.BankedLinear(d_head, d_model, num_heads, expand=False, squeeze=True)
    def forward(self,
                embedding: torch.Tensor,
                selection: Tuple[torch.Tensor, torch.Tensor]
                )->torch.Tensor:
        """
        Forward pass, to recombine the headed embeddings.
        :param embedding: Shape (..., active_heads, d_head)). The embeddings to expand
        :param the head selection:
            - Shape (..., active_heads). Integers. Indicates what heads to use. We actually care about this
            - Shape (..., active_heads). Probabilities. We need this.
        :return: Shape (..., active_heads, d_head)
        """
        embeddings = self.projector(embedding, selection)
        return embeddings


class LinearRoutingAttention(nn.Module):
    """
    Recurrent linear routing attention, with head
    decay based on natural decay and update strength.
    It operates in linear time.

    Is a recursive cell: Expect inputs to have nature
    of (..., d_model) with no sequence dimension!
    """

    @property
    def decay_factor(self)->torch.Tensor:
        return torch.sigmoid(self.decay_logits)

    def __init__(self,
                 d_model: int,
                 d_key: int,
                 d_value: int,
                 d_head: int,
                 num_heads: int,
                 num_active_heads: int,
                 dropout: float = 0.2,
                 write_epsilon: float = 0.001
                 ):

        super().__init__()

        self.d_model = d_model
        self.d_key = d_key
        self.d_value = d_value
        self.d_head = d_head
        self.num_heads = num_heads
        self.num_active_heads = num_active_heads
        self.write_epsilon =write_epsilon

        # Setup head dropout.
        self.dropout_logits = DropoutLogits(dropout)

        # Setup head projectors and restorers

        self.query_head_projector = MakeHeads(d_model, d_head, num_heads)
        self.key_head_projector = MakeHeads(d_model, d_key, num_heads)
        self.value_head_projector = MakeHeads(d_model, d_value, num_heads)
        self.combine_heads = MergeHeads(d_model, d_head, num_heads)

        # Set up expert read selection and expert write selection logit storage

        self.expert_logit_projector = nn.Linear(d_model, num_heads)
        self.write_logits_projector = banks.BankedLinear(d_head, 1, num_heads,
                                                         expand=False, squeeze=False)

        # Set up the attention mechanism.

        self.rla = builders.RecurrentAttentionBuilder.from_kwargs(query_dimensions = d_head)

        # Set up the initial decay strengths. We want decay strengths that corrolate
        # with various decay factors of between 0.2 and 0.001, distributed about
        # evenly. We go backwards through a sigmoid function to find what these are

        decay_factors = torch.zeros([num_heads, d_head])
        decay_factors.uniform_(0.001, 0.2)
        decay_logits = -torch.log((1/decay_factors) - 1)

        self.decay_logits = nn.Parameter(decay_logits, requires_grad=True)


        # Set up the default state.

        self.default_normalizer = nn.Parameter(torch.zeros([num_heads, d_head]))
        self.default_matrix = nn.Parameter(torch.zeros([num_heads, d_head, d_head]))

    def select_experts(self,
                       query: torch.Tensor,
                       )->Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Selects the experts that will be used.
        :param query: The query to base our selection on.
        :return: The selections tuples. Also the logits.
            Selections are Consists of:
            - Selections: (..., active_heads). Integers. Indicates the banks selected
            - Probabilities: (..., active_heads). Float. The probabilities we selected with.
            And logits is:
            - top logits: (..., active_heads). Logits associated with those heads.
        """
        logits = self.expert_logit_projector(query) #(..., experts).
        logits = self.dropout_logits(logits)
        probabilities = torch.softmax(logits, dim=-1) #(..., experts)

        # Figure out the top features
        top_probabilities, top_index = probabilities.topk(self.num_active_heads) #(..., selected_experts)
        top_logits = logits.gather(dim=-1, index=top_index)
        return (top_index, top_probabilities), top_logits

    def setup_state(self,
                  batch_shape: torch.Size,
                  )->Tuple[torch.Tensor, torch.Tensor]:
        """
        Sets up a defaul state with the indicated batch shape

        :param batch_shape: The shape of the batch
        :return: The state.
        """

        normalizer_state = self.default_normalizer
        matrix_state = self.default_matrix

        # Unsqueeze to have batch dimensions
        for _ in batch_shape:
            normalizer_state = normalizer_state.unsqueeze(0)
            matrix_state = matrix_state.unsqueeze(0)

        # Expand
        normalizer_state = normalizer_state.expand(list(batch_shape) + [-1]*2)
        matrix_state = matrix_state.expand(list(batch_shape) + [-1]*3)

        return matrix_state, normalizer_state

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                state: Optional[TensorTree] = None
                )->Tuple[torch.Tensor, Any]:
        """
        The forward pass of the memory routing transformer. Involves
        selecting the correct experts, evaluating them, then
        :param query: The query. Shape (..., d_model)
        :param key: The key. Shape (..., d_model)
        :param value: The value. Shape (..., d_model)
        :param state: The state. Consists of matrix and normalizer.
        :return: The output, and the new state.
        """

        if state is None:
            state = self.setup_state(query.shape[:-1])


        # Select the experts. Also, get the state, or initialize it if needed.

        matrix, normalizer = self.initialize_state(query.shape[:-1], state)
        selection, top_logits = self.select_experts(query)

        # Compute the heads.
        query = self.query_head_projector(query, selection)
        key = self.key_head_projector(key, selection)
        value = self.value_head_projector(value, selection)

        # Compute the subset of the head state we are going to work with

        submatrix = banks.banked_state_select(matrix, selection, dim=-3)
        subnormalizer = banks.banked_state_select(normalizer, selection, dim=-2)

        # Perform attention. Figure out the state update as the difference
        # between the new and original state
        output, (new_submatrix, new_subnormalizer) = self.rla(query, key, value, [submatrix, subnormalizer])

        submatrix_update = new_submatrix - submatrix
        subnormalizer_update = new_subnormalizer - subnormalizer

        # Now we need to actually place these updates into the head specialist. The rules for this
        # are as follows
        #
        # 1): You must pass both the read and the write check to be committed.
        # 2): committing an update means your memory decays a bit.

        # Develop actual write probabilities.
        write_logits = top_logits + self.write_logits_projector(output + query, selection)
        write_probabilities = torch.sigmoid(write_logits)

        # Perform the actual writes. In proportion to how strong I am writing, I decay my memories by
        # the decay factors.

        subnormalizer = subnormalizer*(1 - write_probabilities*self.decay_factor) + \
                        write_probabilities*subnormalizer_update
        submatrix = submatrix*(1-write_probabilities.unsqueeze(-1)*self.decay_factor.unsqueeze(-1)) + \
                    write_probabilities.unsqueeze(-1)*submatrix_update

        # Now insert this back into the main context.

        matrix = banks.banked_state_scatter(matrix, submatrix, selection, dim=-3)
        normalizer = banks.banked_state_scatter(normalizer, subnormalizer, selection, dim=-2)

        # Finally, lets go resolve all those heads!

        output = self.combine_heads(output, selection)

        return output, [matrix, normalizer]

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

