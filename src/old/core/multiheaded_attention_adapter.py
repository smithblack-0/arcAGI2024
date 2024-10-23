from typing import Optional, Tuple

import torch
from torch import nn, device
from src.main.model.virtual_layers import BankedLinear, BankSelector



class RecurrentLinearAttention(nn.Module):
    """Implement fast_transformers.long_term_memories.causal_linear_attention as a
    fixed-dimensional state recurrent model.

    See fast_transformers.long_term_memories.linear_attention and
    fast_transformers.long_term_memories.causal_linear_attention for the general concept
    of replacing the softmax with feature maps.

    Arguments
    ---------
        feature_map: callable, a callable that applies the feature map to the
                     last dimension of a tensor (default: elu(x)+1)
        eps: float, a small number to ensure the numerical stability of the
             denominator (default: 1e-6)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """
    def __init__(self, query_dimensions, feature_map=None, eps=1e-6,
                 event_dispatcher=""):
        super(RecurrentLinearAttention, self).__init__()
        self.feature_map = (
            feature_map(query_dimensions) if feature_map else
            elu_feature_map(query_dimensions)
        )
        self.eps = eps
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def forward(self, query, key, value, state=None, memory=None):
        # Normalize state/memory
        state = check_state(state, memory)

        # If this is a new sequence reinitialize the feature map
        if state is None:
            self.feature_map.new_feature_map(query.device)

        # Apply the feature map to the query and key
        Q = self.feature_map.forward_queries(query)
        K = self.feature_map.forward_keys(key)

        # Extract some shapes
        N, H, D = Q.shape
        _, _, M = value.shape

        # Extract the memory or initialize it
        if state is None:
            Si = query.new_zeros((N, H, D, M))
            Zi = query.new_zeros((N, H, D))
        else:
            Si, Zi = state

        # Ensure the batch size did not change
        if len(Si) != N:
            raise ValueError("The batch size changed during iteration")

        # Update the internal state
        #
        # NOTE: The if clause is added due to GitHub PR #10. Simply using the
        # following two lines does not perform the operation in place which
        # means it is slower for inference.
        if K.grad_fn is not None or value.grad_fn is not None:
            Zi = Zi + K
            Si = Si + torch.einsum("nhd,nhm->nhdm", K, value)
        else:
            Zi += K
            Si += torch.einsum("nhd,nhm->nhdm", K, value)

        # Compute the output
        Z = 1. / (torch.einsum("nhd,nhd->nh", Q, Zi) + self.eps)
        V = torch.einsum("nhd,nhdm,nh->nhm", Q, Si, Z)

        return V, [Si, Zi]
class BankedRecurrentState:
    """
    """
class BankedMakeHead(nn.Module):
    """
    A small module that is responsible for
    making the heads we may need when performing
    the long_term_memories process. It can handle banking.
    """
    def __init__(self,
                 d_in: int,
                 d_model: int,
                 heads: int,
                 bank_size: int,
                 device: torch.device,
                 dtype: torch.dtype,
                 ):
        self.d_heads = d_model//heads
        self.num_heads =  heads
        self.projection = BankedLinear(d_in, d_model, bank_size, device=device, dtype=dtype)

    def forward(self,
                tensor: torch.Tensor,
                bank_selections: torch.Tensor)->torch.Tensor:
        """
        Creates long_term_memories heads, in a banked manner. Can broadcast
        :param tensor: The thing to create long_term_memories heads using
            - Shape (..., d_model)
        :param bank_selections:
            - The selections that will actually be banked
            - Shape (..., banks)
            - Integers selecting banks.
        :return: The banked long_term_memories tensors.
            - Shape (..., banks, heads, d_heads)
        """
        tensor = tensor.unsqueeze(-2) # Add bank dimension (..., 1, d_model)
        tensor = self.projection(tensor, bank_selections) #(..., banks, d_model)
        tensor = tensor.view(*tensor.shape[:-1], self.num_heads, self.d_heads ) #(..., banks, num_heads, d_head)
        return tensor

class BankedMergeHeads(nn.Module):
    """
    A parameter banked version of the
    standard merge head action for transformers.
    """



class RecurrentLinearMHA(nn.Module):
    """
    A form of multiheaded long_term_memories

    """

class MultiheadedAttention(nn.Module):
    """
    An adapter layer. Allows us to use a torch multiheaded long_term_memories layer
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
        Creates and wraps up a multiheaded long_term_memories adapter, capable
        of processing batched data. All parameters have the same effects
        as their torch equivalent

        :param embed_dim: Total dimensions of the model
        :param num_heads: Number of heads
        :param dropout: dropout probability
        :param bias: If specified, adds bias to projection layers
        :param add_bias_kv: If specified, add bias to kv projection layers
        :param kdim: The dimensionalities of the keys
        :param vdim: The dimensionalities of the values
        :return: A multibatch multiheaded long_term_memories layer
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

        # Flatten the long_term_memories mask, if needed.
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

        # Run long_term_memories

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


class BankedMakeHead(nn.Module):
    """
    A make head mechanism designed to handle banked long_term_memories.
    """
    def __init__(self,
                 d_model: int,
                 d_head: int,
                 num_heads: int,
                 device: torch.device,
                 dtype: torch.dtype,
                 ):

        super().__init__()

        self.projector = BankedLinear(d_model, d_head, num_heads, device=device, dtype=dtype)

    def forward(self,
                tensor: torch.Tensor,
                head_selections: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward process that creates the heads
        :param tensor: The tensor to draw from
            - Shape (..., d_model)
            - The content to turn into heads
        :param head_selections:
            - The heads that are actually selected and will subsequently be evaluated.
        """
        tensor = tensor.unsqueeze(-2)
        headed_tensors = self.projector(tensor, head_selections) #(..., heads, d_heads)
        return headed_tensors

class BankedMergeHeads(nn.Module):
    """
    Merges the various heads that have been selected together. Does
    this using both a projection and using head probabilities.
    """
    def __init__(self,
                 d_model: int,
                 d_head: int,
                 num_heads: int,
                 ):

        super().__init__()

        self.projector = BankedLinear(d_head, d_model, num_heads, device=device, dtype=torch.float32)

    def forward(self,
                tensor: torch.Tensor,
                head_selection: torch.Tensor,
                head_probabilities: torch.Tensor
                ):
        """

        :param tensor: The tensor to merge heads on.
            - Shape (..., heads, d_heads)
        :param head_selection: The heads that were selected as active
            - Shape (..., heads)
            - Integers.
        :param head_probabilities:
            - The probabilities with which each head was selected.
            - Shape (..., heads).
        :return: A combined tensor
            - Shape (..., d_model)
        """

        # Perform projection into d_model space
        tensor = self.projector(tensor, head_selection) #(..., heads, d_model)

        # Perform weighted combine
        tensor = torch.matmul(tensor.T, head_probabilities.unsqueeze(-1)).squeeze(-1)

        # Return
        return tensor

class ManageMemories:
    """
    Extracts memories associated with
    the particular heads indicated from
    the more general memory space.
    """
    def __init__(self):
        super().__init__()

    def extract_memory_subset(self,
                              memories: Tuple[torch.Tensor, torch.Tensor],
                              head_select: torch.Tensor,
                              ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract a memory subset to do long_term_memories with, in a sparse
        manner.

        :param memories: The collection of all our memories to work with
            - matrices
                - Shape (...,heads,  keys, values)
            - normalizers:
                - Shape (..., heads, keys)
        :param head_select: The heads that have been selected to be active
            - Shape (..., selected_heads)
        :return: The memories that have been extracted
        """
        matrices, normalizers = memories
        matrices = matrices.gather(dim=-3, index=head_select)
        normalizers = normalizers.gather(dim=-2, index=head_select)
        return matrices, normalizers

    def incorporate_memory_update(self,
                                  memories: Tuple[torch.Tensor, torch.Tensor],
                                  updates: Tuple[torch.Tensor, torch.Tensor],
                                  head_selections: torch.Tensor,
                                  head_probabilities: torch.Tensor
                                  ):
        """
        Incorporates the sparse updates into the memories, while weighting them
        by the activation probabilities.
        :param memories: The memories requiring an update
             - Shape (..., heads, keys, values)
             - Shape (..., heads, keys)
        :param updates: The updates, in the reduced subspace
            - Shape (..., active_heads, keys, values)
            - Shape (..., active_heads, keys)
        :param head_selections: The heads that were selected, corrolated with updates
             - Shape (..., x)
        :param head_probabilities: The probability with which each head was created
             - Shape (... active_heads)
        :return: The new memories
        """
        # Take apart the memories and updates

        matrix_memory, normalizer_memory = memories
        matrix_update, normalizer_update = updates

        # Weight the updates by the head probabilities

        matrix_update = matrix_update*head_probabilities.unsqueeze(-1).unsqueeze(-1)
        normalizer_update = normalizer_update*head_probabilities.unsqueeze(-1)

        # Scatter the updates back into the memories
        matrix_memory = matrix_memory.scatter_add(dim=-3, index=head_selections, src=matrix_update)
        normalizer_memory = normalizer_memory.scatter_add(dim=-2, index=head_selections, src=normalizer_update)

        return matrix_memory, normalizer_memory


class RecurrentMemoryAttention(nn.Module):
    """
    Implements fast recurrent linear long_term_memories as
    a bank of heads that can and will be selected between on the
    fly. Only these heads will be processed, sparsely.
    Each "head" then becomes a memory slot the model can remember
    things in. Finally, the number of heads is usually much larger
    than in a normal model
    """

    def __init__(self,
                 d_model: int,
                 d_head: int,
                 num_heads: int,
                 num_active_heads: int,
                 device: torch.device,
                 dtype: torch.dtype,
                 ):

        super().__init__()

        self.head_selector = BankSelector(d_model, num_heads, device, dtype)
        self.query_projector = BankedLinear(d_model, d_head, num_heads, dtype=dtype, device=device)
        self.key_projector = BankedLinear(d_model, d_head, num_heads, dtype=dtype, device=device)
        self.value_projector = BankedLinear(d_model, d_head, num_heads, dtype=dtype, device=device)
        self.head_combiner = BankedLinear(d_head, d_model, num_heads, dtype=dtype, device=device)

    def forward(self, queries, memory: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        """
        Runs the memory long_term_memories mechanism.
        :param queries:
        :param memory:
        :return:
        """

        head_probabilities, head_selections = self.select_heads(queries)



class MultiHeadedAttention(nn.Module):
    """
    Implements a specialized kind of multiheaded long_term_memories that is
    capable of accepting multidimensional batches, and that is
    capable of
    """
