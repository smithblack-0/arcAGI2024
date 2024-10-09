
import torch
import math
from torch import nn
from typing import Tuple, Any, Optional
from abc import abstractmethod

from src.main.model.subroutine_driver import SubroutineCore
from src.main.model.base import TensorTree, StatefulCore
from src.main.model.banks import BankedLinear


def dot_product_attention(query, key, value, mask=None):
    # Step 1: Compute the raw attention scores (QK^T)
    scores = torch.matmul(query, key.transpose(-2, -1))  # (batch_size, num_heads, seq_len, seq_len)

    # Step 2: Optionally apply the mask
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # Step 3: Apply softmax to get attention weights
    attention_weights = F.softmax(scores, dim=-1)  # (batch_size, num_heads, seq_len, seq_len)

    # Step 4: Multiply the weights with the values
    output = torch.matmul(attention_weights, value)  # (batch_size, num_heads, seq_len, value_dim)

    return output, attention_weights


def compute_attention_weights(query: torch.Tensor,
                              key: torch.Tensor,)

class MakeHead(nn.Module):
    """
    Exactly what it says on the tin. Makes attention heads
    out of the provided. Nothing more to it.
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


        self.projector = nn.Linear(d_model, d_head*num_heads)

    def forward(self, embedding: torch.Tensor)->torch.Tensor:
        """
        Forward pass, to create the embedding
        :param embedding: Shape (..., d_model)
        :return: Shape (..., heads, d_head)
        """
        embedding = self.projector(embedding)
        embedding = embedding.view(list(embedding.shape[:-1]) + [self.heads, self.d_head])
        return

class MergeHeads(nn.Module):
    """
    Exactly what it says on the tin. Merges
    headed attention back to d_model
    """
    def __init__(self, d_model: int, d_head: int, num_heads: int):
        super().__init__()

        self.d_model = d_model
        self.d_head = d_head
        self.num_heads = num_heads

        self.projector = nn.Linear(d_head*num_heads, d_model)
    def forward(self, embeddings: torch.Tensor)->torch.Tensor:
        """
        Forward pass. Combine embeddings
        :param embeddings: The embeddings to combine. Shape (..., heads, d_head)
        :return: The combined embeddigns. Shape (..., d_model)
        """
        embeddings = embeddings.view(list(embeddings.shape[-2]) + [self.heads*self.d_heads])
        embeddings = self.projector(embeddings)
        return embeddings


class MemorySetup(nn.Module):
    """
    Sets up memory, if not already setup.
    """
    def __init__(self,
                 d_head: int,
                 num_memories: int
                 ):
        super().__init__()

        self.d_head = d_head
        self.num_memories = num_memories

        # Setup defaults.
        self.matrix_default = nn.Parameter(torch.zeros([num_memories, d_head, d_head]))
        self.normalizer_default = nn.Parameter(torch.zeros([num_memories, d_head]))
    def forward(self,
                embedding: torch.Tensor,
                memories: Optional[Tuple[torch.Tensor, torch.Tensor]]
                )->Tuple[torch.Tensor, torch.Tensor]:
        """
        :param embedding: The embedding we will be working with. Shape (..., d_model)
        :param memories: The memories to check or initialize. May be none.
            - Matrix: (..., memories, d_head, d_head)
            - Normalizer: (..., memories, d_head)
        :return:  The setup memories.
            - Matrix: (..., memories, d_head, d_head)
            - Normalizer: (..., memories, d_head)
        """
        if memories is None:
            # Get batch shape
            batch_shape = embedding.shape[:-1]

            # Start working on memories and normalizer
            matrix = self.matrix_default
            normalizer = self.normalizer_default

            # Unsqueeze to now possess batch dimensions
            for _ in batch_shape:
                matrix = matrix.unsqueeze(0)
                normalizer = normalizer.unsqueeze(0)

            # Insert batch dimensions
            matrix = matrix.expand(list(batch_shape) + [-1, -1, -1])
            normalizer = normalizer.expand(list(batch_shape) + [-1, -1])

            # package
            memories = (matrix, normalizer)
        return memories

class MemoryConnectiveLogits(nn.Module):
    """
    A small helper class, responsible for
    generating on demand memory expression logits
    that will relate memories to the queries
    from the heads. You can think of it as partially
    completing dot product attention.
    """
    def __init__(self,
                 d_head: int,
                 num_memories: int,
                 ):
        super().__init__()

        self.num_memories = num_memories
        self.addresses = nn.Parameter(torch.empty(num_memories, d_head))
        self.layernorm = nn.LayerNorm(d_head)

        nn.init.normal_(self.addresses, 0, d_head ** -0.5)

    def forward(self,
                query: torch.Tensor,
                addresses: torch.Tensor
                )->torch.Tensor:
        """
        Create the access logits using an attention focus mechanism without
        activation.

        :param query: Shape is usually (..., heads, d_heads)
        :param addresses: Memory addresses. Shape is usually (..., memory, d_memory)
        :return: Transfer logits. Shape (..., heads, memory)
        """
        key = self.layernorm(addresses) + self.addresses #(..., memories, d_heads)
        logits = torch.matmul(query, key.transpose(-2, -1)) #(..., heads, memories)
        return logits

class MemoryExpressionAttention(nn.Module):
    """
    Performs memory expression process. This collects insights across
    all the memories to whatever is immediately useful for the given
    requests from the heads.
    """
    def __init__(self):
        super().__init__()

    def forward(self,
                connective_logits: torch.Tensor,
                matrix: torch.Tensor,
                normalizer: torch.Tensor
                )->Tuple[torch.Tensor, torch.Tensor]:
        """
        :param connective_logits: The computed access logits. Shape (..., heads, memories)
        :param matrix: The matrix accumulators, across all memories. Shape (..., memories, d_head, d_head)
        :param normalizer: the normalizer sums, across all memories. Shape (..., memories, d_head)
        :return: The matrix and normalizer expressed with heads
            - matrix: shape (..., heads, d_head, d_head)
            - normalizer: shape (..., memories, d_head)
        """

        # Activate the expression logit, to get something that can transfer from memories to heads.
        # This effectivelly ends up telling us how strongly a particular memory will express
        # on a particular head during computation.
        expression_probabilities = torch.softmax(connective_logits, dim=-1)

        # Prepare everything for attention, by flattening the matrix. We will restore the
        # shape later

        final_shape = list(matrix.shape) #(..., memories, d_head, d_head)
        final_shape[-3] = connective_logits.shape[-2] #(..., head, d_head, d_head)
        matrix = matrix.flatten(-2, -1) # (..., heads, d_head*d_head)

        # Finish attention

        matrix = torch.matmul(expression_probabilities, matrix)
        normalizer = torch.matmul(expression_probabilities, normalizer)

        # Restore shapes, and return

        matrix = matrix.view(final_shape)

        return matrix, normalizer

class MemoryUpdateAttention(nn.Module):
    """
    Pledges to re-express memory updates that are currently
    found in terms of heads to instead be in terms of
    the memory banks. Uses expression logits to do so.
    """
    def __init__(self,
                 d_head: int,
                 num_memories: int,
                 num_heads: int
                 ):

        super().__init__()
        self.num_heads=num_heads
    def forward(self,
                connective_logits: torch.Tensor,
                update_matrix: torch.Tensor,
                update_normalizer: torch.Tensor,
                )->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Re-expresses the update matrix and update normalizers,
        which are CURRENTLY in terms of the heads, in terms of the
        memory banks.

        :param connective_logits: The connective logits. Shape (..., heads, memories)
        :param update_matrix: The update heads to transiton. Shape (..., heads, d_head, d_head)
        :param update_normalizer: The update normalizer to transition. Shape (..., heads, d_head)
        :return: The transitioned normalizer. It has been transitioned into the memory space
                Matrix: Shape (..., memories, d_head, d_head)
                Normalizer: Shape (..., memories, d_head)
        """
        # Activate the expression logit, to get something that can transfer from memories to heads.
        # This effectively ends up telling us how strongly a particular memory will express
        # on a particular head during computation.
        #
        # Interestingly, we actually want fairly sparse memory updates, so we still active
        # the same way. Every head is built out of a superposition of memories, and we are
        # updating based on how strong those superpositions were selected.

        update_probability = torch.softmax(connective_logits, dim=-1) #(..., heads, memory)
        update_probability = update_probability.swapdims(-1, -2) #(..., memories, heads)

        # Prepare the matrix for attention

        shape = list(update_matrix.shape)
        shape[-3] = connective_logits.shape[-2]
        update_matrix = update_matrix.flatten(-2, -1) #(..., heads, d_model*d_model)

        # Perform attention, creating actual updates.
        #
        # Note the update probability will be needed later on when we start
        # considering how much the memories need to decay
        update_matrix = torch.matmul(update_probability, update_matrix)
        update_normalizer = torch.matmul(update_probability, update_normalizer)

        # This deserves a little discussion. It examines how strongly a particular memory
        # is being attended to. Maximal attention would mean all five heads attend exactly
        # to it, which would result in all those channels being 1.0. Hence a mean rather than
        # a sum.
        update_probability = update_probability.mean(dim=-2) #(..., memories)

        # Restore

        update_matrix = update_matrix.view(shape)

        # Return
        return update_matrix, update_normalizer, update_probability


class LinearMemoriesAttention(StatefulCore):
    """
    A special version of recurrent linear
    """

    def setup_state(self,
                    embeddings: torch.Tensor
                    )-> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sets up the memories configured to be bound onto the indicated
        embeddings.
        :param embeddings: The embeddings. Shape (..., d_model)
        :return:
            - Matrix: KeyValue matrix. Shape (..., d_key, d_value)
            - Normalizers: Sum of activated keys. Shape (..., d_key)
        """




    def __init__(self,
                 d_model: int,
                 d_keys: int,
                 d_values: int,
                 d_heads: int,
                 num_heads: int,
                 num_memories: int
                 ):

        super().__init__()

        # Head creation and collapse mechanisms

        self.make_query_heads = MakeHead(d_model, d_heads, num_heads)
        self.merge_heads = MergeHeads(d_model, d_heads, num_heads)

        # Memory addressing and attentionlike mechanisms.

        self.create_memory_expressions = MemoryExpressionAttention()

        nn.init.normal_(self.addresses, 0, 1)
        nn.init.uniform_(self.decay_rate, 3)
    def forward(self,
                embeddings: torch.Tensor,
                memories: TensorTree,
                *unused: Any
                ):
        """
        Forward pass of recurrent memories attention.
        :param embeddings: The embeddings to process. Shape (..., d_model). No items dimension
        :param memories: The memory banks we have to work with.
            - MatrixBank: (..., memory_banks, k_dim_head, v_dim_head)
            - AddressBank: (..., memory_banks, k_dim_head). Does double duty as normalizers too.
        :param unused:
        :return:
        """
        # Unbox memories.
        # Make heads
        address_memories = memories[1]
        embeddings = self.make_query_heads(embeddings) #(..., heads, d_heads)

        # Create transfer logits. These will allow us to transfer from memories into
        # a headed representation, or alternatively from a headed representation
        # back into memories. They are basically unactivated attention weights.
        expression_logits = self.make_transfer_logits(embeddings, address_memories)


        # Express the memories in the heading space. Then perform linear attention
        expressed_memories = self.create_memory_expressions(transfer_logits, memories)
        embedding, memory_updates = self.linear_attention(embeddings, expressed_memories)


        # Create update associations based on retrieved information. Update the memories
        update_logits = self.make_update_logits(embedding,  address_memories)
        update_logits += expression_logits
        memories = self.update_memory(update_logits, memories, memory_updates)

        # Return
        return embedding, memories

class RecurrentMemoreis