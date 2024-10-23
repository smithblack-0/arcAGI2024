
import torch
import math
from torch import nn
from typing import Tuple, Any, Optional
from src.main.model.base import TensorTree, StatefulCore
from fast_transformers import builders

class MakeHead(nn.Module):
    """
    Exactly what it says on the tin. Makes long_term_memories heads
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
        embedding = embedding.view(list(embedding.shape[:-1]) + [self.num_heads, self.d_head])
        return embedding

class MergeHeads(nn.Module):
    """
    Exactly what it says on the tin. Merges
    headed long_term_memories back to d_model
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
        embeddings = embeddings.view(list(embeddings.shape[:-2]) + [self.num_heads*self.d_heads])
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
    completing dot product long_term_memories.
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
        Create the access logits using an long_term_memories focus mechanism without
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

        # Prepare everything for long_term_memories, by flattening the matrix. We will restore the
        # shape later

        final_shape = list(matrix.shape) #(..., memories, d_head, d_head)
        final_shape[-3] = connective_logits.shape[-2] #(..., head, d_head, d_head)
        matrix = matrix.flatten(-2, -1) # (..., heads, d_head*d_head)

        # Finish long_term_memories

        matrix = torch.matmul(expression_probabilities, matrix)
        normalizer = torch.matmul(expression_probabilities, normalizer)

        # Restore shapes, and return

        matrix = matrix.view(final_shape)

        return matrix, normalizer

class UpdateMemories(nn.Module):
    """
    Pledges to update the memories based on the update
    that was returned during linear long_term_memories process. This means
    expressing that update in terms of the memories space, then
    integrating it and performing any memory decay.
    """
    def __init__(self,
                 d_head: int,
                 num_memories: int,
                 num_heads: int,
                 ):

        # Precompute the mean to scatter around to get a starting decay factor similar to
        # above. A simple set of zeros may be too aggressive to ever form connections.

        super().__init__()
        self.num_heads=num_heads
        self.num_memories=num_memories
        self.d_head = d_head

        # Set up the inital decay strengths. We want decay strengths that corrolate
        # with various decay factors of between 0.2 and 0.001, distributed about
        # evenly. We go backwards through a sigmoid function to find what these are

        decay_factors = torch.zeros([num_memories, d_head])
        decay_factors.uniform_(0.001, 0.2)
        decay_logits = -torch.log((1/decay_factors) - 1)

        self.decay_strength = nn.Parameter(decay_logits, requires_grad=True)


    def flatten_memories(self,
                         memory: Tuple[torch.Tensor, torch.Tensor]
                         )->Tuple[torch.Tensor, torch.Tensor]:
        """
        One of the two memory elements has an extra dimension to it.
        This makes many processes, such as long_term_memories or decay, quite
        inconvenient. We flatten it away, with intention to restore it later

        :param memory: The memory tuple. Consists of
            - Matrix: (..., items, d_head, d_head)
            - Normalizer: (..., items, d_head)
        :return: The flattened memory. Consists of
            - Matrix: (..., memories, d_head*d_head)
            - Normalizer: (..., memories, d_head)
        """

        matrix, normalizer = memory
        matrix = matrix.flatten(-2, -1)
        return matrix, normalizer

    def unflatten_memories(self,
                           memory: Tuple[torch.Tensor, torch.Tensor]
                           )->Tuple[torch.Tensor, torch.Tensor]:
        """
        The inverse of flatten memories. Restores the original and
        useful state.

        :param memory: The flattened memory. Consists of
            - Matrix: (..., memories, d_head*d_head)
            - Normalizer: (..., memories, d_head)
        :return: The memory tuple. Consists of
            - Matrix: (..., items, d_head, d_head)
            - Normalizer: (..., items, d_head)
        """
        matrix, normalizer = memory
        d_head = normalizer.shape[-1]
        matrix = matrix.unflatten(-1, [d_head, d_head])
        return matrix, normalizer

    def forward(self,
                connection_logits: torch.Tensor,
                update: Tuple[torch.Tensor, torch.Tensor],
                memories: Tuple[torch.Tensor, torch.Tensor]
                )->Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs the processes which will result in updated memories. They
        will be updated based on the connection strength, and decay only is
        applied when a memory is being written to.

        :param connection_logits: The connection logits to use during long_term_memories
        :param update: The memory update. It is expressed in the head subspace rather than the memories
            - Matrix: (..., heads, d_head, d_head)
            - Normalizer: (..., head, d_head)
        :param memories: The memories we need to update. We figure out how strongly the update is associated
                         with each particular memory, then merge
            - Matrix: (..., memories, d_head, d_head)
            - Normalizer: (..., memories, d_head)
        :return: The updated memories
            - Matrix: (..., memories, d_head, d_head)
            - Normalizer: (..., memories, d_head)
        """

        # It is MUCH easier to handle further updates with the matrix flattened. Lets revise

        update = self.flatten_memories(update)
        memories = self.flatten_memories(memories)

        # Activate the expression logit, to get something that can transfer from memories to heads.
        # This effectively ends up telling us how strongly a particular memory will express
        # on a particular head during computation.
        #
        # Interestingly, we actually want fairly sparse memory updates, so we still active
        # the same way. Every head is built out of a superposition of memories, and we are
        # updating based on how strong those superpositions were selected.

        update_probability = torch.softmax(connection_logits, dim=-1) #(..., heads, memory)
        update_probability = update_probability.swapdims(-1, -2) #(..., memories, heads)

        # Compute the memories decay factor. This will be a product of the native
        # decay factor - which tells us how much to decay our memories when accessed
        # - and the update strength, which tells us, in turn, how strongly we are actually
        # being accessed.

        update_strength = update_probability.mean(-1) #(..., memories)
        native_decay_factor = torch.sigmoid(self.decay_strengths) #(memories, d_head)
        decay_factor = update_strength.unsqueeze(-1)*native_decay_factor


        # Perform the update process
        final_memories = []
        for memory_case, update_case in zip(memories, update):
            # memory_case: (..., memories, $something)
            # update_case: (..., heads, $something)

            # Perform long_term_memories to express the update case in the native dimensionality.
            update_case = torch.matmul(update_probability, update_case) #(..., memories, $something)

            # The memory case is going to decay according to the product of the activated decay
            # strengths. We

            if memory_case.shape[-1] != self.d_head:
                # This means we are processing the decay on the matrix. Repeat d_head times
                # to apply the same operation across all value channels
                repeat_shape = [1]*(memory_case.dim() - 1) + [self.d_head]
                adjusted_decay_factor = decay_factor.repeat(repeat_shape) #(..., memories, d_head*d_head)
            else:
                # Just apply as is
                adjusted_decay_factor = decay_factor #(..., memories, d_head)

            # Create updated memory. Store
            memory_case = memory_case*(1-adjusted_decay_factor) + update_case
            final_memories.append(memory_case)

        # Unflatten memories
        final_memories = tuple(final_memories)
        final_memories = self.unflatten_memories(final_memories)
        return final_memories

class LinearMemoriesAttention:
    """
    A special version of linear recurrent long_term_memories that draws content from
    a large bank of memories to produce a head specific focus. Linear long_term_memories
    is performed, and the update is then propogated back into the memories.
    A decay factor lets us capture short vs long term dependencies.
    """
    def __init__(self,
                 d_model: int,
                 d_key: int,
                 d_value: int,
                 d_heads: int,
                 num_heads: int,
                 num_memories: int
                 ):

        super().__init__()

        # Head creation and collapse mechanisms, and primary long_term_memories.

        self.make_query_heads = MakeHead(d_model, d_heads, num_heads)
        self.make_key_heads = MakeHead(d_model, d_heads, num_heads)
        self.make_value_heads = MakeHead(d_model, d_heads, num_heads)
        self.merge_heads = MergeHeads(d_model, d_heads, num_heads)

        # Memory logit addressing

        self.create_expression_logits = MemoryConnectiveLogits(d_heads, num_memories)
        self.create_update_logits = MemoryConnectiveLogits(d_heads, num_memories)

        # Linear long_term_memories
        self.rla = builders.RecurrentAttentionBuilder.from_kwargs(query_dimensions = d_heads)

        # memory mapping. In and out.

        self.express_memories_at_heads = MemoryExpressionAttention()
        self.update_memories = UpdateMemories(d_heads, num_memories, num_heads)

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                memories: TensorTree
                ):
        """
        Forward pass of recurrent memories long_term_memories.
        :param embeddings: The embeddings to process. Shape (..., d_model). No items dimension
        :param memories: The memory banks we have to work with.
            - MatrixBank: (..., memory_banks, k_dim_head, v_dim_head)
            - AddressBank: (..., memory_banks, k_dim_head). Does double duty as normalizers too.
        :param unused:
        :return:
        """
        # Unbox memories.
        # Make heads.
        address_memories = memories[1]
        query = self.make_query_heads(query) #(..., heads, d_heads)
        key = self.make_key_heads(key)
        value = self.make_value_heads(value)

        # Create the expression logits, and express the memories in terms of the
        # heads. Then perform the linear long_term_memories.
        expression_logits = self.create_expression_logits(query, address_memories)
        expressed_memories = self.express_memories_at_heads(expression_logits, memories)

        # Perform linear long_term_memories
        # Extract memory update. Fast transformers only returns the new memory state, not
        # the memory update. A quick difference retrieves it.
        embeddings, memory_updates = self.rla(query, key, value, state=expressed_memories)
        memory_updates = tuple(map(lambda x, y: x-y, memory_updates, memories))

        # Create the update logits. We add this to the expression logit, which creates
        # a possible corrolation between what is read and what is updated - usually a safe
        # assumption. Then we update the memory. The output embeddings are used to influence
        # how we store memories.

        expression_logits = expression_logits + self.create_update_logits(embeddings, address_memories)
        memories = self.update_memories(expression_logits, memory_updates, memories)

        # Merge the heads back together

        embeddings = self.merge_heads(embeddings)

        # Return
        return embeddings, memories
