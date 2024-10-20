import torch


class AdaptiveComputationStack:
    """
    A differentiable stack designed to manage computation with probabilistic pointers for tasks involving
    adaptive computation. It enables stack-based subroutine management, where each level of the stack is
    weighted by probabilistic pointers, allowing smooth transitions between subroutine states.

    The stack accumulates decisions about enstack (push), destack (pop), or no-op at each level based on
    action probabilities provided by the external model. It tracks statistics on these transitions over
    time for further analysis.

    Updates for stack adjustment, statistics accumulation, and embeddings are controlled using a `batch_mask`.
    The `batch_mask` is a boolean tensor where a value of `1` indicates that no updates should occur for
    that batch, while a value of `0` allows updates. This is useful for controlling when updates occur
    during adaptive computation, especially when some batches need to be skipped.

    Args:
        stack_depth (int): The depth of the stack (i.e., number of levels).
        embedding_shape (torch.Size): Shape for embeddings. All but the last dimension are batch-like,
                                      and the last dimension corresponds to the embedding size (d_model).
        dtype (torch.dtype, optional): Data type for tensors. Defaults to None.
        device (torch.device, optional): Device on which to store tensors. Defaults to None.

    Properties:
        normalized_decision_statistics: Provides normalized statistics on the decisions (enstack, no-op, destack)
                                        made at each stack level.
        normalized_level_statistics: Provides normalized statistics on activity across stack levels.
        probability_mass: Returns the total accumulated probability mass for further analysis.

    Methods:
        adjust_stack(probabilities, batch_mask): Adjusts the stack based on probabilities of enstack, no-op,
                                                  or destack. Updates occur only where batch_mask is 0.
        get_expression(): Returns the current stack weighted by probabilistic pointers.
        set_expression(embedding, batch_mask): Sets the stack state based on pointer probabilities and a given embedding,
                                               but skips updates where batch_mask is 1.
        __call__(embedding, probabilities, batch_mask): Combines the set, adjust, and get actions for
                                                         convenience, performing all three in sequence.
    """
    @property
    def normalized_decision_statistics(self) -> torch.Tensor:
        """
        Provides information, per stack level, on what the most common decisions that are being made are.
        """
        return self.probability_statistics / (self.probability_statistics.sum(dim=-1, keepdim=True) + 1e-4)

    @property
    def normalized_level_statistics(self) -> torch.Tensor:
        """
        Provides information, per stack level, on how much activity is occurring at that stack level.
        :return: (stack, ...). Normalized activity showing how much activity is occurring at each stack level.
        """
        statistics = self.probability_statistics.sum(dim=-1)  # Sum across actions at each stack level
        return statistics / (statistics.sum(dim=0, keepdim=True) + 1e-9)

    def probability_mass(self) -> torch.Tensor:
        """
        Indicates how much probability was expended on each batch.
        :return: The sum of the probabilities.
        """
        return self.probability_statistics.movedim(0, -1).flatten(-2, -1).sum(dim=-1)

    def __init__(self, stack_depth: int, embedding_shape: torch.Size, dtype: torch.dtype = None,
                 device: torch.device = None):
        """
        Initialize the AdaptiveComputationStack.

        :param stack_depth: The depth of the stack.
        :param embedding_shape: Shape for embeddings. All but the last dimension are batch-like,
                                and the last dimension is d_model.
        """
        self.stack_depth = stack_depth
        self.batch_shape = embedding_shape[:-1]
        self.d_model = embedding_shape[-1]
        self.dtype = dtype
        self.device = device

        # Initialize stack with zeros
        self.stack = torch.zeros([stack_depth, *embedding_shape], dtype=dtype, device=device)

        # Initialize probabilistic pointers with all probability at stack position 0
        self.pointers = torch.zeros(stack_depth, *embedding_shape[:-1], dtype=dtype, device=device)
        self.pointers[0] = 1.0

        # Initialize the probability mass statistics.
        self.probability_statistics = torch.zeros([stack_depth, *self.batch_shape, 3], dtype=dtype, device=device)

    def record_statistics(self, action_probabilities: torch.Tensor, pointer_probabilities: torch.Tensor,
                          batch_mask: torch.Tensor):
        """
        Accumulates statistics at each stack level based on action probabilities and pointer probabilities,
        but only updates the statistics if batch_mask is True.

        :param action_probabilities: The action probabilities. Shape (..., 3).
        :param pointer_probabilities: The pointer probabilities. Shape (stack, ...).
        :param batch_mask: Tensor that indicates whether to proceed with the update or skip it (if False).
        """
        if batch_mask.shape != self.batch_shape:
            raise ValueError("Batch mask must match batch shape")

        batch_mask = batch_mask.unsqueeze(0)  # Expand the mask to stack level
        while batch_mask.dim() < self.probability_statistics.dim():
            batch_mask = batch_mask.unsqueeze(-1)

        # Accumulate probability mass for actions across stack levels.
        probability_mass = action_probabilities.unsqueeze(0) * pointer_probabilities.unsqueeze(-1)
        updated_statistics = self.probability_statistics + probability_mass
        self.probability_statistics = torch.where(batch_mask, self.probability_statistics, updated_statistics)

    def adjust_stack(self, actions_probabilities: torch.Tensor, batch_mask: torch.Tensor):
        """
        Adjust the stack using the provided probabilities for enstack, no-op, and destack.
        Enstack is element 0, no-op is element 1, destack is element 2.

        :param actions_probabilities: Tensor of shape (*batch_shape, 3) representing the probabilities for
                                      enstack, no-op, and destack actions.
        :param batch_mask: Indicator for whether to update the stack and statistics for this batch.
        """
        if actions_probabilities.shape[-1] != 3 or actions_probabilities.shape[:-1] != self.batch_shape:
            raise ValueError(f"Action Probabilities must have shape ({[*self.batch_shape, 3]}) ")
        if batch_mask.shape != self.batch_shape:
            raise ValueError("Batch mask must match batch shape")

        # Record statistics if allowed by batch_mask
        self.record_statistics(actions_probabilities, self.pointers, batch_mask)

        # Apply the mask to the entire stack adjustment process.

        enstack, no_op, destack = actions_probabilities.unbind(-1)
        enstack_diagonals = enstack.unsqueeze(-1).expand([-1] * enstack.dim() + [self.stack_depth - 1])
        no_op_diagonals = no_op.unsqueeze(-1).expand([-1] * no_op.dim() + [self.stack_depth])
        destack_diagonals = destack.unsqueeze(-1).expand([-1] * destack.dim() + [self.stack_depth - 1])

        no_op_diagonals[..., 0] += destack
        no_op_diagonals[..., -1] += enstack

        transition_matrix = torch.diag_embed(enstack_diagonals, offset=-1)
        transition_matrix += torch.diag_embed(no_op_diagonals, offset=0)
        transition_matrix += torch.diag_embed(destack_diagonals, offset=1)

        pointers = self.pointers.movedim(0, -1)  # (...batch_shape, stack_depth)
        pointers = pointers.unsqueeze(-1)  # (batch_shape, stack_depth, 1)
        pointers = torch.matmul(transition_matrix, pointers).squeeze(-1)  # (...batch_shape, stack_depth)
        pointers = pointers.movedim(-1, 0) #(stack_depth, ...batch_shape)

        self.pointers = torch.where(batch_mask.unsqueeze(0), self.pointers, pointers)

    def get_expression(self) -> torch.Tensor:
        """
        Get the current expression of the stack by weighting with probabilistic pointers.

        :return: Tensor of shape embedding_shape, representing the current weighted expression of the stack.
        """
        weighted_stack = self.stack * self.pointers.unsqueeze(-1)
        return weighted_stack.sum(dim=0)

    def set_expression(self, embedding: torch.Tensor, batch_mask: torch.Tensor):
        """
        Sets the current stack level using an interpolation of probabilities.
        :param embedding: Tensor of shape (*batch_shape, d_model).
        :param batch_mask: Tensor that indicates whether to update the stack for this batch.
                           A value of True meant mask, false allows update.
        """
        if embedding.shape != self.stack.shape[1:]:
            raise ValueError("Embedding must match stack shape.")
        if batch_mask.shape != self.batch_shape:
            raise ValueError("Batch mask must match batch shape")

        # Add stack dim
        # Add embedding dim
        batch_mask = batch_mask.unsqueeze(0).unsqueeze(-1)

        # Create updated pointers by interpolation.
        pointers = self.pointers.unsqueeze(-1)
        updated_stack = (1 - pointers) * self.stack + pointers * embedding.unsqueeze(0)

        # Incorporate update. Do not change masked
        self.stack = torch.where(batch_mask, self.stack, updated_stack)

    def __call__(self, embedding: torch.Tensor, probabilities: torch.Tensor, batch_mask: torch.Tensor) -> torch.Tensor:
        """
        Performs adjust, set, get, and eval all in one go.
        :param embedding: The embedding to store somewhere or retrieve.
        :param probabilities: The probabilities for enstack, no-op, and destack.
        :param batch_mask: Tensor of shape (*batch_shape), either 0 or 1, indicating whether
                            to update the stack and statistics for this batch.
        :return: The extracted stack feature.
        """
        self.set_expression(embedding, batch_mask)
        self.adjust_stack(probabilities, batch_mask)
        return self.get_expression()
