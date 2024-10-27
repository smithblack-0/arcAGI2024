from typing import Any, Tuple, Optional

import torch
from torch import nn
from torch.nn import functional as F
from src.main.model.base import TensorTree, parallel_pytree_map


class PointerSuperpositionStack:
    """
    A differentiable stack designed to manage computation with probabilistic pointers for tasks involving
    adaptive computation. It enables stack-based subroutine management, where each level of the stack is
    weighted by probabilistic pointers, allowing smooth transitions between subroutine states.

    The stack accumulates decisions about enstack (push), destack (pop), or no-op at each level based on
    action probabilities provided by the external model. It tracks statistics on these transitions over
    time for further analysis. Multiple things can be tracked in parallel

    Updates for stack adjustment, statistics accumulation, and embeddings are controlled using a `batch_mask`.
    The `batch_mask` is a boolean tensor where a value of `1` indicates that no updates should occur for
    that batch, while a value of `0` allows updates. This is useful for controlling when updates occur
    during adaptive computation, especially when some batches need to be skipped.
    """

    def check_is_sane(self, tensor: Any):
        if not isinstance(tensor, torch.Tensor):
            raise ValueError('`tensor` must be a `torch.Tensor`')
        if tensor.device != self.device:
            raise ValueError('`tensor` must have the same device')
        if tensor.dtype != self.dtype:
            raise ValueError('`tensor` must have the same dtype')
        if tensor.shape[:self.batch_length] != self.batch_shape:
            raise ValueError('`tensor` must have the same inital shape as batch shape')

    def __init__(self,
                 stack_depth: int,
                 batch_shape: torch.Size,
                 action_projector: nn.Linear,
                 focus_projector: nn.Linear,
                 defaults: TensorTree,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        defaults = tuple(defaults)


        # Store pertinent information
        self.stack_depth = stack_depth
        self.batch_shape = batch_shape
        self.batch_length = len(batch_shape)
        self.dtype = dtype
        self.device = device
        self.post_init_done = True
        self.num_invoked = 0

        # Create the defaults and the stack features.
        def setup_stack(default: Any) -> torch.Tensor:
            self.check_is_sane(default)
            return torch.stack([default] * self.stack_depth, dim=0)

        self.defaults = parallel_pytree_map(setup_stack, defaults)
        self.stack = parallel_pytree_map(lambda x: x.clone(), self.defaults)

        # Setup the probability pointers
        # and the probability mass holder

        self.pointers = torch.zeros([stack_depth, *self.batch_shape],
                                    device=device, dtype=dtype)
        self.pointer_prob_masses = torch.zeros([stack_depth, *self.batch_shape],
                                               device=device, dtype=dtype)
        self.pointers[0] = 1.0

        # Store the two helper layers
        self.action_projector = action_projector
        self.focus_projector = focus_projector
    def get_statistics(self)->torch.Tensor:
        return self.pointer_prob_masses

    def adjust_stack(self,
                     embedding: torch.Tensor,
                     batch_mask: torch.Tensor,
                     max_iterations: int,
                     min_iterations: int
                     ) -> torch.Tensor:
        """
        Adjust the stack using information extracted from the
        provided embedding. Does not perform any adjustment when
        masked.

        :param embedding: An embedding of shape (...batch_shape, d_model)
        :param batch_mask: Indicator for whether to update the stack and statistics for this batch.
        :param max_iterations: Once this is exceeded, we only allow destacking
        :param min_iterations: Until here, we do not allow popping off the stack.
        """

        # Compute the action probabilities and focus behavior. This will be used
        # shortly to adjust the stack.

        action_logits = self.action_projector(embedding)
        focus_logits = self.focus_projector(embedding)

        if self.num_invoked >= max_iterations:
            # Exceeding max iterations. Only thing that can happen now
            # is destacking
            action_logits[..., 0] = -1e9
            action_logits[..., 1] = -1e9

        actions_probabilities = torch.softmax(action_logits, dim=-1)
        sharpening = 1 + F.elu(focus_logits).squeeze(-1)

        # Create a set of probabilities that can each
        # be viewed as associated with the enstack,
        # no op, or destack dimension. By rolling,
        # we move the pointers one higher or one lower.

        new_pointers = []
        for shift in [1, 0, -1]:
            new_pointers.append(self.pointers.roll(shifts=shift, dims=0))
        new_pointers = torch.stack(new_pointers, dim=-1)

        # We have some probability that is going to be lost off the top
        # of the stack and the bottom of the stack. Under some conditions
        # this is not allowed. Scavenge that probability

        # Prevents going off the end of the stack. Instead we accumulate by the end
        new_pointers[-1, ..., 0] += new_pointers[0, ..., 0]
        new_pointers[0, ..., 0] = 0

        # Not allowed to pop off the beginning yet.
        if self.num_invoked <= min_iterations:
            new_pointers[0, ..., 0] = new_pointers[-1, ..., 0]
            lost_probability = torch.zeros_like(new_pointers[-1, ..., 0])
        else:
            lost_probability = new_pointers[-1, ..., 0].clone()
        new_pointers[-1, ..., 0] = 0

        ##
        # We combine the various probability cases. We then also
        # sharpen the probabilities, and renormalize. We now
        # actually have the new pointers!
        ##

        new_pointers = torch.sum(new_pointers * actions_probabilities.unsqueeze(0), dim=-1)
        new_pointers = new_pointers ** sharpening.unsqueeze(0)
        new_pointers = new_pointers / new_pointers.sum(dim=0, keepdim=True)

        # Now, there is erasure. Anywhere that we destacked, we needed
        # to erase the prior contents proportional to how much
        # we destacked as we go back.

        erasure_probabilities = actions_probabilities[..., 2].unsqueeze(0)
        erasure_probabilities = self.pointers * erasure_probabilities

        def erase_stack(stack_case: torch.Tensor, default_case: torch.Tensor) -> torch.Tensor:
            ##
            # Perform the erasure action by
            # interpolating using the erasure probabilities
            # between the stack and the default state
            #

            # Set up all broadcasting
            sub_erase_probabilities = erasure_probabilities
            mask_case = batch_mask
            while sub_erase_probabilities.dim() < stack_case.dim():
                sub_erase_probabilities = sub_erase_probabilities.unsqueeze(-1)
                mask_case = mask_case.unsqueeze(-1)

            # Perform the interpolation, and return the update.
            # Do not update where masked.
            update = stack_case * (1 - sub_erase_probabilities) + default_case * sub_erase_probabilities
            return torch.where(~mask_case, update, stack_case)

        # Compute the updated statistics
        updated_prob_mass = self.pointer_prob_masses + new_pointers
        self.pointer_prob_masses = torch.where(~batch_mask, updated_prob_mass, self.pointer_prob_masses)

        # Commit results, and return.
        self.stack = parallel_pytree_map(erase_stack, self.stack, self.defaults)
        self.pointers = new_pointers

        return lost_probability

    def get_expression(self) -> Tuple[TensorTree]:
        """
        Get the current expression of the stack by weighting with probabilistic pointers.

        :return: Storage consisting of the expressed stack contents.
        """

        def weighted_sum(stack_case: torch.Tensor) -> torch.Tensor:
            pointers = self.pointers
            while pointers.dim() < stack_case.dim():
                pointers = pointers.unsqueeze(-1)
            weighted_stack = stack_case * pointers
            return weighted_stack.sum(dim=0)

        return parallel_pytree_map(weighted_sum, self.stack)

    def set_expression(self, batch_mask: torch.Tensor, *tensors: TensorTree):
        """
        Sets the current stack level using an interpolation of probabilities.
        :param batch_mask: Tensor that indicates whether to update the stack for this batch.
                           A value of True meant mask, false allows update.
        :param tensors: The tensors to set using differentiable logic.
        """

        if batch_mask.shape != self.batch_shape:
            raise ValueError("Batch mask must match batch shape")

        def update_stack(stack_case: torch.Tensor, tensor_case: torch.Tensor) -> torch.Tensor:
            ##
            # Updates the stack. Uses the halting probabilities to distribute an interpolation
            # based on the given stack case and tensor case.
            ##

            self.check_is_sane(tensor_case)

            # Do all setup for broadcasting. We must account
            # for any missing stack dimensions, and expand
            # probabilities to match. We also need to unsqueeze
            # the mask case to match.

            tensor_case = tensor_case.unsqueeze(0)
            mask_case = batch_mask
            pointers = self.pointers
            while pointers.dim() < tensor_case.dim():
                pointers = pointers.unsqueeze(-1)
                mask_case = mask_case.unsqueeze(-1)

            # Perform the interpolation, and only return an update where
            # valid
            update = stack_case * (1 - pointers) + tensor_case * pointers
            return torch.where(~mask_case, update, stack_case)

        # Incorporate update. Do not change masked
        self.stack = parallel_pytree_map(update_stack, self.stack, tensors)

    def __call__(self,
                 embedding: torch.Tensor,
                 batch_mask: torch.Tensor,
                 max_iterations: int,
                 min_iterations: int,
                 *tensors: TensorTree
                 ) -> Tuple[Tuple[TensorTree, ...], torch.Tensor]:
        """
        The actual invocation. Consists of using the action probabilities
        to perform stack updates, and then return the resutls
        :param embedding: An embedding to build data with. Shape (...batch_shape, d_model)
        :param batch_mask: Boolean mask. Indicates whether to update the stack for this batch element
        :param tensors: The tensors to store away
        :return: A tuple of two things
            - Tuple: The stack contents for the provided *tensors
            - tensor: Shape (...batch_shape). The probability that fell off the end of the stack.
        """

        self.set_expression(batch_mask, *tensors)
        lost_probability = self.adjust_stack(embedding, batch_mask, max_iterations, min_iterations)
        return self.get_expression(), lost_probability

class StackFactory(nn.Module):
    """
    Creates a pointer superposition stack when invoked with
    defaults and an embedding. The main method of creating
    pointer superposition stacks.
    """
    def __init__(self,
                 stack_size: int,
                 d_model: int,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None
                 ):
        super().__init__()

        # Store information
        self.stack_size = stack_size
        self.d_model = d_model
        self.dtype = dtype
        self.device = device

        # Setup projectors
        self.action_projectors = nn.Linear(d_model, 3)
        self.focus_projector = nn.Linear(d_model, 1)

    def forward(self, embedding: torch.Tensor, *defaults: TensorTree)->PointerSuperpositionStack:
        """
        Factory method to create a pointer superposition stack
        :param embedding: The embedding to create with
        :param defaults: The defaults and stack to setup with
        :return: A setup pointer superposition stack
        """
        batch_shape = embedding.shape[:-1]
        return PointerSuperpositionStack(self.stack_depth,
                                         batch_shape,
                                         self.action_projectors,
                                         self.focus_projector,
                                         defaults,
                                         self.dtype,
                                         self.device
                                         )