from typing import Any, Tuple, Optional, Dict, Union

import torch
from torch import nn
from torch.nn import functional as F
from src.main.model.base import TensorTree, parallel_pytree_map, DropoutLogits
from src.main.model.computation_support_stack.abstract import (stack_controller_registry,
                                                               AbstractSupportStack,
                                                               AbstractControlGates,
                                                               AbstractStackController,
                                                               AbstractStackFactory,
                                                               BatchShapeType)

class PointerSuperpositionStack(AbstractSupportStack):
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
    def __init__(self,
                 pointers: torch.Tensor,
                 pointer_prob_mass: torch.Tensor,
                 stack: Dict[str, TensorTree],
                 defaults: Dict[str, TensorTree],
                 post_init_flag: bool = False
                 ):
        # Super initialization
        stack_depth = pointers.shape[0]
        batch_shape = pointers.shape[1:]
        dtype = pointers.dtype
        device = pointers.device

        super().__init__(stack_depth, batch_shape, dtype, device)

        # Store tensors
        self.pointers = pointers
        self.pointer_prob_masses = pointer_prob_mass
        self.defaults = defaults
        self.stack = stack

        # Store flag
        self.post_init_flag = post_init_flag

    def save_state(self) -> Tuple[TensorTree, Optional[Any]]:
        """
        Saves the state to be in terms of a tensor tree
        and a bypass feature
        :return:
            - Tensor Tree: The internal state
            - Bypass: The bypass features
        """
        save_package = (
            self.pointers,
            self.pointer_prob_masses,
            self.defaults,
            self.stack
        )
        return save_package, self.post_init_flag

    @classmethod
    def load_state(cls, pytree: TensorTree, bypass: Any) -> 'PointerSuperpositionStack':
        """
        Loads the state back into memory
        :param pytree: The state being loaded
        :param bypass: The flag
        :return: A setup instance
        """
        return cls(*pytree, bypass)
    def get_statistics(self) -> Dict[str, torch.Tensor]:
        """
        Returns some important tensor statistics.
        :return: The statistics
            - probability mass
        """
        statistics = {}
        statistics["probability_mass"] = self.pointer_prob_masses
        return statistics

    def adjust_stack(self,
                     controls: Tuple[torch.Tensor, torch.Tensor],
                     batch_mask: torch.Tensor,
                     ) -> torch.Tensor:
        """
        Adjust the stack using information extracted from the
        provided embedding. Does not perform any adjustment when
        masked.

        :param controls: The control tensors
            - Action probabiilities of shape (..., 3)
            - sharpening factor of shape (..., 1)
        :param batch_mask: Indicator for whether to update the stack and statistics for this batch.
        """
        actions_probabilities, sharpening = controls

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

        # Prevents going off the end of the stack.
        # Instead, we accumulate at the end, or the beginning.
        new_pointers[-1, ..., 0] += new_pointers[0, ..., 0]
        new_pointers[0, ..., 0] = 0

        new_pointers[0, ..., 2] += new_pointers[-1, ..., 2]
        new_pointers[-1, ..., 2] = 0

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

    def pop(self, name: Optional[str] = None) -> Union[Dict[str, TensorTree], TensorTree] :
        """
        Get the current expression of the stack by weighting with probabilistic pointers.
        :param name: If provided, only the indicated
             kwarg will be popped
        :return: Storage consisting of the expressed stack contents.
                 Will be either all the kwargs, or onlu the name indicated
        """
        def weighted_sum(stack_case: torch.Tensor) -> torch.Tensor:
            pointers = self.pointers
            while pointers.dim() < stack_case.dim():
                pointers = pointers.unsqueeze(-1)
            weighted_stack = stack_case * pointers
            return weighted_stack.sum(dim=0)
        if name is not None:
            return parallel_pytree_map(weighted_sum, self.stack[name])
        return parallel_pytree_map(weighted_sum, self.stack)

    def push(self, batch_mask: Optional[torch.Tensor], **states):
        """
        Sets the current stack level using an interpolation of probabilities.
        :param batch_mask: Tensor that indicates whether to update the stack for this batch.
                           A value of True meant mask, false allows update.
        :param tensors: The tensors to set using differentiable logic.
        """

        def update_stack(stack_case: torch.Tensor, tensor_case: torch.Tensor) -> torch.Tensor:
            ##
            # Updates the stack. Uses the halting probabilities to distribute an interpolation
            # based on the given stack case and tensor case.
            ##

            self.is_tensor_sane(tensor_case, "during stack update")

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
        self.stack = parallel_pytree_map(update_stack, self.stack, states)



class GateControls(AbstractControlGates):
    """
    Computes the control features
    needed to implement the model.
    """

    def __init__(self,
                 d_model: int,
                 dtype: torch.dtype,
                 device: torch.device,
                 control_dropout: float
                 ):
        super().__init__(d_model)

        # Setup projectors
        self.action_projector = nn.Linear(d_model, 3, dtype=dtype, device=device)
        self.focus_projector = nn.Linear(d_model, 1, dtype=dtype, device=device)

        # Setup dropout
        self.dropout_logits =DropoutLogits(control_dropout)

    def forward(self, control_embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Runs the forward mechanism that produces the control features
        :param control_embedding: The control embedding to use in production
        :return:
            - Action probabilities: Probabilities of each action. (...batch_shape, 3)
            - Sharpening rates. Rates of sharpening. Shape (...batch_shape).
        """
        # Compute the action probabilities and focus behavior. This will be used
        # shortly to adjust the stack.

        action_logits = self.action_projector(control_embedding)
        focus_logits = self.focus_projector(control_embedding)

        action_logits = self.dropout_logits(action_logits)

        actions_probabilities = torch.softmax(action_logits, dim=-1)
        sharpening = 1 + F.elu(focus_logits).squeeze(-1)

        return actions_probabilities, sharpening


class StackFactory(AbstractStackFactory):
    """
    Creates a pointer superposition stack when invoked with
    defaults and an embedding. The main method of creating
    pointer superposition stacks.
    """

    def __init__(self,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None
                 ):

        super().__init__(dtype, device)

        # Store information
        self.dtype = dtype
        self.device = device
    def forward(self,
                batch_shape: BatchShapeType,
                stack_depth: int,
                **defaults: TensorTree
                ) -> AbstractSupportStack:
        """
        Sets up a working pointer superposition stack.
        :param batch_shape: The batch shape to match to
        :param stack_depth: The depth to make the stack to
        :param defaults: The default stack pytrees. We will assume stack locations
                         that are empty should look like this
        :return: The initialized stack.
        """

        # Set up the probability features, consisting of the pointers
        # and the pointer probability masses

        pointers = torch.zeros([stack_depth, *batch_shape],
                                    device=self.device, dtype=self.dtype)
        pointer_prob_masses = torch.zeros([stack_depth, *batch_shape],
                                               device=self.device, dtype=self.dtype)
        pointers[0] = 1.0

        # The stack, and the default values, also need to be setup.
        # Create a function that will add and extra

        def setup_stack(default: Any) -> torch.Tensor:
            default = self.is_tensor_sane(default, "default stack state", batch_shape)
            return torch.stack([default] * stack_depth, dim=0)

        defaults = parallel_pytree_map(setup_stack, defaults)
        stack = parallel_pytree_map(lambda x: x.clone(), defaults)
        return PointerSuperpositionStack(pointers, pointer_prob_masses, stack, defaults)

@stack_controller_registry.register("Default")
class StackController(AbstractStackController):
    """
    Implementation of the stack controller, including
    setup and usage mechanisms. We extend initialization
    to make sure we can create the needed factories
    """
    def __init__(self,
                 d_model: int,
                 dtype: torch.dtype,
                 device: torch.device,
                 control_dropout: float
                 ):
        stack_factory = StackFactory(dtype, device)
        gate_controls = GateControls(d_model, dtype, device, control_dropout)
        super().__init__(gate_controls, stack_factory)

