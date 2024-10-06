import torch
from torch import nn
from typing import List, Dict, Tuple, Any, Protocol, TypeVar, Union, Optional, Callable
from abc import ABC, abstractmethod

TensorTree = Union[
    torch.Tensor,  # Base case: Tensor
    List['TensorTree'],  # Recursion for lists
    Tuple['TensorTree', ...],  # Recursion for tuples
    Dict[str, 'TensorTree']  # Recursion for dictionaries
]

StackTree = Union['DifferentiableStack',
                 List['StackTree'],
                 Tuple['StackTree', ...],
                 Dict[str, 'StackTree']]
class StackCore(nn.Module, ABC):
    """
    The interface for the computational engine that can be
    put inside a stack driver. It provides the parameters
    to use the state of the stack to perform computations.
    """
    @abstractmethod
    def setup_state(self, tensor: torch.Tensor)->TensorTree:
        """
        Sets up state based on the provided tensor of embeddings
        :param tensor: The tensor of embeddings
        :return: Whatever state we need. Can be none.
        """

    @abstractmethod
    def forward(self, tensor: torch.Tensor, states: TensorTree)->Tuple[torch.Tensor, TensorTree]:
        """
        Performs the forward pass. Tensor is a tensor of embeddings, while states is any
        state information that needs to be tracked.
        :param tensor:
        :param states:
        :return:
        """
        pass

class DifferentiableSubroutineStack:
    """
    A differentiable stack for handling subroutine execution and residual bypass.

    This stack integrates both subroutine handling and residual normalization
    to perform complex, stateful computations in a neural network model. The
    stack allows for operations to occur in a superposition across multiple
    stack levels, controlled by a probabilistic stack pointer.

    **Actions**:
      1. **Enstack**: Initiates a subroutine, pushing a new tensor into the stack
         while moving the probabilistic stack pointer deeper. The new tensor
         defines the context for the subroutine execution.
      2. **Destack**: Exits a subroutine by retrieving and merging the subroutine's
         result back into the prior context. The probabilistic pointer shifts back
         up the stack.
      3. **No-op**: Maintains the current context, allowing operations without
         stack changes.

    **Stack Returns**:

    The **returns** generated during the **update** step represent the results of
    the stack computations, particularly during the **destack** action. When
    **destack** is chosen, features that "fall off" the top of the stack are
    captured. These features are returned along with the probability with which
    said feature was active.

    **Internal Mechanism**:
    The stack maintains two components:
      - **Tensors**: Each level of the stack holds a tensor.
      - **Probabilistic Pointer**: This pointer distributes attention across stack
        levels, indicating how much focus is on each level. The pointer allows for
        superposition, where multiple stack levels are active at once.

    - **Enstack**: Moves the pointer deeper into the stack and adds the new tensor
      to each stack level according to the probabilistic pointer.
    - **Destack**: Pops elements off the stack by shifting the pointer up and
      combining the retrieved context with the current state. The popped elements
      are the **returns** from the stack, representing completed computations.
    - **No-op**: Keeps the current state and adds any new residuals before
      normalizing.

    **Residual Bypass and Normalization**:
    Each action works with unnormalized tensors:
      - **Enstack**: Normalizes the incoming tensor and inserts it into deeper
        levels.
      - **Destack**: Combines the popped tensors with the current stack context
        and normalizes.
      - **No-op**: Adds the current tensor's residual to the active context and
        normalizes.

    **Stack Depth and Probabilistic Pointer**:
    The stack has a fixed depth, and the probabilistic pointer, represented by
    probabilities for each level, controls the depth of the stack’s focus. The
    stack’s operations are vectorized to support efficient tensor manipulations.
    Operations update the tensor across active stack levels and shift the pointer
    as needed to control the flow of subroutine calls.

    The stack depth is always the first dimension of the tensor, facilitating easy
    integration into complex tensor operations.
    """

    @property
    def stack_depth(self) -> int:
        return self.probabilities.shape[0]

    @property
    def stack_probabilities(self) -> torch.Tensor:
        return self.probabilities.sum(dim=0)

    @property
    def stack_empty(self):
        return torch.all(self.stack_probabilities < 1e-4)

    def __init__(self,
                 probabilities: torch.Tensor,
                 stack: torch.Tensor,
                 position_markers: torch.Tensor,
                 layernorm: nn.LayerNorm,
                 min_iterations_before_dequeue: int,
                 max_iterations_before_flush: int
                 ):
        """
        The setup for the differentiable stack
        :param probabilities: The stack probabilities
            - Shape (depth, ...)
        :param stack: The stack being managed
            - Shape (depth, ..., d_model)
        :param position_markers: The position markers for each stack level
            - Shape (depth, ...,  d_model)
        :param layernorm: The layernorm used to manage residuals
        :param min_iterations_before_dequeue: The number of times the
            stack must have been accessed before dequeue is allowed.
        :param max_iterations_before_flush:
            The maximum number of iterations that can occur before the stack is forced into
            a flush condition that will inevitably terminate.
        """
        assert probabilities.shape == stack.shape[:-1]

        # Monitoring
        self.action_statistics = torch.zeros(list(probabilities.shape[1:]) + [3], device=probabilities.device)

        # Main
        self.probabilities = probabilities
        self.stack = stack
        self.position_markers = position_markers
        self.layernorm = layernorm

        # Training and constants
        self.num_times_invoked = 0
        self.max_times_before_flush = max_iterations_before_flush
        self.destack_threshold = min_iterations_before_dequeue

    def get(self) -> torch.Tensor:
        """
        Retrieves the current expression of the stack based on the probabilistic pointer.

        This method computes a weighted average of tensors across all stack levels,
        determined by the probabilistic pointer, and returns the combined result.

        :return: The current stack superposition (weighted combination of stack tensors)
            - Shape (..., d_model)
        """

        # Setup the probabilities for matrix multiplication. Do the same for the stack

        probabilities = self.probabilities.movedim(0, -1).unsqueeze(0)  # (..., depth, 1)
        stack = self.stack.movedim(0, -1)  # (..., d_model, depth)

        # Run superposition. Return
        return torch.matmul(probabilities, stack).squeeze(-1)

    def create_action_probabilities(self, action_logits: torch.Tensor) -> torch.Tensor:
        """
        Generates action probabilities from the provided logits and applies masking.

        The logits for enstack, no-op, and destack actions are processed into probabilities,
        while ensuring that invalid actions (e.g., enstacking off the edge of the stack) are masked out.
        Additionally, destacking is restricted based on the minimum iterations before dequeue, and once
        the maximum number of iterations is exceeded, only the destack option is allowed.

        :param action_logits: Logits for the actions (depth, ..., 3) for enstack, no-op, destack.
        :return: Action probabilities, masked appropriately for each stack level (depth, ..., 3).
        """

        # Expand the action probabilities to be in syncronization with the internal probabilities.
        # In particular, to have depth. This will let us set restrictions when attempting to enqueue off
        # the edge of the stack, or dequeue off it when not yet allowed

        expansion = [-1] * action_logits.dim()
        expansion.insert(0, self.stack_depth)
        action_logits = action_logits.unsqueeze(0)  # (1, ..., 3)
        action_logits = action_logits.expand(*expansion)  # (depth, ..., 3)

        # Apply enstack masking. You are not allowed to enstack off the edge of the
        # of the stack. Any attempt to do so will instead result in probabilities of
        # zero for that action

        action_logits[self.stack_depth - 1, ..., 2] = -float('inf')

        # Apply destack masking. In order to force the model to train to use the stack,
        # it is possible to restrict the model from emitting output until a certain
        # amount of probability has been enqueued. This performs that masking

        destack_logits = action_logits[0, ..., 0]
        destack_logits = torch.where(self.num_times_invoked >= self.destack_threshold,
                                     destack_logits,
                                     -float('inf')
                                     )
        action_logits[0, ..., 0] = destack_logits

        # If the flush threshold has been exceeded, our only allowed option is to dequeue.
        # We mask everything else in that situation

        if self.num_times_invoked > self.max_times_before_flush:
            action_logits[..., 1] = -float('inf')
            action_logits[..., 2] = -float('inf')

        # Create probabilities.

        probabilities = torch.softmax(action_logits, dim=-1)

        # Return the probabilities for each portion of the stack.

        return probabilities

    def compute_enstack(self,) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Executes the enstack action, moving the probabilistic pointer deeper.

        This operation shifts the probabilistic pointer into a deeper stack level and modifies the
        tensor states accordingly based on the enstack probability.  It returns the new
        state under these modifications, ready to be weighted.

        :return: The new stack probabilities and the new stack state.
        """

        probabilities = self.probabilities  # (depth, ...)
        stack = self.stack  # (depth, ..., d_model)
        zero_probs = torch.zeros_like(probabilities[0])

        # Shift all probabilities over. Retain only based on the
        # strength of the action. We are effectively moving the stack pointer over

        probabilities = torch.concat([zero_probs, probabilities[:-1]], dim=-1)

        # Return

        return probabilities, stack

    def compute_destack(self,) -> Tuple[torch.Tensor,
                                   Tuple[torch.Tensor, torch.Tensor]]:
        """
        Executes the destack action, retrieving and combining previous context.

        This operation shifts the probabilistic pointer upward, effectively popping elements
        off the stack. The tensors that fall off are returned and can be processed further in
        the model.

        :return:
            - The destacked (lost) probability
            - The updated stack (new probabilities, new stack).
        """

        # The destack action is perhaps the most confusing action
        # that can occur within this class. Three things need to happen.
        #
        # First, erasure. When finishing up the current subroutine, its
        # context should be erased from the stack. This is a memory cleanup
        # step
        #
        # Second, pointer shift. We shift the probabilistic pointers up the
        # stack by one,
        #
        # Third, the probability at the beginning of the stack will be lost.
        # It is actually corrolated with our return. We capture that for use

        probabilities = self.probabilities  # (depth, ...)
        stack = self.stack  # (depth, ..., d_model)
        zero_probs = torch.zeros_like(probabilities[0])

        # Erase the stack subroutine context based on how active it is.
        # This gets rid of the context used by the subroutine, freeing up that for
        # future stack actions

        stack = stack*(1-probabilities.unsqueeze(-1)) + torch.zeros_like(stack)*probabilities.unsqueeze(-1)

        # Catch the lost probability
        output_probabilities = probabilities[0]

        # Move probabilistic pointer up the stack by one
        probabilities = torch.concat([probabilities[1:], zero_probs], dim=0)

        # Return results

        return output_probabilities, (probabilities, stack)

    def compute_no_op(self):
        """
        Executes the no-op action, maintaining the current context.

        The no-op action retains the current stack state, performing no changes to the stack pointer
        or tensor. However, it will still result in incorporating residuals and maintaining the
        current context.

        :param action_probabilities: The probability of performing the no-op action (depth, ...).
        :return: The updated stack probabilities and the unchanged stack state.
        """

        return self.probabilities, self.stack

    def commit_tensor(self, tensor: torch.Tensor):
        """
        Commits a new tensor into the stack, applying residual updates and normalization.

        The provided tensor is combined with the current stack across all levels, based on the
        current probabilistic pointer. The updated stack is normalized to ensure numerical stability.

        :param tensor: The tensor to commit to the stack (Shape: (..., d_model)).
        """

        # Run all possible updates in parallel
        update = self.layernorm(tensor.unsqueeze(0) + self.stack)  # (depth, ..., d_model)

        # Update the stack only in proportion to how strong the element is active

        probabilities = self.probabilities.unsqueeze(-1)
        self.stack = self.stack * (1 - probabilities) + update * probabilities

    def update(self,
               action_logits: torch.Tensor,
               tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the stack by applying action logits and commits the new tensor state.

        The stack performs one of three actions (enstack, no-op, or destack) based on the logits,
        adjusting the probabilistic pointer accordingly. The method then normalizes the updated
        stack and returns the results from any destacked features that fell off the top of the stack.

        :param action_logits: Logits specifying the action for each stack level (depth, ..., 3).
        :param tensor: The new tensor to commit to the stack (Shape: (..., d_model)).
        :return:
            - The probabilities and tensor representing the destacked elements
                - Shape (...)
                - Shape (..., d_model)
        """

        # Get the action probabilities, and unbind it into each action
        action_probabilities = self.create_action_probabilities(action_logits)  # (depth, ..., 3)

        # Run each action. Get revised state and probability tensors

        enstack_probs, enstack_stack = self.compute_enstack()
        no_op_probs, no_op_stack = self.compute_no_op()
        output_probs, (destack_probs, destack_stack) = self.compute_destack()

        # Weight each outcome by its action probability. We get the updated pointers and
        # stack.

        probabilistic_pointers = torch.stack([enstack_probs, no_op_probs, destack_probs], dim=-1) #(depth, ..., 3)
        stack = torch.stack([enstack_stack, no_op_stack, destack_stack], dim=-1) #(depth, ..., d_model, 3)

        probabilistic_pointers = (probabilistic_pointers*action_probabilities).sum(dim=-1)
        stack = (stack*action_probabilities.unsqueeze(-2)).sum(dim=-1)

        # Finally, we are ready to integrate the tensor! We emplace it with a strength based
        # on how strong each probabilistic pointer is. Also, we include the position markers
        # while we are at it. Then we layernorm
        #
        # If the pointers are focused on one stack level, this acts just like a normal add+layernorm.
        # Well, except for the position markers, of course.

        update = tensor.unsqueeze(0) + stack + self.position_markers
        update = self.layernorm(update)
        stack = stack*(1-probabilistic_pointers.unsqueeze(-1))+update*probabilistic_pointers.unsqueeze(-1)

        # Updates! Store, and scatter the tensor into their positions based on the
        # probabilistic pointers.

        self.probabilities = probabilistic_pointers
        self.stack = stack
        self.action_statistics += action_probabilities
        self.num_times_invoked += 1
        self.commit_tensor(tensor)

        # Some percentage of the tensor we are given is actually being lost!
        # That happens in the destack action, when some pointer probability rolll off
        # the top of the stack. This percentage is actually the return,
        # We return the tensor with it's associated output probability.

        return output_probs, tensor


class SubroutineStackFactory(nn.Module):
    """
    The differentiable stack class used with the model
    """

    def __init__(self,
                 stack_depth: int,
                 d_model: int, ):
        super().__init__()

        self.stack_depth = stack_depth
        self.d_model = d_model

        self.stack_markers = nn.Parameter(torch.zeros([stack_depth, d_model]))
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self,
                tensor: torch.Tensor,
                min_iterations_before_destack: int,
                computation_limit: int
                ) -> DifferentiableSubroutineStack:
        """
        Creates a differentiable stack on demand.
        :param tensor: The tensor to build a stack around
        :param min_iterations_before_dequeue: Used during training, forces the model
               to wait a bit before it is allowed to return content from the top of the stack
        :param computation_limit: How many iterations may occur before we get locked into only dequeing
        :return: The active differentiable stack
        """

        # Setup probabilities. We insert 100% of probability at position 0

        probabilities = torch.zeros([self.stack_depth] + list(tensor.shape[:-1]),
                                    device=tensor.device, dtype=tensor.dtype)
        probabilities[0] = 1.0

        # Setup stack. Insert all context at position 1

        stack = torch.zeros([self.stack_depth] + list(tensor.shape), device=tensor.device, dtype=tensor.dtype)
        stack[0] = tensor

        # Setup position markers
        stack_pos_markers = self.stack_markers
        while stack_pos_markers.dim() < stack.dim():
            stack_pos_markers = stack_pos_markers.unsqueeze(1)

        # Return stack

        return DifferentiableSubroutineStack(probabilities,
                                             stack,
                                             stack_pos_markers,
                                             self.layernorm,
                                             min_iterations_before_destack,
                                             computation_limit)

class StackDriver:
    """
    A core stack driver layer designed to allow the usage
    of the differentiable stack factory automatically with an
    internal core computation layer.
    """

    @classmethod
    def parallel_pytree_map(cls, func: Callable[..., Any], *pytrees: Any) -> Any:
        """
        Recursively applies a function to corresponding leaves of multiple pytrees with the same structure.

        Args:
            func (Callable[..., Any]): A function to apply to corresponding leaves of the pytrees.
            *pytrees (NestedTensor): Multiple pytrees with the same structure.

        Returns:
            NestedTensor: A new pytree with the function applied to corresponding leaves.
        """
        # Check if all pytrees are lists, tuples, or dicts
        if all(isinstance(pytree, list) for pytree in pytrees):
            return [cls.parallel_pytree_map(func, *elems) for elems in zip(*pytrees)]
        elif all(isinstance(pytree, tuple) for pytree in pytrees):
            return tuple(cls.parallel_pytree_map(func, *elems) for elems in zip(*pytrees))
        elif all(isinstance(pytree, dict) for pytree in pytrees):
            return {key: cls.parallel_pytree_map(func, *(pytree[key] for pytree in pytrees))
                    for key in pytrees[0]}
        else:
            # These are leaves, apply the function to them
            return func(*pytrees)

    def __init__(self,
                 d_model: int,
                 stack_depth: int,
                 core: StackCore
                 ):
        self.actions = nn.Linear(d_model, 3)
        self.stack_factory = SubroutineStackFactory(stack_depth, d_model)
        self.core = core

    def initialize_stacks(self,
                          tensor: torch.Tensor,
                          state: TensorTree,
                          min_iterations_before_destack: int,
                          max_iterations_before_flush: int,
                          )->Tuple[DifferentiableSubroutineStack, StackTree]:
        """
        Initializes the stacks using the tensor and the state, for the subsequent
        computation.

        :param tensor: The tensor to initialize with
        :param state: The state to initialize with
        :param min_iterations_before_destack: The number of iterations to wait until stack removal is allowed
        :param max_iterations_before_flush: The maximum number of iterations before the stack is forced to flus
                                            its contents.
        :return:
            - The tensor stack
            - The state stack
        """
        setup_stack = lambda x: self.stack_factory(x, min_iterations_before_destack, max_iterations_before_flush)
        tensor_stack = self.stack_factory(tensor, min_iterations_before_destack, max_iterations_before_flush)
        state_stack = self.parallel_pytree_map(setup_stack, state)
        return tensor_stack, state_stack

    def initialize_accumulators(self,
                                tensor: torch.Tensor,
                                state: TensorTree
                                )->Tuple[torch.Tensor, TensorTree]:
        """
        Initialize the output accumulators as well.
        :param tensor: The tensor whose shape we accumulat
        :param state: The state we will accumulate
        :return: The setup accumulators
        """
        setup_accumulator = lambda x: torch.zeros_like(x)
        tensor_accumulator = torch.zeros_like(tensor)
        state_accumulator = self.parallel_pytree_map(setup_accumulator, state)
        return tensor_accumulator, state_accumulator

    def forward(self,
                tensor: torch.Tensor,
                max_computation_iterations: int,
                state: Optional[TensorTree] = None,
                min_iterations_before_destack: int = 0
                )->Tuple[torch.Tensor, TensorTree]:
        """
        Performs the stack drive action.

        :param tensor: The tensor of, presumably, embeddings to process
        :param max_computation_iterations:
            - The number of iterations to wait until the stack is forced to flush.
        :param state: Any state that is available.
        :param min_iterations_before_destack: AS it says
        :return: The embedding output, and any new state
        """
        # Setup state if needed
        if state is None:
            state = self.core.setup_state(tensor)
            assert state is not None


        # Setup features
        tensor_stack, state_stack = self.initialize_stacks(tensor, state,
                                                           min_iterations_before_destack,
                                                           max_computation_iterations)
        tensor_accumulators, state_accumulators = self.initialize_accumulators(tensor, state)

        # Drive solution. We
        while not tensor_stack.stack_empty:

            # Get the current embeddings, state
            get_from_diff_stack = lambda x : x.get()
            embeddings = tensor_stack.get()
            state = self.parallel_pytree_map(get_from_diff_stack, state_stack)

            # Run using core, and compute stack actions

            embeddings, state = self.core(embeddings, state)
            actions = self.actions(embeddings)

            # Update stacks, get outputs

            update_action = lambda stack, update : stack.update(actions, update)
            tensor_outcome = tensor_stack.update(actions, embeddings) # Tuple(probs, embeddings)
            state_outcome = self.parallel_pytree_map(update_action, state_stack, state)

            # Use outputs to update accumulators
            def accumulate_action(accumulator: torch.Tensor,
                                  update: Tuple[torch.Tensor, torch.Tensor]
                                  ):
                probs, embeddings = update
                accumulator += embeddings*probs.unsqueeze(-1)

            accumulate_action(tensor_accumulators, tensor_outcome)
            self.parallel_pytree_map(accumulate_action, state_accumulators, state_outcome)

        return tensor_accumulators, state_accumulators






