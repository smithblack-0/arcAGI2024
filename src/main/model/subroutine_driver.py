import torch
from torch import nn
from typing import List, Dict, Tuple, Any, Union, Optional, Callable
from abc import ABC, abstractmethod

from src.main.model.base import TensorTree,StatefulCore

StackTree = Union['DifferentiableStack',
                 List['StackTree'],
                 Tuple['StackTree', ...],
                 Dict[str, 'StackTree']]
class SubroutineCore(StatefulCore):
    """
    The interface for the computational engine that can be
    put inside a subroutine driver. It provides the parameters
    to use the state of the stack to perform computations. It also
    will produce embeddings and state output that can be used
    to manage the subroutine stack
    """
    @abstractmethod
    def setup_state(self, tensor: torch.Tensor)->TensorTree:
        """
        Sets up state based on the provided tensor of embeddings. Note that
        if you do not use state, just return an empty dict.

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



class ActionsManagement:
    """
    Class for managing action probabilities and tracking action statistics.

    This class tracks certain important statistics and constructs action
    probabilities with the proper masking. The probabilities are derived from
    logits corresponding to different stack actions (enstack, no-op, destack),
    ensuring proper masking based on predefined thresholds.

    It should be noted that the action logits are expected to be defined along
    the last channel in terms of the actions, and

    0: destack
    1: no op
    2: enstack

    """

    def __init__(self,
                 num_times_before_pop_allowed: int,
                 num_times_before_flush_begins: int,
                 stack_size: int,
                 action_statistics: torch.Tensor):
        """
        Initializes the ActionsManagement instance.

        :param num_times_before_pop_allowed: Minimum number of invocations before destack is allowed.
        :param num_times_before_flush_begins: Maximum number of invocations before forced flush begins.
        :param stack_size: The depth of the stack.
        :param action_statistics: Tensor for tracking statistics of the three actions.
        """
        self.num_times_before_pop = num_times_before_pop_allowed
        self.num_times_before_flush = num_times_before_flush_begins
        self.stack_size = stack_size
        self.action_statistics = action_statistics
        self.num_times_invoked = 0  # Always initialized as 0

    def __call__(self, action_logits: torch.Tensor) -> torch.Tensor:
        """
        Generates action probabilities from the provided logits and applies masking.

        The logits for enstack, no-op, and destack actions are processed into probabilities,
        while ensuring that invalid actions (e.g., enstacking off the edge of the stack) are masked out.
        Additionally, destacking is restricted based on the minimum iterations before dequeue, and once
        the maximum number of iterations is exceeded, only the destack option is allowed.

        :param action_logits: Logits for the actions (depth, ..., 3) for enstack, no-op, destack.
        :return: Action probabilities, masked appropriately for each stack level (depth, ..., 3).
        """
        # Copy action logits

        action_logits = action_logits.clone()

        # Expand the action probabilities to synchronize with the internal probabilities.
        # In particular, ensure depth alignment to handle restrictions on dequeuing and enqueueing.

        expansion = [1] * action_logits.dim()
        expansion.insert(0, self.stack_size)
        action_logits = action_logits.unsqueeze(0)  # (1, ..., 3)
        action_logits = action_logits.repeat(*expansion)  # (depth, ..., 3)

        # Apply enstack masking: enstack cannot occur beyond the stack's maximum depth.
        # Setting enstack logits at the max depth to -inf prevents enstack beyond that point.

        action_logits[self.stack_size - 1, ..., 2] = -float('inf')

        # Apply destack masking: destacking isn't allowed until enough invocations.
        # Set destack logits at depth 0 to -inf if the threshold isn't met.

        if not self.num_times_invoked >= self.num_times_before_pop:
            action_logits[0, ..., 0] = -float('inf')

        # Apply flush logic: If the flush threshold is exceeded, only destack is allowed.
        # Other actions (no-op, enstack) are masked by setting their logits to -inf.

        if self.num_times_invoked >= self.num_times_before_flush:
            action_logits[..., 1] = -float('inf')
            action_logits[..., 2] = -float('inf')

        # Create probabilities by applying softmax over the logits.

        probabilities = torch.softmax(action_logits, dim=-1)

        # Update action statistics.
        self.action_statistics += probabilities
        self.num_times_invoked += 1

        # Return the final action probabilities.
        return probabilities


class ProbabilisticPointers:
    """
    The probabilistic pointer management class. Tracks and modifies
    the probabilistic pointers.

    Probabilistic pointers distribute the focus across stack levels, allowing
    actions like `enstack`, `no-op`, and `destack` to adjust how attention is
    split across the levels of the stack.

    It handles rolling of the pointers based on the action probabilities and
    ensures that any probability that rolls off the end (or beginning) of the
    stack is masked out and discarded instead of wrapping around.
    """

    def __init__(self, probabilistic_pointers: torch.Tensor):
        """
        Initializes the probabilistic pointer class.

        :param probabilistic_pointers: The probabilistic pointers matching the
                                       situation. It has been assumed they were loaded already.
                                       Shape (stack_size, ...)
        """
        self.pointers = probabilistic_pointers

    def get(self) -> torch.Tensor:
        """
        Returns the current probabilistic pointers.

        :return: A tensor representing the current probabilistic pointers.
        """
        return self.pointers

    def change_superposition(self, action_probabilities: torch.Tensor) -> torch.Tensor:
        """
        Changes the superposition of the probabilistic pointers based on action probabilities.

        This method adjusts the pointers to reflect enstack, no-op, and destack actions.
        The enstack action shifts pointers deeper into the stack, destack shifts them earlier,
        and no-op maintains the current position.

        The lost probability is discarded and returned. This probability reflects the focus
        that rolls off the top or bottom of the stack based on the actions.

        :param action_probabilities: The action probabilities. Shape (stack_size, ..., 3)
        :return: The lost probability. The probability that is being lost off the end of the stack.
        """

        # Run the enstack, no_op, and destack actions. The enstack action
        # moves all pointers deeper into the stack, the destack moves
        # pointers earlier in the stack, and the no_op keeps pointers unchanged.

        # Setup pointers with action masks. Apply action probabilities across
        # each corresponding action: destack, no_op, and enstack.
        pointers = self.pointers.unsqueeze(-1)  # Shape (stack_size, ..., 1)
        pointers = pointers * action_probabilities  # Shape (stack_size, ..., 3)

        # Set aside the probability that is about to roll off the end of the stack
        # during the destack action. This value will be discarded.
        lost_probabilities = pointers[0, ..., 0].clone()  # Destack lost probability

        # For destack, we set the first element (the "top" of the stack) to zero
        # since it is about to be "pushed off" as we roll pointers to the left.
        # For enstack, we mask out the last element, as it is about to roll off
        # the "bottom" of the stack when shifted to the right

        pointers[0, ..., 0] = 0
        pointers[-1, ..., 2] = 0

        # Now we roll each set of pointers based on the action probabilities:
        # - Destack action rolls pointers earlier in the stack by one.
        # - No-op action leaves pointers unchanged.
        # - Enstack action rolls pointers deeper into the stack by one.

        # Destack: Roll pointers one position earlier in the stack.
        # No-op: Keep the pointers unchanged.
        # Enstack: Roll pointers one position deeper in the stack.

        new_pointers = []
        new_pointers.append(pointers[..., 0].roll(-1, dims=0))  # Roll to earlier in the stack for destack
        new_pointers.append(pointers[..., 1])  # No-op changes nothing
        new_pointers.append(pointers[..., 2].roll(1, dims=0))  # Enstack goes one deeper

        # Combine the updated pointer tensors into a single tensor.
        # Sum across the different actions (destack, no-op, enstack) to get the final updated pointers.

        pointers = torch.stack(new_pointers, dim=-1)  # Shape (stack_size, ..., 3)
        self.pointers = pointers.sum(dim=-1)  # Shape (stack_size, ...)

        # Return the lost probability that was discarded.
        return lost_probabilities

class SubroutineStateTracker:
    """
    Manages the context of state stacks allowing retrieval and
    insertion of new state based on the probabilistic pointers

    **Core Responsibilities**:

   The class is responsible for maintaining the stack context across subroutine
   calls. Get can then be used to superimpose them together into something
   that is differentiable, producing a differentiable call stack.

    - `stack`: The current stack setup, containing the context of subroutines and
               layers.

    **Methods**:
    1. **get(pointer_probabilities: torch.Tensor) -> torch.Tensor**:
       Retrieves the superimposed stack state based on the probabilistic pointer
       values. This method computes a weighted combination of the stack levels
       (based on the probabilistic pointers) to return the current context.
       - **Param**: `pointer_probabilities` (Tensor): The probabilistic pointers that
                    distribute attention across stack levels.
       - **Return**: The combined stack context, weighted by the pointer probabilities.

    2. **change_superposition(pointer_probabilities: torch.Tensor, action_probabilities: torch.Tensor)**:
       Modifies the stack based on the provided action probabilities. For the
       destack action, the current subroutine context is erased from the stack,
       cleaning up the virtual layer state.
       - **Param**: `pointer_probabilities` (Tensor): The probabilistic pointers that
                    represent the distribution across stack levels.
       - **Param**: `action_probabilities` (Tensor): Probabilities indicating the
                    action to take (enstack, no-op, or destack).

    3. **update(pointer_probabilities: torch.Tensor, tensor: torch.Tensor)**:
       Updates the stack by integrating the result of this level of subroutine execution with
       the current stack context. The class performs an add + layernorm operation to
       ensure smooth integration of the new tensor into the stack.
       - **Param**: `pointer_probabilities` (Tensor): The probabilities representing
                    the distribution across the stack levels.
       - **Param**: `tensor` (Tensor): The new output from the subroutine execution
                    to be integrated into the main stack.
    """
    def __init__(self,
                 stack: torch.Tensor,
                 ):
        """
        :param stack: The current stack setup
        """
        self.stack = stack

    def get(self, pointer_probabilities: torch.Tensor)->torch.Tensor:
        """
        Gets the stack, based on the provided pointer probabilities.
        :param pointer_probabilities: The poitner probabilities. Shape (stack_depth, ...)
        :return: The superimposed stack. Shape (..., d_model)
        """

        # Setup the probabilities for matrix multiplication. Do the same for the stack

        probabilities = pointer_probabilities.movedim(0, -1).unsqueeze(-1)  # (..., depth, 1)
        stack = self.stack.movedim(0, -1)  # (..., d_model, depth)

        # Run superposition. Return
        return torch.matmul(stack, probabilities).squeeze(-1)

    def change_superposition(self,
                             pointer_probabilities: torch.Tensor,
                             action_probabilities: torch.Tensor):
        """
        Changes the underying state based on the action probabilities.

        Both the enstack and no_op action do not alter the current state,
        since we will want to come back to it later or update it. However,
        the dequeue action will clean up its context call by erasing the
        state.


        :param pointer_probabilities: The probabilities that the stack
            pointers are pointing at a particular location.
        :param action_probabilities: The probabilities of each action
               for the various states.
        """

        # Combine together and isolate the probabilities with which the two events we
        # care about can happen. We are either going to
        #
        # 1) Erase the current subroutine context to clean up the state. Destack action does this
        # 2) Do nothing, so we can come back later or use the current state. Other two do this
        #
        # Fortunately, we can just figure out what is erased, and use that to figure
        # out how much is retained. Then we multiply so that we only keep the retained.

        erasure_probability = action_probabilities[..., 0]*pointer_probabilities # Figure out the erasure probability
        self.stack = self.stack*(1-erasure_probability.unsqueeze(-1)) # Multiply by the retention probability.

    def update(self,
               pointer_probabilities: torch.Tensor,
               tensor: torch.Tensor):
        """
        Performs an update step, integrating the new tensor and performing
        the add+layernorm. This will resolve the layer.
        :param pointer_probabilities: The existing pointer probabilities. Shape (size, ...)
        :param tensor: The tensor to integrate. Shape (..., d_model)
        """
        # Get the stack

        stack = self.stack

        # Extrapolate the stack update by how much the pointer is active.
        tensor = tensor.unsqueeze(0)*pointer_probabilities.unsqueeze(-1)
        update = stack*(1-pointer_probabilities.unsqueeze(-1))
        stack = tensor + update

        # Store
        self.stack = stack


class SubroutineEmbeddingTracker(SubroutineStateTracker):
    """
    Manages the context of virtual layers and the embeddings flowing through them.
    Extends the SubroutineStateTracker to ALSO handle the add plus layernorm responsibility
    that embeddings need to be run through

    **Additional Responsibilties**:

       In addition to what is defined in subroutine state tracker,
       After each subroutine call (handled by external computation), this class
       manages the residual bypass operation, integrating the results of the
       subroutine call into the original layer's context using an add + layernorm
       operation. This normalization ensures stability and consistency between
       layers.

    **Constructor Parameters**:
    - `stack`: The current stack setup, containing the context of subroutines and
               layers.
    - `layernorm`: The layer normalization function applied after updating the stack
                   with new results.
    - `positions`: The positional embeddings that track the relative position within
                   the stack
    """

    def __init__(self,
                 stack: torch.Tensor,
                 layernorm: nn.LayerNorm,
                 positions: torch.Tensor,
                 ):
        """
        :param stack: The current stack setup
        :param layernorm: The layernorm to use when updating
        """
        super().__init__(stack)
        self.layernorm = layernorm
        self.positions = positions

    def update(self,
               pointer_probabilities: torch.Tensor,
               tensor: torch.Tensor):
        """
        Performs an update step, integrating the new tensor and performing
        the add+layernorm. This will resolve the layer.
        :param pointer_probabilities: The existing pointer probabilities. Shape (size, ...)
        :param tensor: The tensor to integrate. Shape (..., d_model)
        """
        # Get the stack

        stack = self.stack

        # Perform the process of actually computing an update, by integrating the existing
        # features then adding plus layernorming.
        update = tensor.unsqueeze(0) + stack + self.positions
        update = self.layernorm(update)

        # Extrapolate the stack update by how much the pointer is active.
        stack = stack*(1-pointer_probabilities.unsqueeze(-1)) \
                +update*pointer_probabilities.unsqueeze(-1)

        # Store
        self.stack = stack


class DifferentiableSubroutineStack:
    """
    A differentiable stack for handling subroutine execution and residual bypass.

    This stack integrates both subroutine handling and residual normalization
    to perform complex, stateful computations in a neural network model. The
    stack allows for operations to occur in a superposition across multiple
    stack levels, controlled by a probabilistic stack pointer. It is designed
    to allow a model to compute a quantity, then return to the previous context,
    in a differentiable manner.

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

    **Residual Bypass and Normalization**:
    Each action works with unnormalized tensors:
      - **Enstack**: Normalizes the incoming tensor and inserts it into deeper
        levels.
      - **Destack**: Combines the popped tensors with the current stack context
        and normalizes.
      - **No-op**: Adds the current tensor's residual to the active context and
        normalizes.
    """


    @property
    def stack_depth(self) -> int:
        return self.pointers.pointers.shape[0]

    @property
    def stack_probabilities(self) -> torch.Tensor:
        return self.pointers.get().sum(dim=0)

    @property
    def stack_empty(self):
        return torch.all(self.stack_probabilities < 1e-4)
    def __init__(self,
                 action_manager: ActionsManagement,
                 pointers: ProbabilisticPointers,
                 stack: SubroutineStateTracker
                 ):

        self.action_manager = action_manager
        self.pointers = pointers
        self.stack = stack

    def get(self) -> torch.Tensor:
        """
        Retrieves the current expression of the stack based on the probabilistic pointer.

        This method computes a weighted average of tensors across all stack levels,
        determined by the probabilistic pointer, and returns the combined result.

        :return: The current stack superposition (weighted combination of stack tensors)
            - Shape (..., d_model)
        """
        pointer_probabilities = self.pointers.get()
        return self.stack.get(pointer_probabilities)

    def update(self,
               action_logits: torch.Tensor,
               tensor: torch.Tensor
               )->Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the stack by applying action logits and commits the new tensor state.

        The stack performs one of three actions (enstack, no-op, or destack) based on the logits,
        adjusting the probabilistic pointer accordingly. The method then normalizes the updated
        stack and returns the results from any destacked features that fell off the top of the stack.

        :param action_logits: Logits specifying the action for each stack level (depth, ..., 3).
        :param tensor: The new tensor to commit to the stack (Shape: (..., d_model)).
        :return: The probabilities and tensor representing the destacked elements
                - Shape (...)
                - Shape (..., d_model)
        """

        # Get some features that will be needed
        pointer_probabilities = self.pointers.get()
        action_probabilities = self.action_manager(action_logits)

        # Modify the stack superposition. Then integrate the new tensor
        destack_probability = self.pointers.change_superposition(action_probabilities)
        self.stack.change_superposition(pointer_probabilities, action_probabilities)
        self.stack.update(self.pointers.get(), tensor)

        # Return the probability and the tensor
        return destack_probability, tensor



class SubroutineStackFactory(nn.Module):
    """
    A factory class for creating DifferentiableSubroutineStack instances.

    This factory generates instances of the DifferentiableSubroutineStack class, which
    handles subroutine execution and residual bypass in neural network models. The
    factory configures these stacks based on the provided tensor and the model's
    requirements, such as stack depth and the computation limit.

    Attributes:
        stack_depth (int): The number of stack levels the subroutine can use.
        d_model (int): The dimensionality of each tensor within the stack.

    Parameters:
        stack_depth (int): Depth of the stack, indicating how many subroutine layers can
                           be stacked.
        d_model (int): Dimensionality of each tensor at each level of the stack.

    Methods:
        forward(tensor: torch.Tensor, min_iterations_before_destack: int, computation_limit: int) -> DifferentiableSubroutineStack:
            Creates a differentiable subroutine stack instance based on the provided tensor,
            initializing the stack, probabilities, and position markers.
    """


    def __init__(self,
                 stack_depth: int,
                 d_model: int):
        """
        Initializes the SubroutineStackFactory.

        :param stack_depth: The maximum depth of the subroutine stack.
        :param d_model: The dimensionality of each tensor in the stack.
        """
        super().__init__()

        # Set the stack depth and model dimension
        self.stack_depth = stack_depth
        self.d_model = d_model

        # Create the position markers for each stack level. This helps in maintaining
        # positional information across different stack layers.
        self.stack_markers = nn.Parameter(torch.zeros([stack_depth, d_model]))

        # Use layer normalization to maintain stability during training by
        # normalizing the input features for each stack level.
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self,
                tensor: torch.Tensor,
                min_iterations_before_destack: int,
                computation_limit: int,
                is_embedding: bool) -> DifferentiableSubroutineStack:
        """
        Creates a differentiable subroutine stack.

        This method initializes the stack with the provided tensor, sets up stack probabilities,
        and configures position markers. It returns a DifferentiableSubroutineStack instance
        ready for use in handling subroutine execution in the model.

        :param tensor: The tensor to initialize the stack with.
            - Shape: (..., d_model), represents the initial context tensor.
        :param min_iterations_before_destack: Minimum iterations before destacking is allowed.
            - Ensures that the stack is not popped until some minimum number of operations have occurred.
        :param computation_limit: Maximum number of iterations allowed before the stack
            forces a flush (destack operation).
        :param is_embedding: Whether this is embedding state, or just computational state.
        :return: An instance of DifferentiableSubroutineStack, initialized with probabilities,
                 stack tensors, and position markers.
        """

        # Step 1: Set up the initial probabilities for the stack.
        # Only the first level of the stack has 100% probability initially, indicating that
        # the subroutine begins at the first stack level.
        probabilities = torch.zeros([self.stack_depth] + list(tensor.shape[:-1]),
                                    device=tensor.device, dtype=tensor.dtype)
        probabilities[0] = 1.0  # Full focus on the first level.
        pointer_probabilities = ProbabilisticPointers(probabilities)

        # Step 2: Set up the stack itself.
        # The provided tensor is placed at the first stack level, while all other levels are initialized to zero.
        # This prepares the stack for subroutine execution where tensors can be added to deeper levels.


        stack = torch.zeros([self.stack_depth] + list(tensor.shape), device=tensor.device, dtype=tensor.dtype)
        stack[0] = tensor  # Insert the context tensor at the base of the stack.

        stack_pos_markers = self.stack_markers
        while stack_pos_markers.dim() < stack.dim():
            stack_pos_markers = stack_pos_markers.unsqueeze(1)

        if is_embedding:
            stack_data = SubroutineEmbeddingTracker(stack, self.layernorm, stack_pos_markers)
        else:
            stack_data = SubroutineStateTracker(stack)

        # Step 3: Configure the actions feature, that manages statistics
        # computing actions and masking.
        statistics = torch.zeros(list(probabilities.shape) + [3],
                                 device=probabilities.device, dtype=probabilities.dtype)

        actions_manager = ActionsManagement(min_iterations_before_destack, computation_limit,
                                            self.stack_depth, statistics)


        # Return the DifferentiableSubroutineStack with the initialized components.
        return DifferentiableSubroutineStack(actions_manager, pointer_probabilities, stack_data)

class SubroutineDriver:
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
                 core: SubroutineCore
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






