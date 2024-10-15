"""
Module: Subroutine Stubs for Differentiable Stack Management

This module provides the core *Subroutine Stubs* used to construct a differentiable stack mechanism,
enabling efficient management of subroutine calls in recurrent neural networks or other architectures
that require dynamic subroutine control. The stubs function within a *pytree structure*, which defines
the shape and position of tensors across layers and subroutines. This structure is initialized later to
create a fully differentiable subroutine stack, allowing for gradient-based optimization across dynamic
subroutine behavior.

# Key Concepts and Design
1. **PyTree Structure**: Subroutine stubs are organized into a pytree structure. This pytree represents
the layout of tensors for each layer or subroutine, ensuring a consistent shape across all layers. Each
subroutine stack can be initialized and combined, with options determined by the stub configuration.
The pytree allows for parallel computation and easy traversal of different subroutine states across the stack.

2. **Customizable Subroutine Behavior**: The module allows users to define various strategies for managing
subroutines, such as:
    - **Create Subroutine**: Defines the behavior when initializing a new subroutine. This could involve
      copying the current state, using a default state, or starting with zeros. These options are provided
      by classes like `CreateSubroutineUsingDefaults`, `CreateSubroutineUsingState`, and `CreateBlankSubroutine`.
    - **Maintain Subroutine**: Controls how subroutines are maintained across iterations. For example,
      the `MaintainSubroutine` class leaves the subroutine unchanged, though other custom behaviors can be implemented.
    - **Return from Subroutine**: Manages the behavior when returning from a subroutine call. For example,
      `ReturnAndDiscardContext` discards the subroutine context, while `ReturnAndMerge` merges the subroutine state
      with the calling context.
    - **Update Subroutine**: Integrates new information from model computations into the subroutine state.
      For instance, `UpdateByAddLayernorm` applies an add + layernorm strategy, whereas `UpdateBySet` replaces
      the subroutine state with the new value, as seen in finite state machines (FSM).

3. **Parallel Computation of Subroutine Options**: At each iteration, the subroutine logic computes three options:
    - **Create a new subroutine**
    - **Maintain the current subroutine state**
    - **Return from the subroutine**

   These three possibilities are computed in parallel for all levels in the stack, and the final outcome is determined
   by the model's action probabilities. This parallelism allows the model to dynamically adjust between subroutine
   creation, maintenance, and return without the need for sequential decision-making.

4. **Update Integration and Selection**: After the subroutine options are computed, the update action integrates the
new information (from the model's forward pass) into the stack. This integration ensures that the correct option—whether
it’s creating, maintaining, or returning from a subroutine—is selected based on the model's current computation.

# Extensibility
New behaviors can be implemented by extending the abstract classes defined in this module. Users can define
custom strategies for initializing, maintaining, or returning from subroutines, or for updating subroutine
states based on specific tasks.
"""

import torch
from torch import nn
from typing import List, Dict, Tuple, Any, Union, Optional, Callable
from abc import ABC, abstractmethod

from src.main.model.base import TensorTree,StatefulCore


class CreateSubroutineStub(ABC):
    """
    An abstract base class representing a strategy for creating subroutine options
    at different stack levels.

    Each subroutine offers options based on the current context of the subroutine stack. Conceptually,
    this can be viewed as providing possible states the subroutine could start with at any stack level,
    without actually modifying the stack itself. The method of generating these options varies depending
    on the implementation.

    Subclasses should define specific creation mechanisms, generating options that can be selected later
    based on action probabilities.
    """

    @abstractmethod
    def create_subroutine(self, stack: torch.Tensor) -> torch.Tensor:
        """
        Defines how to generate options for a subroutine state at a given stack level.

        :param stack: The current stack context, representing the states of parent subroutines.
                      Shape (stack_depth, ...)
        :return: The proposed subroutine state options at each stack level. The stack itself is not modified.
                 Shape (stack_depth, ...)
        """

    def __call__(self, stack: torch.Tensor) -> torch.Tensor:
        """
        Invokes the subroutine creation mechanism and verifies the shape of the resulting options.

        :param stack: Shape (stack_depth, ...)
        :return: The proposed subroutine options. Shape (stack_depth, ...)
        """
        output = self.create_subroutine(stack)
        assert output.shape == stack.shape
        return output


class CreateSubroutineUsingDefaults(CreateSubroutineStub):
    """
    A subroutine creation strategy that generates subroutine options initialized
    with a default state at each stack level.

    This is useful when each new subroutine should always begin with a predefined state,
    regardless of the other elements in the stack. For example, initializing a new subroutine
    in a fixed Markov state. These options can be selected later based on the model’s behavior.

    The default state is broadcast to match the stack’s dimensions, allowing for superposition
    during subroutine creation.
    """
    def __init__(self, default_state: torch.Tensor):
        """
        :param default_state: The default state used to initialize subroutine options. This tensor
                              can be broadcast to match the required stack dimensions.
        """
        super().__init__()
        self.default_state = default_state

    def create_subroutine(self, stack: torch.Tensor) -> torch.Tensor:
        """
        Generates subroutine options initialized with the default state at each stack level.

        :param stack: The current stack context, representing the possible parent states.
                      Shape (stack_depth, ...)
        :return: The new subroutine context options initialized with the default state.
                 Shape (stack_depth, ...)
        """
        output = self.default_state
        if stack.dim() > output.dim():
            output = output.unsqueeze(0)
        output = output.expand_as(stack)
        return output


class CreateSubroutineUsingState(CreateSubroutineStub):
    """
    A subroutine creation strategy that generates subroutine options by copying
    the parent subroutine's state.

    Instead of starting with a blank state or a default state, the new subroutine
    options at each stack level are created by rolling the stack one position deeper
    and filling the subroutine with the parent’s state. This allows subroutines to inherit
    from the existing context, providing potential options that can be selected later.
    """
    def create_subroutine(self, stack: torch.Tensor) -> torch.Tensor:
        """
        Generates subroutine options by rolling the stack deeper and copying the parent’s state.

        :param stack: The current stack context, which provides the parent state.
                      Shape (stack_depth, ...)
        :return: The proposed subroutine context options, rolled one depth deeper to inherit the parent
                 state. The first stack position is masked with zeros.
                 Shape (stack_depth, ...)
        """
        # Roll one level deeper
        stack = stack.roll(1, dims=0)

        # Mask the top stack level (was rolled from the end to the beginning)
        stack[0] = 0

        return stack



class CreateBlankSubroutine(CreateSubroutineStub):
    """
    A subroutine creation strategy that generates subroutine options initialized with zeros
    at each stack level.

    This strategy is useful when each subroutine should start with a blank context, i.e., without
    any inherited state from parent subroutines. The options created reflect a "clean slate," providing
    potential initializations that can be selected during the model’s forward pass.
    """
    def create_subroutine(self, stack: torch.Tensor) -> torch.Tensor:
        """
        Generates subroutine options initialized with zeros.

        :param stack: The current stack context, used only to determine the shape of the new subroutine context.
                      Shape (stack_depth, ...)
        :return: Subroutine options initialized with zeros at each stack level.
                 Shape (stack_depth, ...)
        """
        return torch.zeros_like(stack)



# Subroutine maintence.

class MaintainSubroutineStub(ABC):
    """
    An abstract base class responsible for generating options when maintaining
    the current subroutine across iterations. These options reflect potential
    states the subroutine could continue in, without transitioning or creating
    a new one.

    Conceptually, this class provides different possible maintenance strategies
    for the current subroutine, which will later be selected from and merged
    during the subroutine stack update process. The goal is to provide viable
    options that represent how the subroutine can proceed if chosen to be maintained.
    """

    def __init__(self, **kwargs):
        super().__init__()

    @abstractmethod
    def maintain_subroutine(self, stack: torch.Tensor) -> torch.Tensor:
        """
        Generates the state options for continuing the current subroutine at a given
        stack depth. These options reflect how the subroutine might progress if it
        continues to run.

        :param stack: The current subroutine stack, representing the state of the subroutine
                      at each depth.
                      Shape (stack_depth, ...)
        :return: The possible subroutine state options for maintaining the subroutine
                 at the given stack depth.
                 Shape (stack_depth, ...)
        """

    def __call__(self, stack: torch.Tensor) -> torch.Tensor:
        """
        Invokes the subroutine maintenance method and ensures that the generated options
        are consistent with the stack's shape.

        :param stack: The stack context representing the current subroutine state.
                      Shape (stack_depth, ...)
        :return: The subroutine state options for maintaining the subroutine, ready for
                 selection and merging during the stack update.
                 Shape (stack_depth, ...)
        """
        output = self.maintain_subroutine(stack)
        assert output.shape == stack.shape
        return output


class MaintainSubroutine(MaintainSubroutineStub):
    """
    Provides an option to continue the current subroutine without making any
    modifications to the subroutine state.

    This class represents a simple "no-op" maintenance strategy, where the
    subroutine's state remains exactly as it is. The existing subroutine state
    is returned unchanged, and no updates or transformations are applied.
    """

    def maintain_subroutine(self, stack: torch.Tensor) -> torch.Tensor:
        """
        Returns the current subroutine stack unchanged, indicating that no
        modifications or updates have been applied to the subroutine.

        :param stack: The current subroutine stack representing the state
                      at each depth. Shape (stack_depth, ...)
        :return: The unmodified subroutine stack. Shape (stack_depth, ...)
        """
        return stack


# Return from subroutine

class ReturnFromSubroutineStub(ABC):
    """
    Abstract base class responsible for providing options for
    the return action from a subroutine.

    The `return` action offers different options based on stack levels,
    representing the possible states to return to if the return branch
    is selected. Each option reflects what the subroutine stack would look
    like after returning from the subroutine at a particular depth. These
    options are then combined probabilistically with other branch actions
    (such as create or maintain).

    This mechanism does not directly modify the stack, but it provides the
    proposed new states for each stack level, to be selected later based
    on the action probabilities.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def return_from_subroutine(self, stack: torch.Tensor) -> torch.Tensor:
        """
        Provides options for returning from a subroutine at each level.

        :param stack: The current stack, representing different contexts
                      at each depth. Shape (stack_depth, ...)
        :return: The proposed new stack states, one for each depth,
                 assuming the return action was taken at that level.
                 Shape (stack_depth, ...)
        """
        pass

    def __call__(self, stack: torch.Tensor) -> torch.Tensor:
        """
        Invokes the `return_from_subroutine` method and ensures shape consistency.

        :param stack: The current stack. Shape (stack_depth, ...)
        :return: The stack after applying the return options. Shape (stack_depth, ...)
        """
        output = self.return_from_subroutine(stack)
        assert output.shape == stack.shape
        return output

class ReturnAndDiscardContext(ReturnFromSubroutineStub):
    """
    Provides an option for returning from a subroutine by discarding
    the subroutine's context entirely and reverting back to the previous
    state.

    When this option is selected, the stack will be reset to the previous
    context, with the subroutine's context being ignored.

    This is useful when the result of the subroutine is no longer needed,
    and the model needs to revert fully to its prior state.
    """

    def return_from_subroutine(self, stack: torch.Tensor) -> torch.Tensor:
        """
        Returns the current stack, reverting fully to the previous context.

        :param stack: The current stack. Shape (stack_depth, ...)
        :return: The stack as-is, assuming the subroutine's context is discarded.
                 Shape (stack_depth, ...)
        """
        return stack

class ReturnAndMerge(ReturnFromSubroutineStub):
    """
    Provides an option for returning from a subroutine by merging
    the subroutine's context with the previous state.

    The subroutine's result is combined with the previous context,
    allowing for integration of the subroutine's effect into the
    ongoing process.

    This is achieved by rolling the stack and masking out the topmost
    entry, ensuring that any information that rolled off the end is
    appropriately handled.
    """

    def return_from_subroutine(self, stack: torch.Tensor) -> torch.Tensor:
        """
        Merges the subroutine's context with the previous stack state.

        Rolls the stack down one position, masking the topmost entry,
        and adds the rolled context to the current state, allowing for
        a smooth transition from the subroutine back into the previous
        context.

        :param stack: The current stack. Shape (stack_depth, ...)
        :return: The merged stack states after applying the return action.
                 Shape (stack_depth, ...)
        """
        subroutine_context = stack.roll(-1, dims=0)
        subroutine_context[-1] = 0  # Mask out the topmost entry to avoid leakage.
        return stack + subroutine_context



# Update mechanisms. Incorporates new information into the stack
class UpdateStateStub(ABC):
    """
    Abstract base class for handling the state update mechanism within
    subroutine stacks. This class defines how new information (updates)
    is integrated into the existing context at each stack depth.

    For every stack depth, it computes potential state options for
    each of the action branches:
      - Element 0: Return action.
      - Element 1: Maintain action.
      - Element 2: Create action.

    The goal is to propose updates to the state under each scenario,
    and these proposed options will later be selected probabilistically
    based on the action branch taken.

    The resulting state reflects what the context would look like
    after integrating the new update with the options for return,
    maintain, or create.
    """

    @abstractmethod
    def update_state(self,
                     update: torch.Tensor,
                     options: torch.Tensor
                     ) -> torch.Tensor:
        """
        Integrates the given update into the existing context options
        (one for each action branch) at every stack depth.

        :param update: The new update to integrate. Shape (...).
        :param options: Existing state options at each stack depth
                        and action branch. Shape (stack_depth, ..., 3).
                        - The last dimension represents the action branch
                          (return, maintain, or create).
        :return: The revised state options for each action branch,
                 after integrating the update. Shape (stack_depth, ..., 3).
        """

    def __call__(self,
                 update: torch.Tensor,
                 options: torch.Tensor
                 ) -> torch.Tensor:
        """
        Calls the `update_state` method, ensuring the output retains the same
        shape as the input options.

        :param update: The update to integrate into the options. Shape (...).
        :param options: The options to use for each action branch
                        and stack depth. Shape (stack_depth, ..., 3).
        :return: The updated options after applying the update. Shape (stack_depth, ..., 3).
        """
        output = self.update_state(update, options)
        assert output.shape == options.shape
        return output


class UpdateByAddLayernorm(UpdateStateStub):
    """
    Implements the state update mechanism by performing an add + layernorm
    operation. This is commonly used in scenarios such as managing embeddings,
    where residual connections are needed to maintain smooth state transitions.

    For each stack depth and action branch, it computes an updated state by
    adding the new information (update) and then applying layer normalization.
    """

    def __init__(self, layernorm: nn.LayerNorm):
        """
        :param layernorm: The layer normalization module to use after updating
                          the state.
        """
        self.layernorm = layernorm

    def update_state(self,
                     update: torch.Tensor,
                     options: torch.Tensor
                     ) -> torch.Tensor:
        """
        Adds the update to the existing options for each stack depth and
        action branch, and then applies layer normalization.

        :param update: The new update to integrate. Shape (...).
        :param options: Existing options for each stack depth and action branch.
                        Shape (stack_depth, ..., 3).
        :return: The updated state options after applying the add + layernorm
                 operation. Shape (stack_depth, ..., 3).
        """
        options = options + update.unsqueeze(0).unsqueeze(-1)
        stack = self.layernorm(options)
        return stack

class UpdateBySet(UpdateStateStub):
    """
    Implements the state update mechanism by directly setting the state
    to the provided update. This is useful for cases where a direct
    replacement of the state is needed, such as in a finite state machine.

    For each stack depth and action branch, it sets the state to the
    provided update, completely replacing the previous option.
    """

    def update_state(self,
                     update: torch.Tensor,
                     options: torch.Tensor
                     ) -> torch.Tensor:
        """
        Directly sets the state for each stack depth and action branch
        to the provided update, completely replacing the current state.

        :param update: The new update to apply as the state. Shape (...).
        :param options: Existing options for each stack depth and action branch.
                        Shape (stack_depth, ..., 3).
        :return: The updated state options after setting the state to the update.
                 Shape (stack_depth, ..., 3).
        """
        update = update.unsqueeze(0).unsqueeze(-1)
        return update.expand_as(options)




# The actual subroutine interface itself

class SubroutineLogicStub:
    """
    Defines the core logic for handling subroutines at an abstract level. It specifies how subroutines
    are created, maintained, returned from, and updated, without involving the specifics of stack depth
    or position. The `SubroutineLogicStub` is used as part of a pytree structure that will later be
    initialized into a full subroutine stack during runtime.

    **Key Responsibilities**:
    - **create_subroutine**: Defines how new subroutines are initialized based on the parent subroutine’s state.
    - **maintain_subroutine**: Specifies what happens when the current subroutine is continued.
    - **return_from_subroutine**: Provides options for what happens when the subroutine is exited and returns to
      a higher level.
    - **update_subroutine**: Manages how new information is integrated into the subroutine state, for example
      by replacing state or adding to it via residual mechanisms.

    **should_accumulate**:
    This flag indicates whether or not this subroutine should accumulate its output across calls. In practical
    terms, some subroutine outputs—like embeddings—might need to be accumulated and returned as part of the
    final output, while other aspects, like internal state transitions (e.g., a Markov state), may not require
    accumulation. When `should_accumulate` is `True`, the stack will ensure the results of this subroutine
    are preserved and returned for later processing.

    **Key Components**:
    - `seed_state`: The initial state used to create new subroutines. This serves as the base context for a subroutine.
    - `create_subroutine`: Defines how a subroutine is initialized at different stack levels.
    - `maintain_subroutine`: Handles the logic for maintaining an active subroutine.
    - `return_from_subroutine`: Dictates how context is handled when returning from a subroutine (merging, discarding, etc.).
    - `update_subroutine`: Manages how updates (e.g., from other layers) are integrated into the subroutine state.
    - `should_accumulate`: Boolean flag that controls whether the output from this subroutine is accumulated.
    """

    def __init__(self,
                 seed_state: torch.Tensor,
                 create_subroutine: CreateSubroutineStub,
                 maintain_subroutine: MaintainSubroutineStub,
                 return_from_subroutine: ReturnFromSubroutineStub,
                 update_subroutine: UpdateStateStub,
                 should_accumulate: bool):
        """
        Initializes the subroutine logic stub with key behaviors and a seed state.

        :param seed_state: The initial state for the subroutine, used when creating new subroutines.
        :param create_subroutine: Defines the behavior for initializing new subroutines.
        :param maintain_subroutine: Defines how to continue execution when the subroutine remains active.
        :param return_from_subroutine: Defines the logic for returning from an active subroutine (e.g., merging or discarding state).
        :param update_subroutine: Specifies how updates are integrated into the subroutine's state.
        :param should_accumulate: A boolean indicating whether results should be accumulated across subroutine calls.
        """
        self.seed_state = seed_state
        self.create_subroutine = create_subroutine
        self.maintain_subroutine = maintain_subroutine
        self.return_from_subroutine = return_from_subroutine
        self.update_subroutine = update_subroutine
        self.should_accumulate = should_accumulate


# Factories


class SubroutineLogicStubFactory(nn.Module, ABC):
    """
    An abstract base class for creating `SubroutineLogicStub` objects that define the
    behavior of subroutine stacks. These logic stubs represent the options and logic
    to be used in different stack operations such as creating, maintaining, updating,
    and returning from subroutines.

    **Responsibilities**:
    - This factory is responsible for constructing logic stubs that will later be initialized
      into full subroutine stacks, based on the provided configuration.
    - Subclass-specific implementations will define how to generate the `SubroutineLogicStub`
      based on the task's requirements.

    Subclasses should implement the `forward` method to return a `SubroutineLogicStub`.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, *args, **kwargs) -> SubroutineLogicStub:
        """
        Abstract method that creates and returns a `SubroutineLogicStub`, which defines
        the behavior of a subroutine for tasks such as embeddings management or state tracking.

        :return: A `SubroutineLogicStub` representing the logic for subroutines.
        """
        pass


class EmbeddingsManagerLogicStubFactory(SubroutineLogicStubFactory):
    """
    A factory that produces logic stubs for managing embeddings in a subroutine stack.
    These stubs define how to handle the subroutine lifecycle for embeddings, including
    initializing, maintaining, and updating embeddings.

    **Responsibilities**:
    - Initializes new subroutines with zeros, which are later updated with the provided values.
    - Maintains existing embeddings in subroutines as-is during the forward pass.
    - On returning from a subroutine, adds the current context to the previous one, followed by
      layer normalization for stability.
    - Specifies whether the embeddings in this subroutine should accumulate and be returned as
      part of the final output, controlled by `should_accumulate`.

    :param d_model: The dimensionality of the embeddings.
    :param should_accumulate: Boolean indicating if the embeddings in this subroutine
                              should be accumulated and returned in the output.
    """

    def __init__(self, d_model: int, should_accumulate: bool = True):
        super().__init__()
        self.layernorm = nn.LayerNorm(d_model)
        self.should_accumulate = should_accumulate

    def forward(self, embedding: torch.Tensor) -> SubroutineLogicStub:
        """
        Creates a `SubroutineLogicStub` that manages the lifecycle of embeddings, including
        how new subroutines are initialized, maintained, and returned from.

        :param embedding: The initial embedding tensor that defines the seed state of the subroutine.
        :return: A `SubroutineLogicStub` for managing embeddings, with logic for creating, maintaining,
                 and updating subroutines.
        """
        # Initialize new subroutines with zeros
        creation_routine = CreateBlankSubroutine()

        # Maintain subroutines as-is
        maintenance_routine = MaintainSubroutine()

        # Return by discarding the current subroutine context and reverting to the previous one
        return_routine = ReturnAndDiscardContext()

        # Update the subroutine by adding the new values, followed by layer normalization
        update_routine = UpdateByAddLayernorm(self.layernorm)

        # Return the logic stub
        return SubroutineLogicStub(
            embedding,
            creation_routine,
            maintenance_routine,
            return_routine,
            update_routine,
            self.should_accumulate
        )


class SetStateManagerLogicStubFactory(SubroutineLogicStubFactory):
    """
    A factory that produces logic stubs for managing a finite-state-machine-like (FSM)
    subroutine stack. This setup directly sets the subroutine state and discards the previous
    state when returning from a subroutine. It is quite suitable for handling state from
    recurrent architetures.

    **Responsibilities**:
    - Initializes new subroutines by setting the state to a specified default value or copying
      the current state, based on configuration.
    - Maintains the state unchanged during the forward pass.
    - On returning from a subroutine, the previous state is discarded, and the subroutine's
      state is directly replaced.
    - Specifies whether this state should accumulate and be included in the final output.

    :param should_accumulate: Boolean indicating if the state in this subroutine should accumulate
                              and be returned in the final output.
    """

    def __init__(self, should_accumulate: bool = False):
        super().__init__()
        self.should_accumulate = should_accumulate

    def forward(self, default_state: torch.Tensor) -> SubroutineLogicStub:
        """
        Creates a `SubroutineLogicStub` that directly manages state updates, suitable for
        finite state machines (FSM).

        :param default_state: The default state tensor for initializing new subroutines.
        :return: A `SubroutineLogicStub` for managing FSM-like subroutines, with logic for
                 creating, maintaining, and setting state.
        """
        # Initialize new subroutines by setting the state to the default value
        creation_routine = CreateSubroutineUsingDefaults(default_state)

        # Maintain the current state unchanged
        maintain_routine = MaintainSubroutine()

        # Return by discarding the current state and reverting to the previous one
        return_routine = ReturnAndDiscardContext()

        # Directly set the subroutine state during updates
        update_routine = UpdateBySet()

        # Return the logic stub
        return SubroutineLogicStub(
            default_state,
            creation_routine,
            maintain_routine,
            return_routine,
            update_routine,
            self.should_accumulate
        )
