import torch
from torch import nn
from typing import List, Dict, Tuple, Any, Union, Optional, Callable
from abc import ABC, abstractmethod

from src.main.model.base import TensorTree,StatefulCore


class CreateSubroutineStub(ABC):
    """
    An abstract representation of the idea of creating
    a subroutine.

    A subroutine is created while viewing the context
    from the superroutine. However, exactly how that
    is used might vary.

    On an implementation level, you can imagine what we are
    doing is providing possible initialization strategies. The
    tensor we return has stack depth - each of the stack dimensions
    tells us if you were to initialize a subroutine HERE with 100%
    probability, what would the subroutine end up filled with.
    """
    @abstractmethod
    def create_subroutine(self,
                          stack: torch.Tensor
                          ) -> torch.Tensor:
      """
      Create the subroutine context. This will be used to
      start running the subroutine.

      :param stack: The stack to manipulate. Shape (stack_depth, ...)
      :return: The subroutine init content. NOT the new stack.
        - If initialized per stack in stack depth
        - Shape (stack_depth, ...)
      """

    def __call__(self, stack: torch.Tensor)->torch.Tensor:
        """
        Runs the create subroutine mechanism. Minimal error checking
        :param stack: Shape (stack_depth, ...)
        :return: Shape (stack_depth, ...)
        """
        output = self.create_subroutine(stack)
        assert output.shape == stack.shape
        return output
class CreateSubroutineUsingDefaults(CreateSubroutineStub):
    """
    Creates subroutines using a default state. This might
    be useful if, for instance, you need each subroutine
    to start in the same markov state. Presumably, other
    stack features are NOT set to the same default state
    each iteration.

    In practice, we replicate the default state to fit
    the stack shape. This can be chosen by downstream
    differentiable probabilities, and incorporated
    into updates under superposition.
    """
    def __init__(self, default_state: torch.Tensor):
        """
        :param default_state: The default state to start in. can be broadcast later.
        """
        super().__init__()
        self.default_state = default_state

    def create_subroutine(self, stack: torch.Tensor) -> torch.Tensor:
        """
        Creates subroutines with the default state as their context

        :param stack: The stack. Shape (stack_depth, ...)
        :return: The subroutine context.
        - If initialized per stack in stack depth
        - Shape (stack_depth, ...)
        """

        output = self.default_state
        if stack.dim() > output.dim():
            output = output.unsqueeze(0)
        output = output.expand_as(stack)
        return output

class CreateSubroutineUsingState(CreateSubroutineStub):
    """
    Creates a subroutine while copying the existing context.

    In practice, we roll the stack one deeper, ensuring that
    the new subroutine would have the old subroutine state if
    initialized
    """
    def create_subroutine(self, stack: torch.Tensor) -> torch.Tensor:
        """
        Creates a subroutine. The state will be based on the state of
        the super_routine.
        :param stack: The subroutine stack. Shape (stack_depth, ...)
        :return:
        -  The proposed subroutine state
        -  if we would make a subroutine at each stack location, we would fill it with this
        - Shape (stack_depth, ...)
        """
        # Roll one deeper
        stack = stack.roll(1, dims=0)

        # The end of the stack rolled back into the beginning. Mask it
        stack[0] = 0

        return stack

class CreateBlankSubroutine(CreateSubroutineStub):
    """
    Creates a subroutine. The new subroutine is filled with zeros.
    """
    def create_subroutine(self,
                          stack: torch.Tensor
                          ) -> torch.Tensor:
        """
        :param stack: The stack state. Shape (stack_depth, ...)
        :return: The proposed subroutine starting values. Shape (stack_depth, ...)
        """
        return torch.zeros_like(stack)



# Subroutine maintence.

class MaintainSubroutineStub(ABC):
    """
    Code responsible for maintaining the subroutine situation.

    Usually, this should not do anything. Conceptually, though,
    you can imagine it to indicate what the new subroutine stack
    state should be if the one at the stack depth was x, and we had
    the subroutine running there.
    """

    def __init__(self, **kwargs):
        super().__init__()
    @abstractmethod
    def maintain_subroutine(self, stack: torch.Tensor)->torch.Tensor:
        """
        Maintains the subroutine
        :param stack: The state of the subroutine. Shape (stack_depth, ...)
        :return: The state of the  subroutine. Shape (stack_depth, ...)
        """

    def __call__(self, stack: torch.Tensor)->torch.Tensor:
        """
        Invokes the maintain subroute method
        :param stack: The stack. Shape  (stack_depth, ...)
        :return: The state after the maintence action. Shape (stack_depth, ...)
        """
        output = self.maintain_subroutine(stack)
        assert output.shape == stack.shape
        return output

class MaintainSubroutine(MaintainSubroutineStub):
    """
    Maintains the existing subroutine without changes
    """
    def maintain_subroutine(self, stack: torch.Tensor) ->torch.Tensor:
        return stack


# Return from subroutine

class ReturnFromSubroutineStub(ABC):
    """
    Responsible for performing the return from
    subroutine action. This can optionally mix the old
    and new subroutine context. Keep in mind that downstream
    the update action will happen as well, which means you can
    still integrate the "return" from a subroutine without needing
    to integrate the process used to get there.

    Implementation wise, we are returning a tensor that indicates
    if we were to return from a subroutine to stack_depth location,
    this is what the new content of the stack at that location would
    be.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def return_from_subroutine(self,
                               stack: torch.Tensor,
                               )->torch.Tensor:
        """
        Performs the return from subroutine action.
        :param stack: The current stack. Shape (stack_depth, ...)
        :return: The new proposed subroutine states. Shape (stack_depth, ...)
        """


    def __call__(self, stack: torch.Tensor)->torch.Tensor:
        output = self.return_from_subroutine(stack)
        assert output.shape == stack.shape
        return output

class ReturnAndDiscardContext(ReturnFromSubroutineStub):
    """
    Performs a subroutine return without retaining
    any of the previous context. Instead, we simply
    go back to the prior context.

    This can be done by simply returning the provided stack, as
    the downstream probability pointers will move us to point to
    the right element.
    """
    def return_from_subroutine(self,
                               stack: torch.Tensor,
                               ) ->torch.Tensor:
        """
        Return to the caller subroutine state.
        :param stack: The current stack. Shape (stack_depth, ...)
        :return: The new proposed subroutine states. Shape (stack_depth, ...)
        """
        return stack

class ReturnAndMerge(ReturnFromSubroutineStub):
    """
    Returns from the subroutine, and adds
    together the two contexts. This is done
    with a roll.

    We also mask out information that rolled off the
    top of the stack.
    """
    def return_from_subroutine(self,
                               stack: torch.Tensor,
                               )->torch.Tensor:
        """
        Adds the two states together
        :param stack: The current stack. Shape (stack_depth, ...)
        :return: The new proposed subroutine states. Shape (stack_depth, ...)
        :return: The combined state. Shape (stack_depth, ...)
        """
        subroutine_context = stack.roll(-1, dims=0)
        subroutine_context[-1] = 0
        return stack + subroutine_context


# Update mechanisms. Incorporates new information into the stack
class UpdateStateStub(ABC):
    """
    Performs the state update mechanism, integrating
    new information from external locations into
    this context level.

    Implementationwise, we try all updates, then
    return something which indicates if we integrated the
    update into this stack depth, this is what the new
    state would be.
    """
    @abstractmethod
    def update_state(self,
                     update: torch.Tensor,
                     options: torch.Tensor
                     )->torch.Tensor:
        """
        Will update the state. Merges somehow the
        update and state.
        :param update: The update to integrate. Shape (...)
        :param options: The options to use. Shape (stack_depth, ..., 3)
        :return: The revised options. Shape (stack_depth, ..., 3)
        """
    def __call__(self,
                 update: torch.Tensor,
                 options: torch.Tensor
                 )->torch.Tensor:
        """
        Will update the existing state with new information.
        :param update: The update to incorporate into the stack.  Shape (...)
        :param options: The options to use. Shape (stack_depth, ..., 3)
        :return: The new options. Shape (stack_depth, ..., 3)
        """
        output = self.update_state(update, options)
        assert output.shape == options.shape
        return output


class UpdateByAddLayernorm(UpdateStateStub):
    """
    Updates by adding then layernorm'ing
    the input. Useful when managing embeddings
    """
    layernorm: nn.LayerNorm
    def __init__(self, layernorm):
        self.layernorm = layernorm
    def update_state(self,
                     update: torch.Tensor,
                     options: torch.Tensor
                     )->torch.Tensor:
        """
        Performs the actual update
        :param update: The update to apply. Shape (...)
        :param options: The options to use. Shape (stack_depth, ..., 3)
        :return: The add plus layernorm based state.
        """

        options = options + update.unsqueeze(0).unsqueeze(-1)
        stack = self.layernorm(options)
        return stack

class UpdateBySet(UpdateStateStub):
    """
    Updates work by setting the state to the
    update value. Useful with, for example, finite
    state machines.
    """
    def update_state(self,
                     update: torch.Tensor,
                     options: torch.Tensor
                     )->torch.Tensor:
        """
        Performs the actual update
        :param update: The update to apply. Shape (...)
        :param options: The options to use. Shape (stack_depth, ..., 3)
        :return: The update, as a direct set and replacement
        """

        update = update.unsqueeze(0).unsqueeze(-1)
        return update.expand_as(options)



# The actual subroutine interface itself

class SubroutineLogicStub:
    """
    Contains the four subroutine
    actions we need in order to run the
    subroutine mechanism. Also contains
    the seed state we will initially load the
    subroutine with.
    """
    def __init__(self,
                 seed_state: torch.Tensor,
                 create_subroutine: CreateSubroutineStub,
                 maintain_subroutine: MaintainSubroutineStub,
                 return_from_subroutine: ReturnFromSubroutineStub,
                 update_subroutine: UpdateStateStub
                 ):
        self.seed_state = seed_state
        self.create_subroutine = create_subroutine
        self.maintain_subroutine = maintain_subroutine
        self.return_from_subroutine = return_from_subroutine
        self.update_subroutine = update_subroutine

# Factories


class SubroutineFactory(nn.Module, ABC):
    """
    Holds everything needed to setup
    a working subroutine manager
    on the forward pass, if the forward
    pass is given the batch state.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self,
                *args,
                **kwargs
                )->SubroutineLogicStub:
        pass

# Create some of the presets

class EmbeddingsManagerFactory(SubroutineFactory):
    """
    Creates subroutine managers dedicated to handling
    embeddings. New subroutines are initialized with
    zeros, then filled in with the update. Returning
    from a subroutine results in addition then layer norm.
    """
    def __init__(self,
                 d_model: int
                 ):
        super().__init__()
        self.layernorm = nn.LayerNorm(d_model)
    def forward(self,
                embedding: torch.Tensor,
                batch_shape: torch.Size)->SubroutineLogicStub:
        """
        Sets up the subroutine manager.
        :param batch_shape: The shape of the batch portions of the incoming tensor
        :return: The setup subroutine manager.
        """

        # Set up the creation routine. The new context will just be zeros
        creation_routine = CreateBlankSubroutine()

        # Set up the maintenance routine
        maintenance_routine = MaintainSubroutine()

        # The return will go back to the previous context
        return_routine = ReturnAndDiscardContext()

        # And finally, the update mechanism adds the pending update then layernorms.
        update_routine = UpdateByAddLayernorm(self.layernorm)

        # Return the produced mechanism

        return SubroutineLogicStub(embedding,
                                   creation_routine,
                                   maintenance_routine,
                                   return_routine,
                                   update_routine)


class SetStateManagerFactory(SubroutineFactory):
    """
    State management engine. State is set directly.
    When returning from a subroutine, state is discarded.
    Useful for things like finite state machine?
    """
    def __init__(self,
                 mode: str,
                 default: torch.Tensor,
                 ):
        super().__init__()
        assert mode in ("default", "copy")

        self.mode = mode
        if mode == "default":
            self.default = default

    def forward(self,
                batch_shape: torch.Size
                )->SubroutineLogicStub:
        """
        Set up the state manager.
        :param batch_shape: The batch shape
        :return: The new subroutine manager
        """

        # Set up the creation routine.
        if self.mode == "default":
            # In default mode, we are going to need to expand the given default
            # to have the batch dimensions.

            default = self.default
            for _ in range(len(batch_shape)):
                default = default.unsqueeze(-1)
            default = default.expand(list(batch_shape) + [-1]*default.dim())

            # Now create it
            creation_routine = CreateSubroutineUsingDefaults(default)
        else:
            # Copy the existing routine information in when spinning up a new one
            creation_routine = CreateSubroutineUsingState()

        # Handle the maintence routine

        maintain_routine = MaintainSubroutine()

        # Handle the return subroutine

        return_routine = ReturnAndDiscardContext()

        # Handle the update routine. We just set

        update_routine = UpdateBySet()

        # Return

        return SubroutineLogicStub(creation_routine,
                                   maintain_routine,
                                   return_routine,
                                   update_routine)

