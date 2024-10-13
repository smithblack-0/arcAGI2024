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
    """
    @abstractmethod
    def create_subroutine(self,
                          superroutine_context: torch.Tensor
                          ) -> torch.Tensor:
      """
      Create the subroutine context. This will be used to
      start running the subroutine.

      :param superroutine_context: The context of the superroutine spawning
             this subroutine.
             - Shape (...)
      :return: The context to run the subroutine in.
            - Shape (...)
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
    """

    #TODO: Consider engineering a mechanism that ensures the
    # driver routine does not have to create a superposition
    # it will not use.
    def __init__(self, default_state: torch.Tensor):
        super().__init__()
        self.default_state = default_state

    def create_subroutine(self, super_routine_context: torch.Tensor) -> torch.Tensor:
        """
        Creates subroutines with the default state as
        their context.

        :param super_routine_context: The super routine context.
            - Shape (...)
        :return: The subroutine context.
        """
        output = self.default_state
        return output

class CreateSubroutineUsingState(CreateSubroutineStub):
    """
    Creates a subroutine while copying the existing context.
    """
    def create_subroutine(self, super_routine_state: torch.Tensor) -> torch.Tensor:
        """
        Creates a subroutine. The state will be based on the state of
        the super_routine.
        :param super_routine_state: The super routine state. Shape (...)
        :return: The new subroutine state. Shape ( ...)
        """
        # We literally just copy the super routine state
        return super_routine_state

class CreateBlankSubroutine(CreateSubroutineStub):
    """
    Creates a subroutine. The new subroutine is filled with zeros
    """
    def create_subroutine(self,
                          super_routine_state: torch.Tensor
                          ) -> torch.Tensor:
        """
        Creates a subroutine state. It is the superroutine state filled with zeros
        :param super_routine_state: The super routine state. Shape (...)
        :return: The new subroutine state. Shape ( ...)
        """
        return torch.zeros_like(super_routine_state)



# Subroutine maintence.

class MaintainSubroutineStub(ABC):
    """
    Code responsible for maintaining the subroutine situation.
    This will generally not change anything.
    """

    def __init__(self, **kwargs):
        super().__init__()
    @abstractmethod
    def maintain_subroutine(self, state: torch.Tensor)->torch.Tensor:
        """
        Maintains the subroutine
        :param state: The state of the subroutine. Shape (...)
        :return: The state of the  subroutine. Shape (...)
        """

    def __call__(self, state: torch.Tensor)->torch.Tensor:
        """
        Invokes the maintain subroute method
        :param state: The state. Shape  (...)
        :return: The state after the maintence action. Shape (...)
        """
        output = self.maintain_subroutine(state)
        assert output.shape == state.shape
        return output

class MaintainSubroutine(MaintainSubroutineStub):
    """
    Maintains the existing subroutine without changes
    """
    def maintain_subroutine(self, state: torch.Tensor) ->torch.Tensor:
        return state


# Return from subroutine

class ReturnFromSubroutineStub(ABC):
    """
    Responsible for performing the return from
    subroutine action. This can optionally mix the old
    and new subroutine context.
    """
    def __init__(self, **kwargs):
        super().__init__()

    @abstractmethod
    def return_from_subroutine(self,
                               caller_state: torch.Tensor,
                               subroutine_state: torch.Tensor
                               )->torch.Tensor:
        """
        Performs the return from subroutine action.
        :param caller_state: The state that the subroutine was called from. Shape (...)
        :param subroutine_state: The state of the current subroutine. Shape (...)
        :return: The new caller state. Shape (...)
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
    """
    def return_from_subroutine(self,
                               caller_state: torch.Tensor,
                               subroutine_state: torch.Tensor
                               ) ->torch.Tensor:
        """
        Return to the caller subroutine state. 
        :param caller_state: The caller state
        :param subroutine_state: The current state
        :return: The new caller state
        """
        return caller_state
    
class ReturnAndMerge(ReturnFromSubroutineStub):
    """
    Returns from the subroutine, and adds 
    together the two contexts.
    """
    def return_from_subroutine(self,
                               caller_state: torch.Tensor,
                               subroutine_state: torch.Tensor
                               )->torch.Tensor:
        """
        Adds the two states together
        :param caller_state: The caller state
        :param subroutine_state: The subroutine state
        :return: The combined state
        """
        return caller_state + subroutine_state
    

# Update mechanisms. Incorporates new information into the stack
class UpdateStateStub(ABC):
    """
    Performs the state update mechanism, integrating
    new information from external locations into
    this context level
    """
    @abstractmethod
    def update_state(self,
                     update: torch.Tensor,
                     state: torch.Tensor
                     )->torch.Tensor:
        """
        Will update the state. Merges somehow the
        update and state.
        :param update: The update to make
        :param state: The state to use.
        :return: The new state
        """
    def __call__(self,
                 update: torch.Tensor,
                 state: torch.Tensor
                 )->torch.Tensor:
        """
        Will update the existing state with new information.
        :param update: The update to incorporate into the stack.  Shape (...)
        :param state: The state to update. Shape (...)
        :return: The new stack. Shape (stack_depth, ...)"""
        output = self.update_stack(update, state)
        assert output.shape == state.shape
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
                     state: torch.Tensor
                     )->torch.Tensor:
        """
        Performs the actual update
        :param update: The update to apply. Shape (...)
        :param state: The state to update. Shape (...)
        :return: The add plus layernorm based state.
        """
        state = state + update
        state = self.layernorm(state)
        return state

class UpdateBySet(UpdateStateStub):
    """
    Updates work by setting the state to the
    update value. Useful with, for example, finite
    state machines.
    """
    def update_state(self,
                     update: torch.Tensor,
                     state: torch.Tensor
                     )->torch.Tensor:
        """
        Performs the actual update
        :param update: The update to apply. Shape (...)
        :param state: The state to update. Shape (...)
        :return: The add plus layernorm based state.
        """
        return update


# The actual subroutine interface itself

class SubroutineManager:
    """
    Contains the four subroutine
    actions we need in order to run the 
    subroutine mechanism.
    """
    def __init__(self,
                 create_subroutine: CreateSubroutineStub,
                 maintain_subroutine: MaintainSubroutineStub,
                 return_from_subroutine: ReturnFromSubroutineStub,
                 update_subroutine: UpdateStateStub
                 ):
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
    def forward(self, batch_shape: torch.Size)->SubroutineManager:
        """
        This needs to be implemented
        :param batch_shape: The batch shape
        :return: The working subroutine manager.
        """
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
    def forward(self, batch_shape: torch.Size)->SubroutineManager:
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

        return SubroutineManager(creation_routine,
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
                )->SubroutineManager:
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

        return SubroutineManager(creation_routine,
                                 maintain_routine,
                                 return_routine,
                                 update_routine)


        

