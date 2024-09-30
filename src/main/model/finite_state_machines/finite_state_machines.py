"""
Lets go outline what needs to be tracked in order to convert everything to lookup tables

- TRANSITIONS:
 - Mode_select: immediately after complet

------------------------
VOCABULARY_PERSPECTIVE:

mode_select
shape_select_1a
shape_select_2a
shape_select_1b
shape_select_2b
block_decode_1
block_decode_2

Requires: State, Mode

WRITEPERSPECTIVE:

mode_select
shape_select_1
shape_select_2
block_decode

Requires: State, Submode

INDEXPERSPECTIVE:

everything_else
block_decode.

Requires: Shape, Index

TRANSITION_PERSPECTIVE

mode_select
shape_select_1
shape_select_2
block_select

Requires: State, Index, Shape





"""


from typing import Tuple, Dict, Any, List

import torch
from config import states, modes, mode_dims, vocabularies
from src.main.CBTensors.channel_bound_tensors import CBTensor
from src.main.CBTensors.channel_bound_utilities import CBIndirectionLookup
from abc import ABC, abstractmethod
from dataclasses import dataclass


# Define lookup tables and such
@dataclass
class LookupTables:
    """
    Lookup tables to greatly assist with managing the finite
    state machine.
    """


    transitions: CBIndirectionLookup
    shape_dims: CBIndirectionLookup
    vocabulary: CBIndirectionLookup
    index_transitions: CBIndirectionLookup



def compile_lookup_tables(config: Dict[str, Any])->LookupTables:
    """
    Compiles the lookup tables that can be used to transition and operate.

    :param config:
    :return:
    """

class TensorTransition(ABC):
    """
    A state transition for the tensor logic.

    A collection of state transition tensors allows for the model
    to elegantly move between various states based on predictions made
    by the model, supporting the block based idea. It does thies
    by keeping a context which is updated by state transitions based
    on the context itself.
    """

    def __init__(self,
                 lookup_tables: LookupTables
                 ):
        self.tables = lookup_tables


    @abstractmethod
    def vocabulary(self)->int:
        """
        Tells us what the size of the vocabulary which can be used
        to make this transition will be. This is VERY important, as
        it tells us how much of the logit will be active when we decode
        it.
        """

    @abstractmethod
    def predicate(self, context: CBTensor)->torch.Tensor:
        """
        A predicate that should be returned based on the
        current context. We should be able to use this to
        tell, per element, whether we will go through a
        particular transition.

        :param context:
            - A CBTensor. It should match the configured spec.
            - Shape (
        :return: A predicate of the shape shape as context, that tells us whether
                 or not each element is going to need to be transitioned.
        """
    @abstractmethod
    def transition(self,
                   context: CBTensor,
                   data: torch.Tensor,
                   )->Tuple[CBTensor, torch.Tensor]:
        """
        Performs an actual transition.

        :param context: The context to use for transitioning
        :param data: The data to be inserted or used for transitioning
        :return:
            - The new context or tensor.
            - The size of the vocabulary for this feature.
        """


class ModeSelectTransition(TensorTransition):
    """
    This class is responsible for the mode select
    transition process.

    Mode select occurs when beginning to generate a block, and is an operation
    in which we predict or decide what the upcoming block's content will be filled with.
    """
    def predicate(self, context: CBTensor) -> torch.Tensor:
        # When the mode feature is set to mode select, we
        # need to transition using this class
        mode_info = context.gather_channels("state").get_tensor()
        return mode_info == states.get_mode("mode_select")
    def transition(self,
                   context: CBTensor,
                   data: torch.Tensor,
                   ) ->Tuple[CBTensor, torch.Tensor]:
        # Transitions responsibilities are as follows
        #
        # 1) Set the mode to the mode provided in the data entry, which is presumed to match one
        #    of the supported modes
        # 2) Set the aux state element to match the number of dimensions this mode will need
        #    to generate

        # Compute the new state and mode values. Also, compute the vocabulary size

        state = torch.full_like(data, states.get_mode("state_select"))
        vocab_size = torch.full_like(data, vocabularies.get_number("state_select"))

        # Compute the number of shapes to set
        num_needed_shapes = torch.zeros_like(data)
        for mode in modes.registry.keys():
            requires_set = (modes.get_mode(mode) == data)
            num_needed_shapes[requires_set] = mode_dims.get_number(mode)

        # Return update

        update = CBTensor.create_from_channels({"state" : state,
                                                "mode" : data,
                                                "counter" : num_needed_shapes})
        context = context.set_channels(update)
        return context, vocab_size

class ShapeSelectTransition(TensorTransition):
    """
    The shape selection transition. This is responsible for
    performing selection of a certain number of shapes. The underlying
    FSM can either transition back to itself, in which case the number
    of remaining
    """
    def predicate(self, context: CBTensor) -> torch.Tensor:
        state = context.gather_channels("state").tensor()
        return state == states.get_mode("shape_select")

    def transition(self, context: CBTensor, data: torch.Tensor) -> Tuple[CBTensor, torch.Tensor]:
        # Get the shape, and the current count of shapes to make
        shape = context.gather_channels("shape").tensor()
        count = context.gather_channels("counter").tensor()

        # Use the count as a pointer into the shapes. Insert the data
        shape[count] = data
        count =- 1

        # Transition to the next one.


