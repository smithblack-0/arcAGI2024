"""
Finite State Machine (FSM) for Transformer Decoding and Training

This module defines a Finite State Machine (FSM) that is integrated into a transformer-based model
for decoding and reinforcement training. The FSM is designed to manage and transition between
states during the decoding process, based on predefined triggers and state-change operations.

The FSM consists of two primary components:
1. **IntakeMachine**: This component processes the current state of the FSM and determines the next
   operation to perform. It uses a set of triggers to evaluate conditions on the FSM state (represented
   as a `CBTensor`). Once the triggers have been evaluated, the `IntakeMachine` outputs an integer
   representing the operator that should be executed next.

2. **OutputMachine** (Not yet implemented): After the `IntakeMachine` selects the operator, the
   `OutputMachine` will decode the operator and execute the necessary state transitions. It will apply
   the selected operator to the current state, producing a new FSM state and advancing the decoding process.

---

FSM Flow:
- **Input State Processing**: The `IntakeMachine` evaluates multiple registered triggers in parallel.
  These triggers are responsible for analyzing the current state (channels like `state`, `mode`, etc.).
  Once all triggers are satisfied, the `IntakeMachine` returns an index corresponding to the next operator
  to execute.

- **Operator Execution**: The `OutputMachine` will take the index returned by the `IntakeMachine`, decode
  it, and apply the appropriate state changes or actions (e.g., updating channels, transitioning to a new
  state).
"""


import torch
from torch import nn
from torch.nn import Parameter

from .finite_state_operators import FSMOperator, Matching, NotMatching, Int, TriggerOperand, ChangeStateOperand
from src.main.CBTensors import CBTensor, CBTensorSpec
from abc import ABC, abstractmethod
from typing import List, Dict, Union, Tuple, Any, Optional


class IntakeTrigger(nn.Module):
    """
    A specialized layer meant to be used as part of the
    finite state machine. It is responsible for providing
    some level of filtration and rejecting invalid cases.
    It must keep track internally of what operands actually
    are requesting this service, and handle that accordingly
    """
    @abstractmethod
    def register_noop(self):
        """
        Must register a noop that has no effect. Anything registered as a
        no-op must return true when filtration is attempted in order to be valid
        """


    @abstractmethod
    def register_trigger(self, *args, **kwargs):
        """
        Register the intake filter. Should
        :param args:
        :param kwargs:
        :return:
        """

    @abstractmethod
    def forward(self, tensor: CBTensor)->torch.Tensor:
        """
        This should take the cb tensor, and return a
        torch.tensor of the same shape. That tensor should be bool, and
        indicate if one of the filtration conditions was met.

        :param tensor:
        :return:
        """

    def __init__(self, spec: CBTensorSpec):
        super().__init__()
        self.spec = spec

class IntStateTrigger(IntakeTrigger):
    """
    An integer-based state trigger. It will keep track of certain integer
    patterns it wants to match, and return true if they match. It examines
    the FSM CBTensor to determine this
    """
    def register_noop(self):
        """Registers a noop"""

        #No ops are handled by masking out all channels. This will automatically make that
        # return true.

        new_channel_mask =  torch.full([1, self.spec.total_width], True,
                                                     device=self.no_ops.device,
                                                      dtype = torch.bool)
        new_match_targets = torch.zeros([1, self.spec.total_width],
                                        dtype=torch.long,
                                        device=self.no_ops.device)

        self.channel_masks = torch.concat([self.channel_mask, new_channel_mask], 0)
        self.match_values = torch.concat([self.match_targets, new_match_targets], 0)

    def register_trigger(self, pattern: Dict[str, Union[int, List[int]]]):
        """
        Registers a particular pattern of integer content as triggering an
        int filtration. Any channels defined but not provided are assumed to be
        wildcards and masked away

        :param pattern: The pattern to register. Must be a dict of string keys
        that are in the spec, and then the values to look at for those keys
        """
        # Standardize
        pattern = {key : [item] if isinstance(item, int) else item for key, item in pattern.items()}
        pattern = {key : torch.tensor(item, device=self.no_ops.device) for key, item in pattern.items()}

        # Create the new content
        new_channel_mask = torch.full([1, self.spec.total_width], False,
                                      device=self.no_ops.device,
                                      dtype=torch.bool)
        new_match_targets = torch.zeros([1, self.spec.total_width],
                                        dtype=self.match_values.dtype,
                                        device=self.no_ops.device)

        for key in self.spec.channels:
            if key in pattern:
                # We put the match target into the pattern
                new_match_targets[self.spec.slices[key]] = pattern[key]
            else:
                # It is assumed to be wildcard. Mask it
                new_channel_mask[self.spec.slices[key]] = True

        # Store the new updates.
        self.channel_masks = torch.concat([self.channel_mask, new_channel_mask], 0)
        self.match_values = torch.concat([self.match_targets, new_match_targets], 0)

    @classmethod
    def create(cls,
               spec: CBTensorSpec,
               operators: List[FSMOperator],
               device: torch.device=None)->'IntStateTrigger':
        """
        Creates a IntStateTrigger that is properly associated
        with the given set of operators
        :param operators: The operators to trigger with
        :return: The IntStateTrigger
        """
        # Create the trigger
        trigger = cls(spec, device)

        # Attach the operator details
        for operator in operators:

            # Fetch out all the int operator details
            channel_operators = {key : [None]*spec.channel_widths[key]
                                 for key in spec.channels}
            for trigger in operator.trigger_pattern.patterns:
                if isinstance(trigger, Int):
                    bucket = channel_operators[trigger.channel]
                    bucket[trigger.position] = trigger.value
            channel_operators = {key: item for key, item in channel_operators.items() if len(item) > 0}
            if len(channel_operators) == 0:
                # If nothing is left, we do a noop
                trigger.register_noop()
            else:
                # We register what we gathered
                trigger.register_trigger(channel_operators)
        return trigger
    def __init__(self,
                 spec: CBTensorSpec,
                 device: torch.device = None,
                 ):

        super().__init__(spec)

        self.channel_masks = torch.zeros([0, spec.total_width], dtype=torch.bool, device=device)
        self.match_values = torch.zeros([0, spec.total_width], dtype=torch.long, device=device)

    def forward(self, tensor: CBTensor) ->torch.Tensor:
        """
        The forward mechanism actually performs the match and returns the activated operands.

        :param tensor: A CBTensor of shape (...)
        :return: A boolean tensor of shape (..., num_operations).
        """

        # Setup for a broadcast based index search. We are going to take every
        # pattern, and see if it matches on every index

        tensor = tensor.rebind_to_spec(self.spec, allow_channel_pruning=True)
        tensor: CBTensor = torch.unsqueeze(tensor, -1)
        tensor = tensor.get_tensor()

        # Handle masking of wildcards. Also, get actual mask tensor we could sanely return
        raw_matches = torch.eq(tensor, self.match_values) #(..., num_patterns). Bool
        raw_matches = torch.logical_or(raw_matches, self.channel_mask)
        matches = torch.all(raw_matches, dim=-1)

        return matches

class MatchingCasesTrigger(IntakeTrigger):
    """
    Defines a trigger to go off when matching patterns occur, or alternatively
    when nonmatching patterns occur. The logic is ALMOST the same for both, so
    we go ahead and handle both cases in one class to save on vectorized lookups
    """
    def register_noop(self):
        """
        Registers a noop. Everything ends up being masked to true.
        """
        relations_mask = torch.zeros([self.spec.total_width, self.spec.total_width],
                                     dtype=torch.bool, device=self.relations_mask.device)
        mode_mask = torch.tensor([False])

        self.relations_mask = torch.concat([self.relations_mask, relations_mask], 0)
        self.mode_mask = torch.concat([self.mode_mask, mode_mask], 0)
    def register_trigger(self, channel_a: str, channel_b: str, on_match: bool):
        """
        Registers a trigger for the finite state machine on matching channels

        :param channel_a: The first channel to match
        :param channel_b: The second channel to match it with
        :param on_match: Bool. On true, we trigger when matched. On false, we trigger
                              when NOT matched.
        """
        assert channel_a in self.spec.channels
        assert channel_b in self.spec.channels

        # Define the masked regions that matter when performing inner product
        relation_mask = torch.full(self.relations_mask[1:], False,
                                   dtype=torch.bool, device=self.relations_mask.device)
        relation_mask[self.spec.slices[channel_a], self.spec.slices[channel_b]] = True
        relation_mask = relation_mask.unsqueeze(0)

        # Define the mode mask

        mode_mask = torch.tensor([on_match], device=self.mode_mask.device)

        # Store

        self.relations_mask = torch.concat([self.relations_mask, relation_mask], 0)
        self.mode_mask = torch.concat([self.mode_mask, mode_mask], 0)
    def __init__(self,
                 spec: CBTensorSpec,
                 device: torch.device = None
                 ):
        super().__init__(spec)

        # The relations mask is used to do basically everything we need
        # It tells us what entries MUST be similar when comparing channels,
        # and has shape (operators, channels, channels)
        #
        # The mode mask, meanwhile, lets us know if this is operating in mask
        # or not mask mode
        self.relations_mask =torch.zeros([0, spec.total_width, spec.total_width],
                                         dtype=torch.bool, device=device)
        self.mode_mask = torch.zeros([0], dtype=torch.bool, device=device)

    def forward(self, tensor: CBTensor)->torch.Tensor:
        """

        :param tensor:
        :return:
        """
        # Get only the things I care about monitoring
        tensor = tensor.rebind_to_spec(self.spec, allow_channel_pruning=True)
        tensor = tensor.get_tensor() #(..., channels)

        # Perform inner product, then expand and mask out

        inner_product = (tensor.unsqueeze(-1) == tensor.unsqueeze(-2)) # #(..., channels, channels)
        inner_product = inner_product.unsqueeze(-3)
        inner_product = torch.logical_or(inner_product, self.relations_mask) #(..., operator, channels, channels)

        # Flatten, and figure out whether operator was matched
        # Handle NOT matching cases by inverting matched
        output = torch.flatten(inner_product, -2, -1).all(dim=-1)
        output[..., ~self.mode_mask] = ~output[~self.mode_mask]

        return output
class IntakeMachine(ABC):
    """
    IntakeMachine: Processes FSM State to Select an Operator

    The `IntakeMachine` is responsible for processing the current FSM state, evaluating triggers, and
    selecting the next operator to execute. It combines multiple triggers (e.g., integer-based triggers,
    channel matching triggers), evaluates them in parallel, and performs a logical AND on the results
    to select a single operator.

    Once the triggers are satisfied, the `IntakeMachine` returns an integer index representing the next
    operation (operator) to be executed. This index is later passed to the `OutputMachine`, which will
    apply the state changes.

    ---

    Methods:
    - `register_trigger()`: Registers a new trigger for the FSM to evaluate.
    - `forward(tensor: CBTensor) -> int`: Runs all registered triggers and returns an index corresponding
        to the operator to execute next.

    Example Usage:
        intake_machine = IntakeMachine()
        intake_machine.register_trigger(IntStateTrigger(input_spec))
        next_operator = intake_machine(fsm_state_tensor)  # Returns an index for the next operator
    """

    def __init__(self, operators: List[]):

