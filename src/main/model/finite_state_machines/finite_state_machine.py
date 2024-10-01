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

class IntakeMachine(nn.Module):
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
    _trigger_classes = []
    @classmethod
    def register(cls, trigger: 'IntakeTrigger')->'IntakeTrigger':
        """
        Registers various intake triggers that can exist. These will be
        setup by feeding them the operator chain on initialization, and are responsible
        for then compiling it into a trigger.

        :param trigger: The trigger to store, and build later.
        """
        cls._trigger_classes.append(trigger)
        return trigger
    def __init__(self,
                 spec: CBTensorSpec,
                 operators: List[FSMOperator],
                 device: torch.device = None
                ):
        """
        Build the operator trigger machine.

        :param operators: The operators to build our triggers out of
        """
        self._num_operators = len(operators)
        self._triggers = [trigger_class.create_trigger(operators, spec, device) for
                          trigger_class in self._trigger_classes]

    def forward(self, tensor: CBTensor):
        """
        Performs the logical process which tells us what operator is active.
        Basically, we test all strategies, only one should survive, and then
        we get the index associated with that operator.
        :param tensor:
        :return:
        """
        # Create a feature to perform logical and against with all the test cases
        # This forms an "index mask" that will tell us what operator we need to trigger
        index_mask = torch.full([*tensor.shape, self._num_operators], True, device=tensor.device)

        # Test each case, and discard nonmatching
        for trigger in self._triggers:
            trigger_mask = trigger(tensor)
            index_mask = torch.logical_and(trigger_mask, index_mask)

        # Extract the integer index associated with each active case
        if torch.any(index_mask.sum(dim=-1) != 1):
            raise ValueError("Some values never had a triggered state, or had multiple")


        # Get the actual operator indices.
        operator_indexes = torch.arange(self._num_operators, device=tensor.device)
        operator_indexes = operator_indexes.expand_as(index_mask)
        operator_indexes = operator_indexes.masked_select(index_mask)
        operator_indexes = operator_indexes.view(tensor.shape)

        return operator_indexes



class IntakeTrigger(nn.Module, ABC):
    """
    A specialized layer meant to be used as part of the
    finite state machine. It is responsible for providing
    some level of filtration and rejecting invalid cases.
    It must keep track internally of what operands actually
    are requesting this service, and handle that accordingly
    """

    @abstractmethod
    @classmethod
    def create_trigger(cls,
                       spec: CBTensorSpec,
                       operators: List[FSMOperator],
                       device: torch.device)->'IntakeTrigger':
        """
        Creates a setup intake trigger bound to the operator collection.
        :param spec: The spec to bind to
        :param operators: The operator collection to bind to
        :param device: The device to work with.
        :return: The setup intake trigger.
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

    def __init__(self,
                 spec: CBTensorSpec,
                 device: torch.device):
        super().__init__()
        self.spec = spec
        self.device = device

@IntakeMachine.register
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
    def create_trigger(cls,
                       spec: CBTensorSpec,
                       operators: List[FSMOperator],
                       device: torch.device
                       )->'IntStateTrigger':
        """
        Registers a collection of operators based on whether
        they had Int trigger matches on them.
        """

        instance = cls(spec, device)

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
                instance.register_noop()
            else:
                # We register what we gathered
                instance.register_trigger(channel_operators)
        return instance
    def __init__(self,
                 spec: CBTensorSpec,
                 device: torch.device = None,
                 ):

        super().__init__(spec, device)
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

@IntakeMachine.register
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
        mode_mask = torch.zeros([self.spec.total_width, self.spec.total_width],
                                     dtype=torch.bool, device=self.relations_mask.device)

        self.relations_mask = torch.concat([self.relations_mask, relations_mask], 0)
        self.mode_mask = torch.concat([self.mode_mask, mode_mask], 0)
    def register_trigger(self,
                         channel_a: List[str],
                         channel_b: List[str],
                         trigger_on_match: List[bool]
                         ):
        """
        Registers a trigger for the finite state machine to perform by
        comparing and considering matching channels. Note that all
        channels being compared must have the same width. Note also that
        all conditions must be triggered for the output to go off.

        :param channel_a: A list of channels to compare with. The first
        :param channel_b: A list of the other channels to compare to
        :param trigger_on_match: A list of indicators on whether to trigger on match, or
               on no match.
        """

        relations_mask = torch.full([self.spec.total_width, self.spec.total_width],
                                     False,
                                     dtype=torch.bool, device=self.relations_mask.device
                                    )
        mode_mask = relations_mask.clone()

        for channel_a, channel_b, mode in zip(channel_a, channel_b, trigger_on_match):
            # Some simple validation
            assert channel_a in self.spec.channels
            assert channel_b in self.spec.channels
            assert self.spec.channel_widths[channel_a] == self.spec.channel_widths[channel_b]

            # Fill a portion of the mask, that now specifies this region of cross multiplication
            # matters.
            relations_mask[self.spec.slices[channel_a], self.spec.slices[channel_b]] = True
            mode_mask[self.spec.slices[channel_a], self.spec.slices[channel_b]] = mode

        # Store
        relations_mask = relations_mask.unsqueeze(0)
        mode_mask = mode_mask.unsqueeze(0)

        self.relations_mask = torch.concat([self.relations_mask, relations_mask], 0)
        self.mode_mask = torch.concat([self.mode_mask, mode_mask], 0)

    @classmethod
    def create_trigger(cls,
                       spec: CBTensorSpec,
                       operators: List[FSMOperator],
                       device: torch.device) ->'MatchingCasesTrigger':
        """
        Creates the entire class, and registers the matching cases.

        :param spec: The spec
        :param operators: The operators to register in it
        :param device: The device
        :return: The intake trigger
        """

        # create instance

        instance = cls(spec, device)

        # Go over each operator
        for operator in operators:

            # Build the register parameters
            channel_a = []
            channel_b = []
            trigger_on_match = []

            # Build the actual features
            for trigger in operator.trigger_pattern.patterns:
                if isinstance(trigger, Matching):
                    channel_a.append(trigger.channel)
                    channel_b.append(trigger.target_channel)
                    trigger_on_match.append(True)
                elif isinstance(trigger, NotMatching):
                    channel_a.append(trigger.channel)
                    channel_b.append(trigger.target_channel)
                    trigger_on_match.append(False)

            # Store the operator, whatever is needed
            if len(channel_a)  == 0:
                instance.register_noop()
            else:
                instance.register_trigger(channel_a,
                                          channel_b,
                                          trigger_on_match)

        # Return the instance
        return instance


    def __init__(self,
                 spec: CBTensorSpec,
                 device: torch.device = None
                 ):
        super().__init__(spec, device)

        # The relations mask is used to do basically everything we need
        # It tells us what entries MUST be similar when comparing channels,
        # and has shape (operators, channels, channels)
        #
        # The mode mask, meanwhile, associates elements of the relation mask with
        # a particular matching mode. True means trigger on match, false means
        # trigger on no match. We can then use boolean logic to behave properly.


        self.relations_mask =torch.zeros([0, spec.total_width, spec.total_width],
                                         dtype=torch.bool, device=device)
        self.mode_mask = torch.zeros([0, spec.total_width, spec.total_width],
                                     dtype=torch.bool, device=device)

    def forward(self, tensor: CBTensor)->torch.Tensor:
        """

        :param tensor:
        :return:
        """
        # Get only the things I care about monitoring
        tensor = tensor.rebind_to_spec(self.spec, allow_channel_pruning=True)
        tensor = tensor.get_tensor() #(..., channels)

        # Perform inner product, giving me a feature that tells me whether
        # element 'i' is equal to element 'j'

        inner_product = (tensor.unsqueeze(-1) == tensor.unsqueeze(-2)) # #(..., channels, channels)
        inner_product = inner_product.unsqueeze(-3) # (..., 1 (operators), channels, channels)

        # Perform a 'XAND' with the inner product and the mode mask. If the mode mask is false,
        # indicating we want to trigger on NOT masking, the two falses give rise to a true.
        # Likewise for two trues.
        #
        # Then, we use the relations mask to set anything that will not be considered
        # to true. This ignores channel cross relationships that do not matter.

        inner_product = ~torch.logical_xor(inner_product, self.mode_mask)
        inner_product[..., ~self.relations_mask] = True

        # Flatten, and figure out whether we triggered.
        output = torch.flatten(inner_product, -2, -1).all(dim=-1)
        return output

class OutputMachine(nn.Module):
    """
      OutputMachine: Applies FSM Operators to Modify FSM State

      The `OutputMachine` is responsible for executing the operator selected by the `IntakeMachine`. It
      accepts a tensor representing the current FSM state, an operation index (selected operator), and
      the predicted content (e.g., transformer logits).

      This machine performs the necessary state updates, writing the prediction into the appropriate
      FSM channels or updating internal state based on the selected operator.

      Example Usage:
          output_machine = OutputMachine(spec, operators)
          new_state = output_machine(current_state, selected_operation, predicted_content)

      ---

      Parameters:
      - `tensor`: The current FSM state (CBTensor).
      - `operation`: The selected operator (torch.Tensor) to execute.
      - `prediction`: The predicted content to write into the FSM state.

      Returns:
      - A modified FSM state (CBTensor) after applying the selected operator.
      """

    _output_classes = [] # The registered output classes

    @classmethod
    def register(cls, output_class: 'OperatorAction'):
        """
        Registers an operator class which can be compiled to do something
        when it's trigger condition is met.

        :param output_class: The operator
        """
        cls._output_classes.append(output_class)
        return output_class
    def __init__(self,
                 spec: CBTensorSpec,
                 operators: List[FSMOperator],
                 device: torch.device = None
                 ):

        self._num_operators = len(operators)
        self._output_operators = nn.ModuleList(output_action(operators) for output_action in self._output_classes)

    def forward(self,
                tensor: CBTensor,
                operation: torch.Tensor,
                prediction: torch.Tensor,
                )->CBTensor:
        """
        Performs the actual operation. The tensor will be the unmodified
        CBTensor, and the operation is the chosen operation to execute

        :param tensor:
            - The unmodified input tensor. A CBTensor
            - Shape (...)
        :param operation:
            - The operation that needs to be executed
            - Shape (...)
            - Index indicates what operation to run
        :param prediction:
            - The content that has been predicted by the model.
            - It will need to be written somewhere, but sometimes
              that write can just be discarded.
        :return: The new operator action
        """
        # Apply the machines in order
        output = tensor
        for action in self._output_operators:
             output = action(output, operation, prediction)
        assert output.shape == tensor.shape
        return tensor

class OperatorAction(nn.Module, ABC):
    """
    An actual implementation of an action that can
    change the finite state, the operator action
    might be responsible for doing a particular
    thing, like writing
    """
    @abstractmethod
    @classmethod
    def create_action(cls,
                      spec: CBTensorSpec,
                      operators: List[FSMOperator],
                      device: torch.device = None
                      )->'OperatorAction':
        """
        Creates a operator action associated with this spec
        :param spec:
        :param operators:
        :param device:
        :return:
        """

