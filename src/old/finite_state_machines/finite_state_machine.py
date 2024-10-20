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

from .finite_state_operators import (FSMOperator, MatchingOperand, NotMatchingOperand,
                                     IntOperand, TriggerOperand, ActionOperand,
                                     WriteData, CountUpWithRegroup)
from src.main.CBTensors import CBTensor, CBTensorSpec
from abc import ABC, abstractmethod
from typing import List, Dict, Union, Tuple, Any, Optional

class IntakeMachine(nn.Module):
    """
    The Intake State Machine is responsible for taking in a state tensor which can consist of
    an extremely complex finite state machine state and converting it into one of N defined
    finite states. We then return what the FSM state was associated with. This allows downstream
    entities to process it.

    To make this happen, from the backend perspective, we need to recognize when a finite state tensor
    is associated with a state we need to trigger on. This is done by registering trigger handler classes
    that are defined to accept a CBTensor of FSM state, then emits a tensor that quantifies by boolean
    mask whether we are matching the conditions for this state. When all such states are matched on all
    triggers for a particular n, that state enters a triggered condition which can be handled downstream.

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
    _trigger_handler_classes = []

    @classmethod
    def register_trigger(cls, trigger_handler: 'IntakeTrigger')->'IntakeTrigger':
        """
        Registers a trigger handling class. This should be able to be
        initialized with a spec and device, and has a register operator
        method that can be used to add a handling behavior

        :param trigger_handler: The trigger handler to register.
        """
        if not issubclass(trigger_handler, IntakeTrigger):
            raise ValueError("Trigger was not of intake trigger type")
        cls._trigger_handler_classes.append(trigger_handler)
        return trigger_handler

    def __init__(self,
                 operators: List[FSMOperator],
                 spec: CBTensorSpec,
                 device: torch.device):
        super().__init__()
        self._num_states = len(operators)

        # Define the trigger handlers and action handlers
        self.trigger_handlers = nn.ModuleList(handler(spec, device)
                                              for handler in self._trigger_handler_classes)
        # Load all the operators into the handlers
        for operator in operators:
            for handler in self.trigger_handlers:
                handler.register_operator(operator)
    def forward(self, state_tensor: CBTensor)->torch.Tensor:
        """
        Runs the actual forward pass

        :param state_tensor:
            - The state tensor
            - All trigger handlers will be tested against this, and
              all must be vaild for a action to be selected.
            - shape (...)
        :return:
            - The action, for each shape in (...), that was identified as triggered
            - Shape (..., actions)
            - Boolean mask.
            - One of actions MUST be triggered.
        """
        # Create a feature to perform logical and against with all the test cases
        # This forms an "index mask" that will tell us what operator we need to trigger
        # All options start as true. As trigger checks fail, they are slowly set to false

        actions_mask = torch.full([*state_tensor.shape, self._num_states],
                                  True, device=state_tensor.device)

        # Test each case, and discard nonmatching.
        for trigger in self.trigger_handlers:
            trigger_mask = trigger(state_tensor)
            actions_mask = torch.logical_and(trigger_mask, actions_mask)

        # One, and only one, trigger must be active at each moment. Else we throw
        if torch.any(actions_mask.sum(dim=-1) != 1):
            raise ValueError("Some values never had a triggered state, or had multiple")

        # Turn into indexes
        indexes = torch.arange(self._num_states, device =state_tensor.device)
        indexes = indexes.expand_as(actions_mask)
        indexes = indexes.masked_select(actions_mask)
        indexes = indexes.view(state_tensor.shape)

        return indexes

class IntakeTrigger(nn.Module, ABC):
    """
    A specialized layer meant to be used as part of the
    finite state machine. It is responsible for providing
    some level of filtration and rejecting invalid cases.
    It must keep track internally of what operands actually
    are requesting this service, and handle that accordingly
    """

    @abstractmethod
    def register_operator(self, operator: FSMOperator):
        """
        Registers a single FSM operator onto the intake
        trigger.

        :param operator: The operator to register
        """

    def __init__(self,
                 spec: CBTensorSpec,
                 device: torch.device):
        super().__init__()
        self.spec = spec
        self.device = device

    @abstractmethod
    def forward(self, tensor: CBTensor)->torch.Tensor:
        """
        This should take the cb tensor, and return a
        torch.tensor of the same shape. That tensor should be bool, and
        indicate if one of the filtration conditions was met.

        :param tensor:
        :return:
        """

@IntakeMachine.register_trigger
class IntStateTrigger(IntakeTrigger):
    """
    An integer-based state trigger. It will keep track of certain integer
    patterns it wants to match, and return true if they match. It examines
    the FSM CBTensor to determine this
    """
    def register_trigger(self,
                         channels: List[str],
                         positions: List[int],
                         values: List[int]
                         ):
        """
        Registers a particular pattern of integer content as triggering an
        int filtration. Any channels defined but not provided are assumed to be
        wildcards and masked away

        All lists are syncronized:

        :param channels: The channels to register to. Whatever they might be
        :param positions: The position within the channel to register to
        :param values: The values to match to.
        """

        # Create the new mask features. We will set individual elements of them.
        #
        # Note that since the channel mask starts out as true, everything
        # is assumed to be masked until set otherwise.
        new_channel_mask = torch.full([self.spec.total_width], True,
                                      device=self.device,
                                      dtype=torch.bool)
        new_match_targets = torch.zeros([self.spec.total_width],
                                        dtype=self.match_values.dtype,
                                        device=self.device)

        # Load information from each trigger into the mask. All must be met to trigger
        # We also need to remember to disable the wildcard masking so we have to care
        # what the value is
        for channel, position, value in zip(channels, positions, values):
            # Basic validation
            if channel not in self.spec.channels:
                raise ValueError("Channel was missing")
            if position >= self.spec.channel_widths[channel]:
                raise ValueError("Position was greater than channel width")

            # We are using pointer arithmetic to address so far since the start of the
            # channel. We set the targets to the value, and the mask to false so we
            # care about them
            pointer = self.spec.start_index[channel] + position
            new_match_targets[pointer] = value
            new_channel_mask[pointer] = False



        # Store the new updates.

        new_channel_mask = new_channel_mask.unsqueeze(0)
        new_match_targets = new_match_targets.unsqueeze(0)

        self.channel_masks = torch.concat([self.channel_masks, new_channel_mask], 0)
        self.match_values = torch.concat([self.match_values, new_match_targets], 0)
    def register_operator(self, operator: FSMOperator):
        """
        Registers the given operator onto the IntStateTrigger, if int
        triggers exist. Otherwise, it becomes a noop

        :param operator: The operator to register
        """

        channels = []
        positions = []
        values = []
        for trigger in operator.triggers:
            if isinstance(trigger, IntOperand):
                channels.append(trigger.channel)
                positions.append(trigger.position)
                values.append(trigger.value)

        self.register_trigger(channels, positions, values)

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
        raw_matches = torch.logical_or(raw_matches, self.channel_masks)
        matches = torch.all(raw_matches, dim=-1)

        return matches

@IntakeMachine.register_trigger
class MatchingCasesTrigger(IntakeTrigger):
    """
    Defines a trigger to go off when matching patterns occur, or alternatively
    when nonmatching patterns occur. The logic is ALMOST the same for both, so
    we go ahead and handle both cases in one class to save on vectorized lookups
    """
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
        # Create the blank masks. By default, we specify everything as
        # not important, and the mode mask value then does not matter and
        # is arbitrarily false.
        relations_mask = torch.full([self.spec.total_width, self.spec.total_width],
                                     False,
                                     dtype=torch.bool, device=self.relations_required_mask.device
                                    )
        mode_mask = torch.full([self.spec.total_width, self.spec.total_width],
                                     False,
                                     dtype=torch.bool, device=self.relations_required_mask.device
                                    )
        # Loop through
        for channel_a, channel_b, mode in zip(channel_a, channel_b, trigger_on_match):
            # Some simple validation
            assert channel_a in self.spec.channels
            assert channel_b in self.spec.channels
            assert self.spec.channel_widths[channel_a] == self.spec.channel_widths[channel_b]

            # Fill a portion of the mask, that now specifies this region of cross multiplication
            # matters. In particular, we now specify in the relations mask that THIS region
            # needs to be cross similar, and the associated mode.
            relations_mask[self.spec.slices[channel_a], self.spec.slices[channel_b]] = True
            mode_mask[self.spec.slices[channel_a], self.spec.slices[channel_b]] = mode

        # Store
        relations_mask = relations_mask.unsqueeze(0)
        mode_mask = mode_mask.unsqueeze(0)

        self.relations_required_mask = torch.concat([self.relations_required_mask, relations_mask], 0)
        self.mode_mask = torch.concat([self.mode_mask, mode_mask], 0)

    def register_operator(self, operator: FSMOperator):
        """
        Registers a particular operator and any match or notmatching
        cases that it might possess to be part of this trigger.

        :param operator: A FSM operator to register
        """
        # Create containers
        channel_a = []
        channel_b = []
        trigger_on_match = []

        # Fetch information off operator
        for trigger in operator.triggers:

            # Handles situation in which matching operand appears
            if isinstance(trigger, MatchingOperand):
                channel_a.append(trigger.channel)
                channel_b.append(trigger.target_channel)
                trigger_on_match.append(True)

            # Handles situation in which nonmatching operand appears
            if isinstance(trigger, NotMatchingOperand):
                channel_a.append(trigger.channel)
                channel_b.append(trigger.target_channel)
                trigger_on_match.append(False)

        self.register_trigger(channel_a, channel_b, trigger_on_match)

    def __init__(self,
                 spec: CBTensorSpec,
                 device: torch.device = None
                 ):
        super().__init__(spec, device)

        # The relations_required_mask is used to do basically everything we need
        # It tells us what entries MUST be similar when comparing channels,
        # and has shape (operators, channels, channels)
        #
        # The mode mask, meanwhile, associates elements of the relation mask with
        # a particular matching mode. True means trigger on match, false means
        # trigger on no match. We can then use boolean logic to behave properly.

        self.relations_required_mask =torch.zeros([0, spec.total_width, spec.total_width],
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

        inner_product = ~torch.logical_xor(inner_product, self.mode_mask) #XAND
        inner_product[..., ~self.relations_required_mask] = True # All masked entries are true

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
        if not issubclass(output_class, OperatorAction):
            raise ValueError("operator action was not of OperatorAction class")
        cls._output_classes.append(output_class)
        return output_class

    def __init__(self,
                 operators: List[FSMOperator],
                 spec: CBTensorSpec,
                 device: torch.device = None
                 ):
        super().__init__()
        self._num_operators = len(operators)
        self._output_action = nn.ModuleList(output_action(spec, device) for output_action in self._output_classes)

        for operator in operators:
            for action in self._output_action:
                action.register_operator(operator)
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
        output = tensor.clone()
        for action in self._output_action:
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
    def register_operator(self, operator: FSMOperator):
        """
        Registers a particular operator, and otherwise handles
        the situation.
        """

    def __init__(self,
                 spec: CBTensorSpec,
                 device: torch.device = None
                 ):

        super().__init__()
        self.spec = spec
        self.device = device

    @abstractmethod
    def forward(self,
                state_tensor: CBTensor,
                operation: torch.Tensor,
                prediction: torch.Tensor) -> CBTensor:
        """
        Performs the actual action that may modify the
        tensor. Whatever that might be.

        :param state_tensor: The finite state tensor.
            - Presumably, we need to fiddle with this
            - Shape (...)
        :param operation: The operation to perform.
            - This is a tensor of integers
            - Shape  (...)
            - Each integer is an index, telling us what action to perform,
              and of length channels.
        :param prediction:
            - The content that has been predicted by the model.
            - An integer, not logits.
            - Some actions may involve writing.
        :return: The new state tensor
        """

@OutputMachine.register
class IntSetAction(OperatorAction):
    """
    The int set action is responsible for statically
    setting integer values to be equal to certain other
    values, when relevant
    """
    def register_operator(self, operator: FSMOperator):
        """
        Registers the given FSM operator to respond
        to the current state index.

        :param operator: The operator we are registering
        """

        # Create the new features that need to be stored
        new_set_state = torch.zeros([self.spec.total_width],
                                    device=self.device,
                                    dtype=torch.long)
        new_set_mask = torch.zeros([self.spec.total_width],
                                   device=self.device,
                                   dtype=torch.bool)

        # Load in the int conditions in the operator
        for action in operator.get_actions():
            if isinstance(action, IntOperand):
                assert action.channel in self.spec.channels
                assert action.position <= self.spec.channel_widths[action.channel]

                # Set the set state to contain a meaningful value, and the set mask
                # to want to set that value
                pointer = self.spec.start_index[action.channel] + action.position
                new_set_state[pointer] = action.value
                new_set_mask[pointer] = True

        # Store
        new_set_state = new_set_state.unsqueeze(0)
        new_set_mask = new_set_mask.unsqueeze(0)

        self.set_states = torch.concat([self.set_states, new_set_mask], dim=0)
        self.set_masks = torch.concat([self.set_masks, new_set_mask], dim=0)
    def __init__(self,
                 spec: CBTensorSpec,
                 device: torch.device = None
                 ):
        super().__init__(spec, device)

        # To accomplish what we desire, we store a compressed representation
        # of everything we want to set in terms of a channel, and then a mask
        # that can ensure those things are actually set.

        self.set_states = torch.zeros([0, spec.total_width],
                                      dtype = torch.long,
                                      device=device
                                      )
        self.set_masks = torch.zeros([0, spec.total_width],
                                     dtype = torch.bool,
                                     device=device)

    def forward(self,
                state_tensor: CBTensor,
                operation: torch.Tensor,
                prediction: torch.Tensor) -> CBTensor:
        """
        Performs the actual action of writing the static features into
        the state tensor.

        :param state_tensor: The finite state tensor.
            - Presumably, we need to fiddle with this
            - Shape (...)
        :param operation: The operation to perform.
            - This is a tensor of integers
            - Shape  (...)
            - Each integer is an index, telling us what action to perform,
              and of length channels.
        :param prediction:
            - The content that has been predicted by the model.
            - An integer, not logits.
            - Some actions may involve writing.
            - Shape (...)
        :return: The new state tensor
        """
        # Rebind to my spec, then get the tensor
        original = state_tensor
        state_tensor = state_tensor.rebind_to_spec(self.spec, allow_channel_pruning=True)
        tensor = state_tensor.get_tensor() #(..., channels)

        # Get the set mask and set values, then set them
        set_masks = self.set_masks[operation, :]
        set_values = self.set_values[operation, :]

        tensor[set_masks] = set_values[set_masks]

        # Now put it back together and return
        state_tensor = state_tensor.set_tensor(tensor)
        output = original.set_channels(state_tensor)

        return output

@OutputMachine.register
class WriteAction(OperatorAction):
    """
    The write action is responsible for writing
    to the state tensor something predicted by
    the model.
    """

    def register_operator(self, operator: FSMOperator):
        """
        Register a particular operator and any write actions which
        may be attached to it. If no write actions are attached,
        :param operator: The operator to register. We will look at the
                         action
        """
        # Create the action to perform index setting using
        write_action = torch.full([self.spec.total_width],
                                  False,
                                  dtype=torch.bool,
                                  device=self.write_mask.device
                                  )
        # Go over all the actions. Set based on what we find
        for action in operator.get_actions():
            if isinstance(action, WriteData):
                assert action.channel in self.spec.channels
                assert action.position <= self.spec.channel_widths[action.channel]

                pointer = self.spec.start_index[action.channel] + action.position
                write_action[pointer] = True

        # Store
        write_action = write_action.unsqueeze(0)
        self.write_mask = torch.concat([self.write_mask, write_action], dim=0)


    def __init__(self,
                 spec: CBTensorSpec,
                 device: torch.device
                 ):
        super().__init__(spec, device)

        # Define the write pointers. These tell us what channel to insert
        # into when we go to write our output.

        self.write_mask = torch.zeros([0, spec.total_width], dtype=torch.bool, device=device)

    def forward(self,
                state_tensor: CBTensor,
                operation: torch.Tensor,
                prediction: torch.Tensor) -> CBTensor:
        """
        Performs the actual action of writing to the state tensor

        :param state_tensor: The finite state tensor.
            - Presumably, we need to fiddle with this
            - Shape (...)
        :param operation: The operation to perform.
            - This is a tensor of integers
            - Shape  (...)
            - Each integer is an index, telling us what action to perform,
              and of length channels.
        :param prediction:
            - The content that has been predicted by the model.
            - An integer, not logits.
            - Some actions may involve writing.
            - Shape (...)
        :return: The new state tensor
        """
        # Bind our tensor
        ordered_tensor = state_tensor.rebind_to_spec(self.spec, allow_channel_pruning=True)
        tensor = ordered_tensor.get_tensor()

        # Get write mask and perform broadcast write
        write_masks = self.write_masks[..., operation]
        tensor[write_masks] = prediction.unsqueeze(-1)

        # Return tensor of original shape.
        ordered_tensor.set_tensor(tensor)
        state_tensor.set_channels(ordered_tensor)

        return state_tensor

@OutputMachine.register
class CountAdvanceRegroupAction(OperatorAction):
    """
    Handles the count up with regroup mechanism
    we use to handle advancing the index. For now, you
    can only have one count with advance per FSM
    """
    def register_operator(self, operator: FSMOperator):
        """
        Registers the operator by peeking into it and handling
        any CountUpWithRegroup cases
        :param operator: The operator to register
        """
        # Create the fresh state to store everything in
        counter_indexes = torch.zeros([self.spec.total_width], dtype=torch.bool, device=self.device)
        regroup_indexes = torch.zeros([self.spec.total_width], dtype=torch.bool, device=self.device)
        counter_start = torch.zeros([self.spec.total_width], dtype=torch.bool, device=self.device)

        # Fetch the counter, if it exists, out of the operator
        counters = []
        for action in operator.get_actions():
            if isinstance(action, CountUpWithRegroup):
                counters.append(action)
        # The current storage protocol does not handle anything other than 0 or 1
        # channels.
        if len(counters) > 1:
            raise ValueError("Cannot have more than one counter per operator for now")

        if len(counters) == 1:
            # insert counter into bool masks
            counter = counters[0]

            # Basic validation
            assert counter.channel in self.spec.channels
            assert counter.helper_channel in self.spec.channels
            assert self.spec.channel_widths[counter.channel] == self.spec.channel_widths[counter.helper_channel]

            # Use the spec to insert true where needed
            counter_indexes[self.spec.slices[counter.channel]] = True
            regroup_indexes[self.spec.slices[counter.helper_channel]] = True
            counter_start[self.spec.start_index[counter.channel]] = True

        #Store
        counter_indexes = counter_indexes.unsqueeze(0)
        regroup_indexes = regroup_indexes.unsqueeze(0)
        counter_start = counter_start.unsqueeze(0)

        self.counter_indexes = torch.concat([self.counter_indexes, counter_indexes], dim=0)
        self.regroup_indexes = torch.concat([self.regroup_indexes, regroup_indexes], dim=0)
        self.counter_start_indexes = torch.concat([self.counter_start_indexes, counter_start], dim=0)

    def __init__(self,
                 spec: CBTensorSpec,
                 device: torch.device):

        super().__init__(spec, device)

        # Define the regroup storage and index storage tables. These will tend to tell us
        # where the pieces associated with the counter and the regroup factor are.
        # We also define the start of the counter here. It turns out that makes life a lot
        # easier later on when regrouping.

        self.counter_indexes = torch.zeros([0, spec.total_width], device=device, dtype=torch.bool)
        self.regroup_indexes = torch.zeros([0, spec.total_width], device=device, dtype=torch.bool)
        self.counter_start_indexes = torch.zeros([0, spec.total_width], device=device, dtype=torch.bool)
    def forward(self,
                state_tensor: CBTensor,
                operation: torch.Tensor,
                prediction: torch.Tensor) -> CBTensor:
        """
        Performs the action of advancing the state

        :param state_tensor: The finite state tensor.
            - Presumably, we need to fiddle with this
            - Shape (...)
        :param operation: The operation to perform.
            - This is a tensor of integers
            - Shape  (...)
            - Each integer is an index, telling us what action to perform,
              and of length channels.
        :param prediction:
            - The content that has been predicted by the model.
            - An integer, not logits.
            - Some actions may involve writing.
            - Shape (...)
        :return: The new state tensor
        """

        # Bind tensor to local spec
        ordered_tensor = state_tensor.rebind_to_spec(self.spec, allow_channel_pruning=True)
        tensor = ordered_tensor.get_tensor() #(..., channels)

        # Use operator to get the operator specific varient for each of the three required tensors

        counter_index_mask = self.counter_indexes[operation, :] #(..., channels)
        regroup_index_mask = self.regroup_indexes[operation, :] #(..., channels)
        counter_start_mask = self.counter_start_indexes[operation, :] #(..., channels)

        # Get the proper counter and regroup data out of the tensor. Also, get the
        # relevant subselection of counter start mask, as it will be important momentarily
        #
        # This is going to extract us into a flattened tensor where, nonethless, channel 2
        # will be before channel 3 if they are all from the same tensor, which will allow
        # some clever tricks in a moment

        flat_counters = tensor[counter_index_mask] #(flattened_data)
        flat_regroups = tensor[regroup_index_mask] #(flattened_data)
        flat_start_mask = counter_start_mask[counter_index_mask] #(flattened_data)

        # Increment the counter by one by targetting and increasing the start position.
        # this will still work since we have the start mask. Then detect and isolate regroupings.
        # Remove those regroupings from the accumulator in the adder

        flat_counters[flat_start_mask] += 1 # Increment count
        requires_regroup = torch.floor_divide(flat_counters, flat_regroups) # Detect regroup requirements
        flat_counters -= flat_regroups*requires_regroup # Subtract off regroup from counter

        # Perform regroup action. This consists of rolling everything to the right,
        # setting to zero anything that is now cross contaminating a different counter,
        # then adding the regrouped element

        requires_regroup = torch.roll(requires_regroup, 1, dim=-1)
        requires_regroup[flat_start_mask] = 0
        flat_counters += requires_regroup

        # The counter is finished being updated. Now we need to get it back into place
        tensor[counter_index_mask] = flat_counters
        ordered_tensor = ordered_tensor.set_tensor(tensor)
        state_tensor = state_tensor.set_channels(ordered_tensor)
        return state_tensor

class VocabularySizeMachine(nn.Module):
    """
    The vocabulary size machine has a singular, but
    very important, purpose. It is responsible for
    ensuring there is an easy way to figure out how
    big any prediction logits that will be made need
    to be.
    """
    def register_operator(self, operator: FSMOperator):
        """
        Registers the given operator, and any FSM logic,
        within the vocabulary size machine. This is done
        by examining WriteData operators.

        :param operator: The operator to register
        """
        # Define the vocabulary size default. By default, it is
        # just 1, giving only one logit option

        vocab_size = torch.tensor([1], device=self.device, dtype=torch.long)

        # Go and transfer the vocab size if it exists
        for action in operator.get_actions():
            if isinstance(action, WriteData):
                vocab_size[0] = action.vocab_size

        # Store
        self.vocabulary_size = torch.concat([self.vocabulary_size, vocab_size], dim=0)

    def __init__(self, operators: List[FSMOperator], spec: CBTensorSpec, device: torch.device):
        super().__init__()

        # Setup
        self.spec = spec
        self.device = device

        # Setup vocabulary tracker
        self.vocabulary_size = torch.zeros([], dtype=torch.long, device=self.device)

        # Register
        for operator in operators:
            self.register_operator(operator)

    def forward(self, operation: torch.Tensor) -> torch.Tensor:
        """
        Fetch the vocabulary size given the operation

        :param operation:
            - The operations that are active
            - Shape (...)
        :return:
            - The vocabulary size during each operation
            - Shape (...)
        """

        return self.vocabulary_size[operation]

class FiniteStateMachine(nn.Module):
    """
    The actual finite state machine itself.
    """
    @classmethod
    def create(self,
               operators: List[FSMOperator],
               spec: CBTensorSpec,
               device: torch.device
               )->'FiniteStateMachine':
        """
        Creates a FSM from a list of operators
        :param operators: Operaters to create from
        :param spec: spec to be bound to when thinking about this.
        :param device: The device to create on
        :return: The finite state machine
        """
        intake_machine = IntakeMachine(operators,spec,device)
        output_machine =  OutputMachine(operators, spec, device)
        vocabulary_machine = VocabularySizeMachine(operators, spec, device)


    def __init__(self,
                 intake_machine: IntakeMachine,
                 output_machine: OutputMachine,
                 vocab_machine: VocabularySizeMachine
                 ):

        super().__init__()

        self.input_machine = intake_machine
        self.output_machine = output_machine
        self.vocabulary_machine = vocab_machine

    def compute_finite_state(self, state_tensor: CBTensor) -> torch.Tensor:
        """
        Computes which of a sequence of finite states the current state is
        in
        :param state_tensor: The state tensor
            - Shape (...)
            - The finite state in tensor form
        :return:
            - The finite state tensor tensor
            - Which of the finite states the state tensor was associated with
            - Shape (...)
        """
        return self.input_machine(state_tensor)

    def get_logit_vocabulary(self, operator: torch.Tensor) -> torch.Tensor:
        """
        Gets the logit vocabulary size for each condition we are dealing with,
        knowing what finite state we are in
        :param finite_state:
            - The finite state tensor
            - Which of N choices we are in
            - Shape (...)
        :return: The vocabulary length
            - An integer, per element
            - Indicates how many logit elements need to be active when making
              a prediction, or even taking loss
            - Shape (...)
        """
        return self.vocabulary_machine(operator)

    def encode_transition(self,
                          state_tensor: CBTensor,
                          finite_state_tensor: torch.Tensor,
                          prediction: torch.Tensor
                          )->CBTensor:
        """
        Updates the state tensor to include new information based on the prediction and
        the decoded finite state.

        :param state_tensor: The state tensor, containing the current state
            - Shape (...)
        :param finite_state_tensor: The finite state we were determined to be in
            - Shape (...)
        :param prediction: The prediction
            - Shape (...)
        :return: The new state tensor
        """
        return self.output_machine(state_tensor, finite_state_tensor, prediction)