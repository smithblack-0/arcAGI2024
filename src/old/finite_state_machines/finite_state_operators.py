from typing import List
from dataclasses import dataclass


# Define basic operand

@dataclass(frozen=True)
class OperandPrimitive:
    """Basic structure of an operand."""
    channel: str


@dataclass(frozen=True)
class TriggerOperand(OperandPrimitive):
    """
    The pattern operand type. Mixin. Tells downstream
    components this can be used to match an FSM state pattern
    to trigger behavior on.
    """


@dataclass(frozen=True)
class ActionOperand(OperandPrimitive):
    """
    Mixin. Tells us that this operand will somehow modify
    the FSM state when used in an action (write) manner.
    """


# Define the operands themselves, including whether
# they are triggers or actions.
@dataclass(frozen=True)
class IntOperand(TriggerOperand, ActionOperand):
    """
    Represents an integer operand.
    Can be used to pattern match or as an action operand.
    """
    value: int
    position: int


@dataclass(frozen=True)
class MatchingOperand(TriggerOperand):
    """
    Triggers when the target channel matches this channel.
    """
    target_channel: str


@dataclass(frozen=True)
class NotMatchingOperand(TriggerOperand):
    """
    Triggers when the target channels do NOT match this channel.
    """
    target_channel: str


# Define FSM action operators
@dataclass(frozen=True)
class WriteData(ActionOperand):
    """
    Action that indicates the need to write predictions to this channel.
    """
    requires_embeddings: bool
    vocab_size: int
    position: int


@dataclass(frozen=True)
class CountUpWithRegroup(ActionOperand):
    """
    Advances the first element on each invocation, until a regroup state is
    reached with respect to the helper channel. At that point, a regroup and carry
    operation is performed on the next element.
    """
    helper_channel: str


class FSMOperator:
    """
    FSMOperator: A parallel FSM operator that manages both triggers and actions.

    This class allows for the registration of triggers (conditions that need to be met)
    and actions (modifications to the FSM state). Once triggers are satisfied, the associated
    actions will be executed to update the FSM state.

    Methods:
    - `register_trigger()`: Register a condition for the FSM to check.
    - `register_action()`: Register an action to perform if the trigger is met.
    - `get_triggers()`: Returns the list of registered triggers.
    - `get_actions()`: Returns the list of registered actions.

    Example Usage:
        fsm_operator = FSMOperator()
        fsm_operator.register_trigger(MatchingOperand('channel_a', 'channel_b'))
        fsm_operator.register_action(WriteData(10, 1))
    """

    def __init__(self):
        """
        Initializes an FSMOperator with empty trigger and action lists.
        """
        self.triggers: List[TriggerOperand] = []
        self.actions: List[ActionOperand] = []

    def register_trigger(self, trigger: TriggerOperand):
        """
        Registers a trigger condition for the FSM operator.

        :param trigger: The trigger operand to register.
        """
        assert isinstance(trigger, TriggerOperand), "Trigger must be of type TriggerOperand."
        self.triggers.append(trigger)

    def register_action(self, action: ActionOperand):
        """
        Registers an action to be executed by the FSM operator.

        :param action: The action operand to register.
        """
        assert isinstance(action, ActionOperand), "Action must be of type ActionOperand."
        self.actions.append(action)

    def get_triggers(self) -> List[TriggerOperand]:
        """
        Returns the list of registered triggers.

        :return: List of triggers.
        """
        return self.triggers

    def get_actions(self) -> List[ActionOperand]:
        """
        Returns the list of registered actions.

        :return: List of actions.
        """
        return self.actions


@dataclass(frozen=True)
class ChannelNames:
    """
    Defines the channel names used in the FSM, including state, mode, shape, and data channels.
    """
    state: str
    substate: str
    mode: str
    shape: str
    index: str
    data: str
