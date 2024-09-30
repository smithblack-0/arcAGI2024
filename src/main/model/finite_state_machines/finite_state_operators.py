"""
FSM Steps:

There are four primary steps to the FSM, found
in an update and vocab fetch process.

vocab:
    - Vocab fetch step

step:
    DATA_WRITE_STEP
    INDEX_WRITE_STEP
    CONTEXT_STEP





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


from typing import Dict, List, Optional

from src.main.CBTensors.channel_bound_tensors import CBTensorSpec
from dataclasses import dataclass


# Define basic operand

@dataclass(frozen=True)
class OperandPrimitive:
    """Basic structure of an operand"""
    channel: str

@dataclass(frozen=True)
class TriggerOperand(OperandPrimitive):
    """
    The pattern operand type. Mixin. Tells downstream
    components this can sanely be used to match a
    FSM state pattern to trigger behavior on
    """

@dataclass(frozen=True)
class ChangeStateOperand(OperandPrimitive):
    """
    Mixin. Tells us that this will somehow change
    the FSM state when used in a write manner.
    """

# Define the operands themselves, including if they
# are trigger or state only.
@dataclass(frozen=True)
class Int(TriggerOperand, ChangeStateOperand):
    """
    Represents an integer.
    Can be used to pattern match

    """
    value: int
    position: int

@dataclass(frozen=True)
class Matching(TriggerOperand):
    """
    Triggers when the target channel matches this channel
    """
    target_channel: str

@dataclass(frozen=True)
class NotMatching(TriggerOperand):
    """
    Triggers when the target channels do NOT match this channel
    """
    target_channel: str

# Define FSM change operators
@dataclass(frozen=True)
class WriteData(ChangeStateOperand):
    """Indicates the need to write predictions to this"""
    vocab_size: int
    position: int

@dataclass(frozen=True)
class CountUpWithRegroup(ChangeStateOperand):
    """
    Used primarily with indexing. Advances the first element on each
    invokation, until a regroup state is reached with respect to the
    helper tensor, at which point a regroup and carry is performed onto
    the next element.
    """
    helper_channel: str

# Define pattern. This is used to match if changes are needed,
# or alternatively to make changes


class TriggerPattern:
    """
    Contains an entire collection of trigger-valid patterns.
    The operator will "be triggered" and go off when all operand
    patterns are matched.
    """
    def register(self, trigger: TriggerOperand):
        assert isinstance(trigger, TriggerOperand)
        self.patterns.append(trigger)
    def __init__(self, operand_primitives: Optional[List[TriggerOperand]]=None):
        self.patterns: List[TriggerOperand] = []
        if operand_primitives is not None:
            for operand in operand_primitives:
                self.register(operand)

class ChangeStatePattern:
    """
    Contains a collect
    """
    def register(self, change: ChangeStateOperand):
        assert isinstance(change, ChangeStateOperand)
        self.patterns.append(change)
    def __init__(self, operand_patterns: Optional[List[ChangeStateOperand]] = None):
        self.patterns: List[ChangeStateOperand] = []
        if operand_patterns is not None:
            for operand in operand_patterns:
                self.register(operand)

class FSMOperator:
    """
    The definition for a parallel FSM operator. Conceptually, on
    pattern match a trigger is pulled that runs the indicated change
    operators. This will later be compiled.
    """
    def __init__(self,
                 trigger_pattern: TriggerPattern,
                 change_pattern: ChangeStatePattern,
                 ):
        assert isinstance(trigger_pattern, TriggerPattern)
        assert isinstance(change_pattern, ChangeStatePattern)

        self.trigger_pattern = trigger_pattern
        self.change_pattern = change_pattern

@dataclass(frozen=True)
class ChannelNames:
    state: str
    substate: str
    mode: str
    shape: str
    index: str
    data: str


