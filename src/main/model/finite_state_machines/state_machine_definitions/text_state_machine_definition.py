from typing import Dict, List, Union, Optional

from src.main.model.finite_state_machines.finite_state_operators import (ChannelNames,
                                                                         FSMOperator,
                                                                         TriggerPattern, Int,
                                                                         ChangeStatePattern,
                                                                         Matching, NotMatching,
                                                                         CountUpWithRegroup,
                                                                         WriteData)


def define_text_shape_select_operator(mode: int,
                                    channel_names: ChannelNames,
                                    states: Dict[str, int],
                                    max_tokens: int,
                                    )->FSMOperator:
    """
    This defines the shape select operator we might need to handle.

    :param mode: The mode int text has been assigned
    :param channel_names: The channel names
    :param states: the state integers
    :param max_tokens: The maximum number of tokens to generate in a given block
    :return: The FSMOperator
    """
    # Define the shape select trigger behavior
    trigger_pattern = TriggerPattern()
    trigger_pattern.register(Int(channel_names.state, states["shape_select"], 0))  # Must be in shape select mode
    trigger_pattern.register(Int(channel_names.mode, mode, 0))  # Must have a text mode
    trigger_pattern.register(Int(channel_names.substate, 0, 0))  # Must have 0 in substate

    # Define the action of writing and transferring
    next_state = states["decoding"]
    change_pattern = ChangeStatePattern(trigger_pattern)
    change_pattern.register(Int(channel_names.state, next_state, 0)) # Transfer to the next state
    change_pattern.register(WriteData(channel_names.shape, max_tokens, 0)) # Write the shape of the block
    change_pattern.register(Int(channel_names.substate, 0, 0)) # Clean up substate. Always
    # Return
    return FSMOperator(trigger_pattern, change_pattern)

def define_decoding_operators(mode: int,
                              channel_names: ChannelNames,
                              states: Dict[str, int],
                              vocabulary_size: int,
                              reset_pattern: ChangeStatePattern,
                              )->List[FSMOperator]:
    """
    This defines the decoding operators we need. We basically need two of them.

    :param mode: The mode int we are operating under
    :param channel_names: The channel names
    :param states: The state mappings
    :return: A list of FSM operators, to be compiled later
    """

    operators = []
    # Define the decoding pattern. This will match and otherwise handle
    # the situation in which we wish to win
    trigger_pattern = TriggerPattern()
    trigger_pattern.register(Int(channel_names.state, states["decoding"], 0)) # Must be in decoding state
    trigger_pattern.register(Int(channel_names.mode, mode, 0)) # Must be decoding text
    trigger_pattern.register(NotMatching("shape", "index"))

    # Define the action that results
    output_pattern = ChangeStatePattern()
    output_pattern.register(WriteData(channel_names.data, vocabulary_size, 0))
    output_pattern.register(CountUpWithRegroup("index", "shape"))
    operators.append(FSMOperator(trigger_pattern, output_pattern))

    # Now, with the decoding done, we ALSO need to define when decoding is done.
    #
    # It is considered "DONE" when the index has been advanced sufficiently to make a
    # match

    trigger_pattern = TriggerPattern()
    trigger_pattern.register(Int(channel_names.state, states["decoding"], 0)) # Must be in decoding state
    trigger_pattern.register(Int(channel_names.mode, mode, 0)) # Must be decoding text
    trigger_pattern.register(Matching("shape", "index")) # Must have match across inde
    operators.append(FSMOperator(trigger_pattern, reset_pattern))

    return operators

def define_text_mode(mode: int,
                     channel_names: ChannelNames,
                     states: Dict[str, int],
                     shape: List[int],
                     vocabulary_size: int,
                     reset_pattern: ChangeStatePattern,
                     )->List[FSMOperator]:
    """
    Defines the FSMOperators to support a text mode.

    :param mode: The mode number that has ended up associated
    :param channel_names: The channel names as found
    :param states: The states as found
    :param shape: The shape.
    :param vocabulary_size: The vocabulary size
    :return: The FSM Operators ready to be compiled later
    """
    assert len(shape) == 1
    max_tokens = shape[-1]
    operators = [define_text_shape_select_operator(mode, channel_names, states, max_tokens)]
