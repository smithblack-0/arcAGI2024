from typing import List, Dict

from src.old.finite_state_machines.finite_state_operators import (FSMOperator)


def compile_vocabulary_lookup_table(operators: List[FSMOperator],
                                    states: Dict[str, int],
                                    modes: Dict[str, int]
                                    ):
    """
    Compiles a vocabulary lookup table to support them odel
    :param operators: The operators to compile from
    :param states: The states we are using, and their ints
    :param modes: The modes we are using, and their ints
    :return: A compiled lookup table, that can be used to get
             the vocabulary for a given situation
    """



def create_voculary_length_fsm_lookup_table(
    modes: List[int],
    shapes_vocab_lengths: List[List[int]],
    vocab_sizes: List[int],
    states: Dict[str, int],
    ) -> CBIndirectionLookup:
    """
    Creates a vocabulary length finite state machine based lookup table. This will
    be able to, conceptually, look at the state of the FSM across many batches and
    items and answer the key question: What is the size of the vocabulary when performing
    this prediction step?

    This is required due to the fact that different steps will require different
    logit configuratios, since logits are shared. It lets one logit prediction
    handle all bases for all different tasks. It is also required both to evaluate
    loss while training and evaluate predictions when generating.

    ---- Parameters ----
    :param modes:
     - A list of mode IDs, each representing a different content generation
       modality (e.g., text, image).

    :param shapes_vocab_lengths:
     - A list of to list int, with each being associated with a particular mode.
       For instance, [30, 20] might mean any image up to a 30x20 image.
    :param vocab_sizes:
     - A list of vocabulary sizes for each mode during the actual block decoding process.

    :param states:
     - A dictionary that associates each state (e.g., "mode_select", "shape_select")
       with a unique integer ID. These states help guide the FSM's transitions.

    ---- Returns ----
    :return:
     - A `CBIndirectionLookup` object that maps from FSM state, mode, and submode
       to the corresponding vocab size (logit restriction) used in that generation step.
     """


    # Pad. Make sure all shape lengths now have the same number of dimensions
    max_shapes_length = max(len(shape) for shape in shapes_vocab_lengths)
    shapes_vocab_lengths = [shape + [0] * (max_shapes_length - len(shape)) for shape in shapes_vocab_lengths]

    # Now insert one everywhere. This gives us "0" as a shape option, and in the padding entries
    # will be all that is available.
    vocab_shape_lengths = [[dim + 1 for dim in shape] for shape in shapes_vocab_lengths]

    # Start table
    input_spec = CBTensorSpec({"state" : 1, "mode" : 1, "submode" : 1})
    output_spec = CBTensorSpec({"vocab_length" : 1})
    table = CBIndirectionLookup(input_spec, output_spec)

    # Bind mode select vocabulary size to it. There will be num_mode vocabulary
    # options, allowing the model to choose the generative mode.

    num_modes = len(modes)
    table.register(input_pattern={"state" : states["mode_select"], "mode" : 0, "submode": 0},
                   output_pattern={"vocab_length" : num_modes})

    # Bind the various vocabulary chains associated with each mode of generation
    for mode in modes:

        # Bind the shape select vocabulary to the lookup table. This will later allow the
        # model to predict it's block size. The various submodes will be advanced through
        # by other finite state processes.
        shape_select = states["shape_select"]
        for i, dim in enumerate(shapes_vocab_lengths[mode]):
            table.register(input_pattern={"state" : shape_select, "mode" : mode, "submode" : i},
                           output_pattern={"vocab_length": dim})

        #Bind the vocabulary used when decoding a block.
        shape_select = states["block_decode"]
        table.register(input_pattern={"state" : shape_select, "mode" : mode, "submode" : 0},
                       output_pattern={"vocab_length" : vocab_sizes[mode]})

    return table

