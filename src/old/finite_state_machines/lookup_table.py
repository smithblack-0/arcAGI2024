

from typing import Tuple, Dict, Any, List

import torch
from src.old.CBTensors import CBTensorSpec, CBIndirectionLookup
from abc import ABC, abstractmethod
from dataclasses import dataclass

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
    # options, allowing the arcAGI2024 to choose the generative mode.

    num_modes = len(modes)
    table.register(input_pattern={"state" : states["mode_select"], "mode" : 0, "submode": 0},
                   output_pattern={"vocab_length" : num_modes})

    # Bind the various vocabulary chains associated with each mode of generation
    for mode in modes:

        # Bind the shape select vocabulary to the lookup table. This will later allow the
        # arcAGI2024 to predict it's block size. The various submodes will be advanced through
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



def create_data_write_lookup_table(num_shape_dims: int,
                               states: Dict[str, int],
                               ) -> Tuple[CBIndirectionLookup, CBTensorSpec]:
    """
    Compiles a lookup table that provides write offsets (channel pointers) for inserting
    predictions into the correct channels of a finite state machine (FSM) state tensor
    during decoding. This enables efficient, vectorized updates to the FSM state across
    multiple elements in a batch.

    ---- Parameters ----
    :param num_shape_dims:
        - The number of dimensions required to represent the shape of a block being
          generated. For example, image data might require two dimensions (width, height),
          while text might only need one (number of tokens).
        - The system supports different maximum dimensions for different modes (e.g., text vs. image).

    :param states:
        - A dictionary mapping FSM state names (e.g., "mode_select", "shape_select",
          "block_decode") to integer indices. These states represent different steps in
          the FSM decoding process, allowing the system to determine what data needs to
          be written to the tensor.

    ---- Returns ----
    :return:
        - A tuple consisting of:
          1. `CBIndirectionLookup`: A lookup table that maps FSM states and submodes to
             channel pointers (write offsets) for writing predictions into the correct
             positions in the tensor.
          2. `CBTensorSpec`: A tensor specification that defines the layout of the FSM
             state tensor (including channels for "state", "mode", "submode", "shape", and "data").
             This spec is used to bind or rebind a `CBTensor` to the correct structure for
             manipulating FSM state.

    ---- How It Works ----

    - The `CBIndirectionLookup` is queried during the decoding process based on the
      current FSM state and submode. It returns a channel pointer (offset) that tells the
      system where to write the next predicted value into the FSM state tensor.
    - The returned `CBTensorSpec` defines the channels and their widths, allowing you to
      bind or rebind a `CBTensor` to this specification. This is important for correctly
      inserting predictions into the appropriate channels of the FSM state.
    - After extracting a subset of the FSM state using the `CBTensorSpec`, you can update
      it based on the predictions, and then insert it back into the full FSM state using
      operations like `set_channels`.

    ---- Efficient Vectorized Updates ----
    - This process enables efficient, batched updates to the FSM state across multiple
      elements at once. By using the `CBTensorSpec` to bind the FSM state to a specific
      structure and then using vector indexing to update the channels, the system avoids
      the need for manual iteration over each element in the batch.
    """

    # Define the channel spec. This will be used to compute the offsets. You can rebind
    # your tensors to these shapes if needed to use the offsets.
    spec = {"state" : 1, "mode" : 1, "submode" : 1, "shape" : num_shape_dims, "data" : 1}
    channel_spec= CBTensorSpec(spec)

    # Define the input requirements and output slots, and the table
    input_spec = CBTensorSpec({"state" : 1, "submode" : 1})
    output_spec = CBTensorSpec({"channel_pointer" : 1})
    table = CBIndirectionLookup(input_spec, output_spec)

    # Store the mode write pointer. This is used to write the selected mode to the FSM state tensor.
    table.register(input_pattern={"state" : states["mode_select"], "submode" : 0},
                   output_pattern={"channel_pointer" : channel_spec.start_index["mode"]})

    # Store the various shape write pointers. Used to write the selected state to the FSM state tensor
    # Use pointer arithmetic for this to go to start of shape, then offset by number
    for i in range(num_shape_dims):
        table.register(input_pattern={"state" : states["shape_select"], "submode" : i},
                       output_pattern={"channel_pointer" : channel_spec.start_index["shape"] + i}
                       )

    # Store the write pattern for when decoding blcoks
    table.register(input_pattern={"state" : states["block_decoding"], "submode" : 0},
                   output_pattern={"channel_pointer" : channel_spec.start_index["data"]}
                   )

    table.indirection_results.tensor.contiguous()
    table.indirection_addresses.tensor.contigous()

    return table, channel_spec


def create_state_transition_lookup_table()