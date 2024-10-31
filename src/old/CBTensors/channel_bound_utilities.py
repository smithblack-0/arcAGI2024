from typing import Union, Dict, List

import torch
from torch import nn

from src.old.CBTensors.channel_bound_spec import CBTensorSpec
from src.old.CBTensors import CBTensor


class CBIndirectionLookup(nn.Module):
    """
    This class is designed to allow for efficient
    indirection based on channel patterns. This can be used
    to build fast and efficient finite state machines, among other
    things.

    ---- initialization and setup ----

    The class is initialized by being provided with an

    - Input spec. A CBTensorSpec saying what input channels SHALL be defined.
    - Output spec. A CBTensorSpec saying what output channels SHALL be returned.

    This configures the schema of the map, and prepares it to receive its actual configuration

    ---- configuration ----

    Once setup, we may provide input-output mappings to identify and return. For
    instance, lets say we configure to look for {"state" : 1, "mode": 1}. And output based
    on {"vocabulary_size" : 1}

    Then you would be able to register an input, and subsequence response, like follows:

    .register(input={"state" : 0, "mode" : 0}, output={"vocabulary_size" : 10})
    .register(input={"state" : 1, "mode" : 0}, output={"vocabulary_size" : 100})
    .register(input={"state" : 1, "mode" : 1}, output={"vocabulary_size" : 40})

    ---- usage: __call__ -----

    When called, we promise to perform indirection. Lets conside the example from above,
    with mode, state, vocabulary size. We would accept a CBTEnsor with mode and state
    information. We would attempt to find a match. When we find it, we would return
    the vocabulary size associated with it.

    This will occur across something like an entire batch if need be.
    """
    def __init__(self,
                 input_spec: CBTensorSpec,
                 output_spec: CBTensorSpec,
                 device: torch.device = None,
                 dtype: torch.dtype = torch.long
                 ):

        super().__init__()

        #
        self.device = device
        self.dtype = dtype

        # Define the specs
        self.input_spec = input_spec
        self.output_spec = output_spec

        # Define the indirection registries
        indirection_addresses = torch.zeros([0, input_spec.total_width], dtype=dtype, device=device)
        indirection_results = torch.zeros([0, output_spec.total_width], dtype=dtype, device=device)

        self.indirection_addresses = CBTensor(input_spec, indirection_addresses)
        self.indirection_results = CBTensor(output_spec, indirection_results)

    def standardize_this(self,
                    inputs: Union[CBTensor, Dict[str, Union[int, List[int], torch.Tensor]]]
                    )->CBTensor:
        """
        Ensures the information being stored here will be presented and can be utilized in terms of
        CBTensors. These can later be concatenated.

        :param inputs:
            - The inputs to standardize.
            - Should in concept contain a single CBTensor entry, and it's contents.
        :return:
            - A standardized CBTensor.
        """
        if isinstance(inputs, dict):
            # We are dealing with specification by dictionary.
            # Standardize it as a collection of tensors
            standard_dict = {
                key : torch.tensor([item], dtype=self.dtype, device=self.device) if isinstance(item, int) else
                torch.tensor(item, dtype=self.dtype, device=self.device) if isinstance(item, list) else
                item
                for key, item in inputs.items()
            }
            # Now create the cb  tensors
            tensor = CBTensor.create_from_channels(standard_dict)
        else:
            tensor = inputs

        if tensor.dim() == 0:
            tensor = torch.unsqueeze(tensor, 0)
        elif tensor.dim() > 1:
            raise ValueError(f"Cannot define more than one pattern at a time")
        return tensor

    def register(self,
                 input_pattern: Dict[str, Union[int, List[int], torch.Tensor]],
                 output_pattern: Dict[str, Union[int, List[int], torch.Tensor]]
                 ):
        """
        Registers a particular lookup pattern to be associated with a particular
        lookup result.


        :param input_pattern: The pattern to match. Must corrolate cleanly with the input schema
        :param output_pattern: The pattern to match. Must corrolate cleanly with the output schema.
        """

        input_pattern = self.standardize_this(input_pattern)
        output_pattern = self.standardize_this(output_pattern)

        self.indirection_addresses = torch.cat([self.indirection_addresses, input_pattern], dim=0)
        self.indirection_results = torch.cat([self.indirection_results, output_pattern])

    def forward(self, input: CBTensor)->CBTensor:
        """

        :param input:
            - Shape (...)
            - Spec includes input spec
            - Need to replace matches with patterns
        :return:
            - The output CBTensor
            - Spec matches output spec
            - Contains results per element.
        """

        # Setup for a broadcast based index search. We are going to take every
        # pattern, and see if it matches on every index

        input = input.rebind_to_spec(self.input_spec, allow_channel_pruning=True)
        input = torch.unsqueeze(input, dim=-1) #(..., 1)
        matches = torch.eq(input, self.indirection_addresses) #(..., num_patterns). Bool

        # If no matches were found, or excessive matches were found,
        # we complain. Past this point, we can conclude there is exactly
        # ONE match in num_patterns

        if torch.any(matches.sum(dim=-1) != 1):
            raise ValueError("Extra pattern matches or no pattern matches detected")

        # Get the associated index from num_patterns. Use vector indexing to extract
        # the result.

        indices = torch.arange(self.indirection_results.shape[0], device=input.device)
        while indices.dim() < matches.dim():
            indices = indices.unsqueeze(0)
        indices = indices.expand_as(matches)
        items = indices[matches]
        outcomes = self.indirection_results[items]
        return outcomes

class CBReplaceOnMatch(nn.Module):
    """
    A utility class for selectively replacing specific channels of a `CBTensor` based on pattern matches.
    When an input pattern matches, the corresponding output pattern channels are written to the input
    tensor, leaving all other channels unchanged. This is particularly useful for managing and updating
    finite state machine (FSM) tensors in a vectorized, batch-friendly manner.

    ---- Initialization ----
    :param input_spec:
        - A `CBTensorSpec` that defines the schema for input patterns. It ensures that the input tensor
          channels align with the patterns used for matching.
    :param output_spec:
        - A `CBTensorSpec` that defines the schema for output patterns. These output patterns will
          selectively overwrite corresponding channels in the input tensor when a match occurs.
    :param device:
        - The device where tensors will be processed (e.g., CPU or GPU).
    :param dtype:
        - The data type of the tensors (default is `torch.long`).

    ---- Core Concepts ----
    * **Pattern Matching**:
      The class allows the registration of input patterns that are matched against input tensor channels.
      Upon finding a match, the specified channels in the `output_spec` are written into the input tensor.

    * **Selective Channel Replacement**:
      Rather than replacing the entire input tensor, only the channels defined in the `output_spec` are
      updated. All other channels in the input remain unaffected.

    * **Batch and Vectorized Operation**:
      This class is optimized to process an entire batch of tensors at once, using broadcasting for pattern
      matching and vector indexing for efficient replacements.

    * **Dimensional Consistency**:
      The output tensor will retain **exactly the same shape** as the input tensor, including both non-channel
      and channel dimensions. The only change is that the channels specified by the `output_spec` are updated
      where matches occur, while all other channels remain unchanged.

    ---- Methods ----

    * `register(input_pattern: Dict[str, Union[int, List[int], torch.Tensor]],
                output_pattern: Dict[str, Union[int, List[int], torch.Tensor]]) -> None`:
      Registers an input-output pattern pair for matching and selective replacement. The input pattern defines
      which entries to match, and the output pattern specifies which channels to overwrite upon a match.

    * `forward(input: CBTensor) -> CBTensor`:
      Processes a batch of tensors and selectively updates the channels specified in `output_spec` where
      patterns match. If no match is found for an entry, that entry remains unchanged. If more than one match
      is found, an error is raised.

      The tensor is processed in a vectorized fashion, ensuring efficiency across potentially large batches.
    """
    def __init__(self,
                 input_spec: CBTensorSpec,
                 output_spec: CBTensorSpec,
                 device: torch.device = None,
                 dtype: torch.dtype = torch.long
                 ):

        super().__init__()

        #
        self.device = device
        self.dtype = dtype

        # Define the specs
        self.input_spec = input_spec
        self.output_spec = output_spec

        # Define the indirection registries
        indirection_addresses = torch.zeros([0, input_spec.total_width], dtype=dtype, device=device)
        indirection_results = torch.zeros([0, output_spec.total_width], dtype=dtype, device=device)

        self.indirection_addresses = CBTensor(input_spec, indirection_addresses)
        self.indirection_results = CBTensor(output_spec, indirection_results)
    def standardize_this(self,
                    inputs: Union[CBTensor, Dict[str, Union[int, List[int], torch.Tensor]]]
                    )->CBTensor:
        """
        Ensures the information being stored here will be presented and can be utilized in terms of
        CBTensors. These can later be concatenated.

        :param inputs:
            - The inputs to standardize.
            - Should in concept contain a single CBTensor entry, and it's contents.
        :return:
            - A standardized CBTensor.
        """
        if isinstance(inputs, dict):
            # We are dealing with specification by dictionary.
            # Standardize it as a collection of tensors
            standard_dict = {
                key : torch.tensor([item], dtype=self.dtype, device=self.device) if isinstance(item, int) else
                torch.tensor(item, dtype=self.dtype, device=self.device) if isinstance(item, list) else
                item
                for key, item in inputs.items()
            }
            # Now create the cb  tensors
            tensor = CBTensor.create_from_channels(standard_dict)
        else:
            tensor = inputs

        if tensor.dim() == 0:
            tensor = torch.unsqueeze(tensor, 0)
        elif tensor.dim() > 1:
            raise ValueError(f"Cannot define more than one pattern at a time")
        return tensor

    def register(self,
                 input_pattern: Dict[str, Union[int, List[int], torch.Tensor]],
                 output_pattern: Dict[str, Union[int, List[int], torch.Tensor]]
                 ):
        """
        Registers a particular lookup pattern to be associated with a particular
        lookup result.


        :param input_pattern: The pattern to match. Must corrolate cleanly with the input schema
        :param output_pattern: The pattern to match. Must corrolate cleanly with the output schema.
        """

        input_pattern = self.standardize_this(input_pattern)
        output_pattern = self.standardize_this(output_pattern)

        self.indirection_addresses = torch.cat([self.indirection_addresses, input_pattern], dim=0)
        self.indirection_results = torch.cat([self.indirection_results, output_pattern])

    def forward(self, input: CBTensor) -> CBTensor:
        """
        Performs the forward match then replace process

        :param input: The CBTensor to match against
        :return: Same shape CBTensor, but some channel features might have been overwritten.
        """

        # Setup for a broadcast based index search. We are going to take every
        # pattern, and see if it matches on every index

        tensor = input.rebind_to_spec(self.input_spec, allow_channel_pruning=True)
        tensor = torch.flatten(tensor, 0, -1) #(batch_dim, 1)
        index_matches = torch.eq(torch.unsqueeze(tensor, dim=-1), self.indirection_addresses)  # (batch_dim, num_patterns). Bool

        # If no matches were found, or excessive matches were found,
        # we complain. Figure out the entries with no matches.
        if torch.any(index_matches.sum(dim=-1) > 1):
            raise ValueError("More than one match found. This is insane")
        matches = (index_matches.sum(dim=-1) == 1)
        matched_tensors: CBTensor = tensor[matches]
        replacements = matches
        index_matches = index_matches[matches, :]
        matches = matches[matches]


        # Get the associated index from num_patterns. Use vector indexing to extract
        # the result.

        indices = torch.arange(self.indirection_results.shape[0], device=input.device)
        while indices.dim() < index_matches.dim():
            indices = indices.unsqueeze(0)
        indices = indices.expand_as(index_matches)
        items = indices[index_matches]
        outcomes = self.indirection_results[items]


        # Overwrite the content in the matched tensors
        matched_tensors = matched_tensors.set_channels(outcomes)
        tensor[replacements] = matched_tensors
        tensor = torch.reshape(tensor, input.shape)

        return tensor