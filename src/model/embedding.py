"""
This module handles the conversion of inputs into embeddings, embeddings into
distributions, and embeddings into outputs. The embeddings themselves are
processed elsewhere, but this module manages the integration and conversion
required for multimodal data.

### Data Representations: Payload

Payload data is represented using **mode-dimension channels**, where each input or
target is defined by two integers: the mode (indicating the type of data, such
as text or image) and the associated data value. The module supports various
dimensionalities (e.g., text as 1D, images as 3D), and all data flows through
the model in batches. Zero-padding is used to manage different dimensional
requirements across modes, allowing diverse data types to coexist within the
same batch.

For instance, text data may be represented as `[2, 1245, 0, 0]` in a padded
format, when coexisting with image data, such as RGB values represented as
`[1, 125, 54, 155]`. The text data is padded to match the length of the image
case, ensuring compatibility during batch processing while retaining the
structure of each mode.

### Data Representation: Headers

Header data is processed in terms of an array of headers, each with a fixed
width. What is in the headers is still somewhat under development,

The header data is needed primarily in order to understand how to encode
positional embeddings and encodings.

### Schemas

The **schema tensor** is a static 2D tensor with shape `(modes, 1 + D)`,
where `D` is the maximum number of dimensions for any mode. The schema
defines how data is structured for each mode. Unused dimensions in the schema
are filled with zeros to ensure consistency across modes.

For example:
- Text data may have a schema `[10000, 0, 0]`, indicating up to 10,000 tokens
  (1D), with padding for unused dimensions.
- RGB image data could have a schema `[256, 256, 256]`, representing RGB
  color channels (3D).

The schema tensor is immutable once created and is used during both setup and
runtime to ensure that the model correctly processes different data types. It
provides a unified mechanism for handling multimodal data, ensuring that even
with different dimensional structures, all data can be processed together
within the same model.

### Responsibilities

The module's key responsibilities are:
1. Embedding mode-dimension channel data for model training.
2. Generating logits according to provided schemas for each mode.
3. Processing losses for each mode of operation.
4. Managing data integration to ensure compatibility between different
   modes within batches.
"""

from dataclasses import dataclass
from typing import List, Callable, Dict, Tuple, Union
from abc import ABC, abstractmethod

import torch
import numpy as np
from torch import nn


##
# Define payload embedding mechanisms.
#
# These will be designed to directly and discretely
# embed certain types of data to vectors.
##

class AbstractModeEmbedding(nn.Module, ABC):
    """
    Abstract class for embedding mode-dimension channel data into actual embeddings.

    This class provides the foundational structure for converting mode-dimension
    channel data into embeddings. The embedding mechanism itself is left to be
    implemented by the user in the `embed` method. The superclass pledges to handle
    mode address translation and provide local mode indices, while the subclass only
    needs to deal with the local modes specified during initialization.

    ### Key Concepts

    - **Mode-Dimension Channels**: The input data is provided in the form of mode-dimension
      channels, where the first entry in each row corresponds to the mode (e.g., text,
      image), and the following entries describe the dimensions of the data in that mode.
    - **Mode Address Translation**: The superclass automatically handles the translation
      between global schema indices and local indices that correspond to the supported
      modes. The user (subclass) is only responsible for processing local mode indices
      provided by the superclass.
    - **Schema and Supported Modes**: The user pledges support for specific modes by
      passing the `supported_modes` tensor during initialization. The `.supported_schemas`
      or `.supported_modes` tensors can be used to determine which modes and in what
      order the input data will be provided for embedding.

    ### How to Implement

    The user must implement the `embed` method, which receives input in terms of mode-dimension
    specifications and is responsible for returning embeddings in a 2D tensor format. The `embed`
    method processes the input data based on the modes specified, converting the mode-dimension
    channels into fixed-length embeddings.

    ### Input to `.embed` Method

    - **Shape**: `(batch, items, mode + dimensions)`
      - The first column is the mode, now localized (i.e., corresponds to the
        supported modes selected during initialization and pledged by the superclass).
      - The remaining columns represent the dimensions of the data for that mode.
      - You can examine `.supported_schemas` or `.supported_modes` to know which
        modes will be provided and in what order they will arrive.

    For example:
    - If the input is text data with mode 2 and the dimensions `[1245]`, and the input
      was globally represented as `[2, 1245]`, after mode address translation, the input
      to `embed` will look like `[0, 1245]` if mode 2 was the first supported mode.

    ### Output of `.embed` Method

    The `.embed` method should return:
    - **Shape**: `(batch, items, embedding_dim)`
      - The output should be a tensor of embeddings corresponding to the input data.
      - For unsupported modes, the output should handle NaN values, ensuring
        the embeddings tensor aligns with the expected format.

    ### Contract with the World

    When the layer is called with mode-dimension data of shape `(…, mode + dimensions)`,
    it processes the input data and conceptually prepares data suitable for insertion
    into a tensor of shape `(…, embedding_dim)`. However, the layer does not directly
    create this output tensor.

    Instead, the layer returns two key outputs:
    1. A **mask** that identifies which entries should be placed in the output tensor.
    2. A set of **processed entries**, which are the actual embeddings generated for
       the pledged modes.

    These two outputs allow external processes to construct the final output tensor.
    The mask can be used to insert the processed entries into the pre-existing output
    tensor, as in `output[mask] = entries`.

    ### Attributes

    - **embedding_dim**: Dimension of the output embeddings.
    - **schemas**: A 2D tensor containing the schema for all available modes.
    - **supported_modes**: A 1D tensor containing the indices of supported schema modes
      for this layer.
    - **supported_schemas**: A subset of the `schemas` tensor that corresponds to the
      modes this embedding layer is responsible for.
    - **address_translation_map**: A translation map that converts global mode indices
      into local indices for supported modes.
    """

    def __init__(self, embedding_dim: int, schemas: torch.Tensor, supported_modes: torch.Tensor):
        """
        Initialize the AbstractEmbedding class.

        Parameters:
        ----------
        embedding_dim : int
            The dimension required for the output embeddings.
        schemas : torch.Tensor
            A 2D integer tensor containing all available schemas. Shape is (modes, dimensions).
        supported_modes : torch.Tensor
            A 1D integer tensor specifying the modes that this class is responsible for encoding.
        """
        super().__init__()

        assert isinstance(embedding_dim, int) and embedding_dim > 0, "embedding_dim must be a positive integer."
        assert isinstance(schemas, torch.Tensor) and schemas.dim() == 2, "schemas must be a 2D tensor."
        assert isinstance(supported_modes,
                          torch.Tensor) and supported_modes.dim() == 1, "supported_modes must be a 1D tensor."

        self.embedding_dim = embedding_dim
        self.schemas = schemas
        self.supported_modes = supported_modes
        self.supported_schemas = schemas.index_select(0, supported_modes)

        self.address_translation_map = self.compute_mode_address_translation_map(schemas, supported_modes)

    def compute_mode_address_translation_map(self, schemas: torch.Tensor,
                                             supported_modes: torch.Tensor) -> torch.Tensor:
        """
        Compute the address translation map for converting global schema indices to local indices.

        Parameters:
        ----------
        schemas : torch.Tensor
            The global schema tensor of all modes.
        supported_modes : torch.Tensor
            The modes supported by this embedding layer.

        Returns:
        -------
        torch.Tensor
            A translation map that converts global mode indices to local mode indices. If a
            mode is not supported, its entry will be set to -1.
        """
        address_translation_map = -torch.ones(schemas.size(0), dtype=torch.int64, device=schemas.device)
        address_translation_map[supported_modes] = torch.arange(supported_modes.size(0), device=schemas.device)

        return address_translation_map

    def mode_address_translation(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Perform mode address translation by converting global mode indices to local ones.

        Parameters:
        ----------
        inputs : torch.Tensor
            Input tensor with shape (..., mode + dimensions).

        Returns:
        -------
        torch.Tensor
            Translated inputs where the mode slot contains local mode indices. Non-supported
            modes will have a mode of -1.
        """
        input_modes = inputs[..., 0]
        dimensions = inputs[..., 1:]

        final_modes = self.address_translation_map.index_select(0, input_modes)
        output = torch.cat([final_modes.unsqueeze(-1), dimensions], dim=-1)

        return output

    @abstractmethod
    def embed(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Abstract method for embedding data. The implementation should return embeddings
        for the provided mode-dimension channel data.

        Parameters:
        ----------
        inputs : torch.Tensor
            Input tensor with shape (batch, items, mode + dimensions).

        Returns:
        -------
        torch.Tensor
            Output tensor with shape (batch, items, embedding_dim).
        """
        pass

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for embedding the incoming data. This involves mode address translation
        and calling the user-implemented embedding mechanism.

        Parameters:
        ----------
        inputs : torch.Tensor
            Input tensor with shape (..., mode + dimensions). Unused dimensions are filled with zeros.

        Returns:
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple containing:
            1. The mask for valid entries.
            2. The embeddings for the translated inputs.
        """
        # Perform mode address translation
        translated_inputs = self.mode_address_translation(inputs)
        translation_mask = translated_inputs[..., 0] != -1

        # Extract valid inputs and flatten to (items, channels)
        valid_inputs = translated_inputs[translation_mask]
        if valid_inputs.numel() == 0:
            return translation_mask, torch.empty((0, self.embedding_dim), device=inputs.device)

        # Compute embeddings
        #
        # Return should be in terms of (batch, embedding_dim)
        embeddings = self.embed(valid_inputs)
        assert embeddings.dim() == 2
        assert embeddings.shape[-1] == self.embedding_dim

        # Return mask and embeddings
        return translation_mask, embeddings


class TokenEmbeddings(AbstractModeEmbedding):
    """
    An embedding mechanism designed to use learnable parameters as embeddings,
    as almost all transformer models today do. This is the normal embedding
    implementation and does not make assumptions that words are related in
    any way.

    Internally, we implement a version of strided sparse logic to map a
    single-dimensional embedding onto all the expressed modes of operation.
    """

    def compute_num_elements_per_mode(self, supported_modes: torch.Tensor) -> torch.Tensor:
        """
        Computes the number of elements per mode by taking the product of active dimensions.

        This method masks inactive elements with 1 to ensure they do not affect the
        product computation, allowing us to compute the correct number of elements
        per mode.

        Returns:
        -------
        torch.Tensor
            A tensor containing the number of elements for each mode.
        """
        inactive_elements = supported_modes == 0
        supported_schemas = supported_modes.masked_fill(inactive_elements, 1)

        # Product of active dimensions
        num_elements_per_mode = supported_schemas.prod(dim=-1)
        return num_elements_per_mode

    @staticmethod
    def compute_sparse_mode_offsets(num_elements_per_mode: torch.Tensor) -> torch.Tensor:
        """
        Compute the offsets for each mode using a sparse logic approach.
        This calculates where each mode's block of elements starts in
        the embedding array.

        Returns:
        -------
        torch.Tensor
            A tensor of offsets indicating where each mode's embeddings start.
        """
        # Insert a tensor of zeros to ensure the first mode offset is zero
        num_elements_per_mode = torch.cat(
            [torch.tensor([0], device=num_elements_per_mode.device), num_elements_per_mode], dim=0)

        # Compute the cumulative sum to get the mode offsets
        mode_offsets = num_elements_per_mode.cumsum(dim=0)[:-1]

        return mode_offsets

    @staticmethod
    def compute_dimension_strides(supported_schemas: torch.Tensor) -> torch.Tensor:
        """
        Compute the strides for each schema across all dimensions.

        The strides represent how many elements to skip in memory to move from
        one element to the next along each dimension.

        Returns:
        -------
        torch.Tensor
            A tensor of strides for each dimension in the schema.
        """
        # Flip the schema to compute strides starting from the smallest dimension
        supported_schemas = supported_schemas.flip(dim=-1)

        # Concatenate a tensor of ones to ensure a stride of 1 for the smallest dimension
        stride_product_setup = torch.cat([torch.ones_like(supported_schemas[:, :1]), supported_schemas], dim=-1)

        # Compute the cumulative product of strides, slicing correctly
        strides = stride_product_setup.cumprod(dim=-1)[:, :-1]

        # Flip back to restore the original order
        return strides.flip(dim=-1)
    def __init__(self,
                 embedding_dim: int,
                 schemas: torch.Tensor,
                 expressed_modes: torch.Tensor):
        """
        Initialize the TokenEmbeddings class.

        Parameters:
        ----------
        embedding_dim : int
            The dimension of the output embeddings.
        schemas : torch.Tensor
            A 2D integer tensor containing the schema for all available modes.
        expressed_modes : torch.Tensor
            A tensor representing the modes this class is responsible for.
        """
        super().__init__(embedding_dim, schemas, expressed_modes)

        self.num_elements_per_mode = self.compute_num_elements_per_mode(self.supported_modes)
        self.sparse_mode_offsets = self.compute_sparse_mode_offsets(self.num_elements_per_mode)
        self.dimension_strides = self.compute_dimension_strides(self.supported_schemas)

        # Create the embedding array based on the number of elements
        self.embeddings = nn.Embedding(int(self.num_elements_per_mode.sum()), self.embedding_dim)

    def embed(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Embeds the actual inputs using sparse mode logic and strided dimension logic.

        Parameters:
        ----------
        inputs : torch.Tensor
            The inputs to process based on the local mode-dimensions data specification.
            Shape: (batch, 1 + dimensions)

        Returns:
        -------
        torch.Tensor
            The outputs to produce in terms of embeddings.
            Shape: (batch, embedding_dim)
        """
        modes, dimensions = inputs[:, 0], inputs[:, 1:]

        # Gather the strides and mode offsets
        strides = self.dimension_strides.gather(dim=0, index=modes)
        converted_index = self.sparse_mode_offsets.gather(dim=0, index=modes)

        # Calculate the final index using the strides and dimensions
        converted_index += torch.sum(dimensions * strides, dim=-1)

        return self.embeddings(converted_index)

class LinearEmbeddings(AbstractModeEmbedding):
    """
    Creates a linear embedding mechanism that processes the entire input (modes + dimensions)
    together, allowing the model to learn how to handle the modes and dimensions jointly.

    ### When to Use
    This layer is suitable when dealing with data where nearby numbers are related,
    such as in RGB image data, continuous shapes, or structured numerical data.

    It is a less appropriate choice for data where nearby numbers are completely
    unrelated, such as tokens in natural language processing, where the relationships
    between elements are more categorical than spatial or numerical.

    ### How to Use
    Initialize the `LinearEmbeddings` layer with the required parameters and then call
    the layer on the input data. The input should contain both the mode and dimension
    data in the same tensor, formatted as `(batch, mode + dimensions)`. The model will
    handle both mode and dimension jointly.

    Example:
    ```python
    embedding_dim = 128
    internal_dims = 64
    num_layers = 3
    schemas = torch.randint(0, 10, (5, 3))  # Example schema with 5 modes, 3 dimensions each
    expressed_modes = torch.tensor([0, 1, 2, 3, 4])  # 5 expressed modes

    # Create an instance of LinearEmbeddings
    linear_embedding_layer = LinearEmbeddings(embedding_dim, schemas, expressed_modes, internal_dims, num_layers)

    # Example input (batch, mode + dimensions)
    inputs = torch.randint(0, 10, (16, 4))  # Batch size 16, mode + 3 dimensions

    # Call the layer directly to get the embeddings
    output_embeddings = linear_embedding_layer(inputs)
    print(output_embeddings.shape)  # Should be (16, 128) -> (batch, embedding_dim)
    ```

    ### Implementation Details
    This class processes the entire input (modes + dimensions) together using several
    linear layers. The input is projected into a tensor of shape `internal_dims`,
    processed through `num_layers` with linear projections and ReLU activations, and
    then projected to the final embedding dimension. The model learns to encode both
    modes and dimensions jointly, rather than separating them.
    """

    def __init__(self, embedding_dim: int, schemas: torch.Tensor, expressed_modes: torch.Tensor,
                 internal_dims: int, num_layers: int):
        """
        Initialize the LinearEmbeddings class.

        Parameters:
        ----------
        embedding_dim : int
            The dimension of the output embeddings.
        schemas : torch.Tensor
            A 2D integer tensor containing the schema for all available modes.
        expressed_modes : torch.Tensor
            A tensor representing the modes this class is responsible for.
        internal_dims : int
            The size of the internal projection tensor.
        num_layers : int
            The number of linear layers to process the data.
        """
        # Call the superclass constructor first
        super().__init__(embedding_dim, schemas, expressed_modes)

        self.internal_dims = internal_dims
        self.num_layers = num_layers

        # Define the projection layers: projecting from (mode + dimensions) to internal_dims
        self.projection = nn.Linear(schemas.size(1), internal_dims)

        # Define the internal processing layers (num_layers deep)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(internal_dims, internal_dims),
                nn.ReLU()
            ) for _ in range(num_layers)
        ])

        # Define the final projection to embedding_dim
        self.final_projection = nn.Linear(internal_dims, embedding_dim)

    def embed(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Embeds the input data using linear projections and processing layers.

        Parameters:
        ----------
        inputs : torch.Tensor
            The inputs to process based on the local mode-dimensions data specification.
            Shape: (batch, mode + dimensions)

        Returns:
        -------
        torch.Tensor
            The outputs to produce in terms of embeddings.
            Shape: (batch, embedding_dim)
        """
        # Project the entire input (modes + dimensions) into the internal_dims space
        projected = self.projection(inputs)

        # Process through num_layers layers with ReLU activations
        for layer in self.layers:
            projected = layer(projected)

        # Final projection to embedding_dim
        embedding = self.final_projection(projected)

        return embedding

