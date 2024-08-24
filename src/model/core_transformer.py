"""
Concerns the core transformer model we will be working with,
including builders to make a fresh one.

We will be using a transformer that operates with block multimodal
encoding, and that can take a conventional transformer as input.
"""

import torch
import numpy as np
from abc import ABC, abstractmethod
from torch import nn
from typing import List, Dict, Tuple, Protocol, Callable, Optional, Any


## Schema tracker registry

class SchemaRegistry:
    """
    A class to manage and track schemas within a multimodal encoding system.

    The SchemaTracker allows for the registration of different schemas, each identified
    by a unique name. These schemas define how various modes of data (e.g., text, image)
    are structured and processed. The class also facilitates the retrieval of schema
    identifiers and the corresponding schema structures based on these identifiers.

    Key Functions:
    --------------
    - Register schemas with a unique name and validate their structure.
    - Retrieve the ID associated with a registered schema name.
    - Fetch schemas based on provided schema IDs.

    Attributes:
    -----------
    num_dimensions : int
        The number of dimensions supported for the schemas.
    logit_slots : int
        The maximum sum allowed for schema values, representing logit slots.
    names : List[str]
        A list storing the names of registered schemas.
    schemas : torch.Tensor
        A tensor storing the registered schemas.
    schema_dict : Dict[str, int]
        A dictionary mapping schema names to their associated schema IDs.
    """

    def __init__(self, num_dimensions: int, logit_slots: int) -> None:
        """
        Initializes the SchemaTracker with the specified number of dimensions and logit slots.

        Parameters:
        -----------
        num_dimensions : int
            The number of supported target dimensions for schemas.
        logit_slots : int
            The maximum allowed sum of schema values (logit slots).
        """
        self.num_dimensions: int = num_dimensions
        self.logit_slots: int = logit_slots
        self.names: List[str] = []
        self.schemas: torch.Tensor = torch.empty(0, num_dimensions, dtype=torch.int64)  # Initially empty
        self.schema_dict: Dict[str, int] = {}

    def register_schema(self, name: str, schema: List[int]) -> None:
        """
        Registers a new schema with a unique name and validates it.

        Parameters:
        -----------
        name : str
            The name to assign to the schema. Must be unique.
        schema : List[int]
            A list of non-negative integers representing the schema.

        Raises:
        -------
        ValueError:
            If the schema name is already in use, if the schema contains invalid values,
            if the schema is too long, or if the sum of the schema exceeds the logit slots.
        """
        # Validate the name
        if name in self.names:
            raise ValueError(f"Schema name '{name}' is already in use.")

        # Validate the schema
        if not all(isinstance(x, int) and x >= 0 for x in schema):
            raise ValueError("Schema must be a list of non-negative integers.")

        schema_length: int = len(schema)

        if schema_length > self.num_dimensions:
            raise ValueError(f"Schemas with dimensions in excess of {self.num_dimensions} are not supported.")

        # Pad the schema if it's not long enough
        if schema_length < self.num_dimensions:
            schema = schema + [0] * (self.num_dimensions - schema_length)

        # Check the sum against the logit slots
        if sum(schema) > self.logit_slots:
            raise ValueError(f"The sum of the schema exceeds the allowed logit slots ({self.logit_slots}).")

        # Store the schema
        schema_tensor: torch.Tensor = torch.tensor(schema, dtype=torch.int64).unsqueeze(0)
        if self.schemas.numel() == 0:  # If schemas tensor is empty
            self.schemas = schema_tensor
        else:
            self.schemas = torch.cat((self.schemas, schema_tensor), dim=0)

        schema_id: int = len(self.names)
        self.names.append(name)
        self.schema_dict[name] = schema_id

    def get_schema_id(self, schema_name: str) -> int:
        """
        Retrieves the schema ID associated with the given schema name.

        Parameters:
        -----------
        schema_name : str
            The name of the schema.

        Returns:
        --------
        schema_id : int
            The ID associated with the schema.

        Raises:
        -------
        KeyError:
            If the schema name was never registered.
        """
        if schema_name not in self.schema_dict:
            raise KeyError(f"Schema name '{schema_name}' was never registered.")
        return self.schema_dict[schema_name]

    def fetch_schemas(self, schema_ids: torch.Tensor) -> torch.Tensor:
        """
        Fetches a tensor of schemas based on the provided tensor of schema IDs.

        Parameters:
        -----------
        schema_ids : torch.Tensor
            A tensor of integers where each integer represents a schema ID.
            The shape can vary depending on the number of schema IDs provided.

        Returns:
        --------
        schemas : torch.Tensor
            A tensor of schemas corresponding to the provided schema IDs.
            The output shape is (..., num_dimensions), where "..." represents
            the shape of the input tensor, and num_dimensions is the number of
            dimensions in each schema.

        Raises:
        -------
        TypeError:
            If schema_ids is not a tensor of integers.
        """
        # Validate the schema_ids tensor
        if not torch.is_tensor(schema_ids) or schema_ids.dtype not in [torch.int32, torch.int64]:
            raise TypeError("schema_ids must be a torch tensor of integers.")

        return self.schemas[schema_ids]


## Data
#
# Data will consist of grids of integers, with, for example, the ability to specify one of
# 256 colors for each channel, or the tokens.
#
# Image data???
# Encoding: SURE
# Decoding: Uh.... data leak?
# Make it entirly pretend to be operating in an encoding mode?
# WTH would the training task look like?x`

class ModeEmbeddingAdapter(nn.Module):
    """
    A class for containing mode-specific embedding logic.
    """
    def __init__(self,
                 schema: str,
                 embedder: nn.Module,
                 encoding_dim: int,
                 patch_shape: Optional[List[int]] = None,
                 ):
        """

        :param schema: The schema to associate this with
        :param embedder: The embedding mechanism. Should handle ND data as appropriate
        :param patch_shape: One can optionally patch sections as in ViT.
        :param encoding_dim: Embeddings must come out of the adapter with this shape
        """

        super().__init__()

        self.schema = schema
        self.embeddings = embedder
        self.adapter = nn.LazyLinear(encoding_dim)

class ModeAdapter(nn.Module, ABC):
    """
    A mode-specific adapter, designed to convert mode-specific
    data into it's more general format.

    We assume in this that
    """
    def __init__(self,
                 schema: str,
                 schema_registry: SchemaRegistry,
                 embedder: nn.Module,
                 decoder: nn.Module,

                 ):
        """
        The setup process for the mode spec

        :param schema: The schema to select for association
        :param schema_registry: The schema registry to draw from
        :param patch_shape: The shape of each patch, as in ViT.
        :param embeddings: The final embedding dim.
        """

        # General behavior and validation
        self.schema = schema
        self.schema_registry = schema_registry

        # Encoding requirements
        self.linear = nn.Linear(embedded_dim*np.prod(patch_shape), encoding_dim)

    @abstractmethod
    def embed(self, int_grid: torch.Tensor)->torch.Tensor:
        """
        This should be capable of embedding an arbitrary
        :param int_grid:
        :return:
        """
    def encode(self, int_grid: torch.Tensor) -> torch.Tensor:


class DataConverter(nn.Module):
    """
    - Embedding and patching control.
    - Associated with a schema - no schema, no converter can be registere.
    - Encode
    - Decode.
    """




class ConverterRegistry(nn.Module):
    """
    A registry system for registering the mechanisms needed

    """
    def __init__(self, logit_slots: int):
        super().__init__()

        self.logit_slots = logit_slots
        self.encoder_registry = nn.ModuleDict()
        self.decoder_registry = nn.ModuleDict()




class LogitSeparator(nn.Module):
    """
    Separates logits in Shared Logit Space into their
    per-dimension format.
    """
    @staticmethod
    def compute_zone_edges(schemas: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the position at which a logit starts being associated
        with a schema, and ends being associated with a schema.

        :param schemas: The schemas tensor, of shape (..., D), where the integers
                        along d indicate schema diemnsion assignment.
        :return: start: The index at which the logits begin to be associated
        with the given schema. (..., D)
        """

        # Figure out how much of the logit space has already
        # been used up by the time we get to a given schema
        # element.

        start_of_zone = schemas.cumsum(dim=-1)
        start_of_zone = start_of_zone.roll(1, dims=-1)
        start_of_zone[..., 0] = 0

        # Figure out when the logit zone ends

        end_of_zone = start_of_zone + schemas

        return start_of_zone, end_of_zone

    @classmethod
    def create_separation_mask(cls, schemas: torch.Tensor, logits: torch.Tensor)->torch.Tensor:
        """
        Creates a seperation mask capable of separating the logit
        into per-dimension associations.

        :param schemas: The per-logit schema assignment. (..., D) where D is dimensions
        :param logit_length: The length of the logits.
        :return: A tensor of shape (..., D, L)
        """

        # Get the zone edges

        start_of_zone, end_of_zone = cls.compute_zone_edges(schemas)

        # Assign true or false based on whether a particular index
        # lies within an active schema zone.

        logit_indices = torch.arange(logits.shape[-1])
        while logit_indices.dim() < logits.dim():
            logit_indices = logit_indices.unsqueeze(0)

        # Unsqueeze for interaction
        logit_indices = logit_indices.unsqueeze(-2)
        start_of_zone = start_of_zone.unsqueeze(-1)
        end_of_zone = end_of_zone.unsqueeze(-1)

        # Create mask

        mask = torch.logical_and(logit_indices >= start_of_zone, logit_indices < end_of_zone)
        return mask

    def forward(self, schemas: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """
        Forward pass, which actually takes the schemas and logits
        :param schemas: A schema for each logit. Shape (..., D). D is dimensions
        :param logits: A logit tensor. Shape (..., L). L is logit lenght.
        :return: A tensor of shape (..., D, L). The logit has been broken up to
                 associate portions with the relevant dimensions.
        """

        # Compute separation mask.

        separation_mask = self.create_separation_mask(schemas, logits)

        # Multiply mask by logits to separate them

        logits = logits.unsqueeze(-2) * separation_mask

        # Sort logits so that nonzero values come first. This negates any
        # need for mapping into or out of logit space outside this
        # unit.

        indices = torch.argsort(separation_mask, dim=-1, descending=True, stable=True)
        logits = logits.gather(dim=-1, index=indices)

        return logits










