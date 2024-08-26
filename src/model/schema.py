from typing import List, Dict

import torch


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
