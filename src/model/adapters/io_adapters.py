"""
IO adapters are responsible, generally, for embedding incoming tensor data into
meaningful embeddings, or converting processed embeddings into predictive distributions.
They separate the IO portion of the model from the core logic.

The core IO adapter is defined here, along with some expected cases. Also
defined is the registry that the builders are expected to work with. The
registry is capable of displaying data from the docstrings of the IOAdapters
and details on the setup requirements

This will later allow the creation of a gui "wizard" that will automatically create
fields to be filled when selecting an io adapter to be utilized. Which should make
config much less tedious.
"""

import torch
import warnings
import inspect
import textwrap
from . import helper_utils
from torch import nn
from abc import ABC, abstractmethod
from typing import Dict, Type, Any, Callable, List, Tuple, Union, Optional


class IORegistry:
    """
    The IO registry keeps track of the various IOAdapter that exist,
    and their setup requirements. Each IOAdapter has to be manually
    registered after the class is created.

    Registration should include information on what keywords need
    to be passed in on setup, and what types they map to.
    """

    @property
    def names(self)->List[str]:
        return list(self.setup_registry.keys())

    def validate_io_adapter_registered(self, name: str):
        """
        Validates whether a particular IO adapter has
        been registered. Raises an error if not available.
        :param name: The name of the adapter to look for
        :raises: KeyError, if the adapter has not been registered.
        """
        if name not in self.model_registry:
            raise KeyError(f"Io adapter of name '{name}' has not been registered and cannot be manipulated.")

    def __init__(self):
        self.model_registry: Dict[str, IOAdapter] = {}
        self.setup_registry: Dict[str, Type[Any]] = {}
    def register(self,
                 name: str,
                 config_spec: Dict[str, Type],
                 adapter: Type["IOAdapter"],
                 ):
        """
        Registers a IOAdapter to be associated with a particular
        mode of operation.

        :param name: The name of the registry.
        :param config_spec: A dictionary that maps keywords
               to types. Used to validate config dictionaries, and even
               tell us what those dictionaries should contain.
        """
        assert isinstance(config_spec, dict), f"Item is not a config dict, '{config_spec}'"
        assert issubclass(adapter, IOAdapter), f"Item is not an io adapter '{adapter}'"

        if name in self.model_registry:
            warnings.warn(f"Warning! Overwriting IOAdapter of name '{name}'")

        self.model_registry[name] = adapter
        self.setup_registry[name] = config_spec

    def register_decorator(self,
                           name: str,
                           config_spec: Dict[str, Type]
                           )->Callable[[Type["IOAdapter"]], None]:
        """
        Creates a decorator for registering an IOAdapter. You
        first specify the setup specification in terms of the
        keywords and the type.

        Then you get a callback to use

        :param config_spec: A mapping of keywords to their required types
        :return: A decorator capable of being called with and registering an
                 IO adapter.
        """
        def decorator(adapter: Type[IOAdapter]):
            self.register(name, config_spec, adapter)
            return adapter
        return decorator

    def setup(self,
              name: str,
              config: Dict[str, Type]
              )->"IOAdapter":
        """
        Set up a new instance of a particular type of adapter
        based on the provided config.

        :param name: The given name of the adapter
        :param config: The config for the name
        :return: An io adapter instance that has been setup
        """

        self.validate_io_adapter_registered(name)
        expected_config = self.setup_registry[name]
        helper_utils.validate_config(config, expected_config)
        return self.model_registry[name].setup(**config)

    def get_structure(self, name)->Dict[str, Type]:
        """
        Gets the structure associated with a particular
        io adapter. Smart use of this will allow the construction
        of an automated wizard.

        :param name: The name of the given adapter
        :return: The config spec dictionary.
        """
        self.validate_io_adapter_registered(name)
        return self.setup_registry[name]
    def get_documentation(self, name: str)->Tuple[str, str]:
        """
        Gets the documentation that was declared on the io adapter
        with a given name. This consists of the class documentation,
        and the setup documentation.

        :param name: The name of the io adapter to get documentation about
        :return: The class docstring
        :return: The setup docstring
        """
        self.validate_io_adapter_registered(name)
        subclass = self.model_registry[name]
        class_docstring = inspect.getdoc(subclass)
        setup_docstring = inspect.getdoc(subclass.setup)
        return class_docstring, setup_docstring



registry = IORegistry()

class IOAdapter(ABC, nn.Module):
    """
    Abstract base class for IO adapters. Responsible for converting model inputs
    into embeddings and output embeddings into distributions or samples.

    This class is abstract and should not be instantiated directly. Subclasses
    should implement the abstract methods and be registered for specific modes
    of operation.
    """

    def __init__(self):
        super(IOAdapter, self).__init__()
        if type(self) is IOAdapter:
            raise NotImplementedError("IOAdapter is an abstract class and cannot be instantiated directly.")

    @classmethod
    @abstractmethod
    def setup(cls, **config: Dict[str, Any]) -> 'IOAdapter':
        """
        Creates and configures a new IOAdapter instance based on the given config.

        :param config: A dictionary containing configuration parameters.
        :return: An instance of the subclassed IOAdapter, configured for the specified mode.
        """
        pass

    @abstractmethod
    def embed_input(self, input: torch.Tensor | Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """
        Converts incoming data into a set of embeddings.

        :param input: A tensor of shape (batch, ..., channels) or (batch, ...).
        :return: A tensor of embeddings of shape (batch, ..., embedding).
        """
        pass

    @abstractmethod
    def create_distribution(self,
                            embeddings: torch.Tensor,
                            schema: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Converts embeddings into a distribution or set of distributions.

        :param embeddings: A tensor of shape (batch, ..., embedding).
        :param schema: Can sometimes be used to pass in additional information about how to interpret embeddings.
        :return: A tensor or tuple of tensors representing the predicted distribution.
        """
        pass

@registry.register_decorator(name = "vocabulary_adapter",
                             config_spec = {"embedding_dim" : int, "vocabulary_size" : int})
class VocabularyIOAdapter(IOAdapter):
    """
    A subclass of IOAdapter designed for handling vocabulary-based problems,
    such as text or some image problems. It uses embeddings and a logit
    projection layer to process input data and produce output distributions.
    It expects to process integer inputs within the vocabulary.
    """

    @staticmethod
    def is_int_tensor(tensor):
        return tensor.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]

    def __init__(self, embedding_bag: nn.EmbeddingBag, logits: nn.Linear):
        """
        Initializes the VocabularyIOAdapter with pre-existing layers.

        :param embedding_bag: The vocabulary embeddings layer.
        :param logits: The logit projection layer.
        """
        super(VocabularyIOAdapter, self).__init__()
        self.embedding_bag = embedding_bag
        self.logits = logits

    @classmethod
    def setup(cls, embedding_dim: int, vocabulary_size: int) -> 'VocabularyIOAdapter':
        """
        Sets up the VocabularyIOAdapter by creating and configuring the necessary layers.

        :param embedding_dim: The dimension of the embeddings of concern
        :param vocabulary_size: The dimension of the vocabulary of concern.
        :return: An instance of VocabularyIOAdapter with the layers properly set up.
        """
        # Create the required layers based on the configuration
        embeddings = nn.EmbeddingBag(vocabulary_size, embedding_dim, mode="sum")
        logits = nn.Linear(embedding_dim, vocabulary_size)
        return cls(embeddings, logits)

    def embed_input(self, input_data: torch.Tensor | Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Converts incoming data into a set of embeddings using the embeddings layer.

        :param inputdata: A tensor of shape (batch, ..., ) where each element is an index in the vocabulary.
        :return: A tensor of embeddings of shape (batch, ..., embedding_dim).
        """
        if isinstance(input_data, torch.Tensor):
            # We are just being passed ordinary collections of integers. Setup code
            # to process them as though we are using embeddings rather than embedding bags.

            # Add extra dimension for embedding bag to sum over,
            # then create per-element weights of 1
            original_shape = input_data.shape
            input_data = input_data.flatten().unsqueeze(-1)
            output_data = self.embedding_bag(input_data)
            output_data = output_data.unflatten(dim=0, sizes=original_shape)
            return output_data

        if isinstance(input_data, tuple) and len(input_data) == 2:
            vocab_indices, probabilities = input_data
            if not self.is_int_tensor(vocab_indices):
                msg = """
                Passed vocab indices was not a tensor of integers. This 
                likely means you got the probabilities and vocab tensors
                swapped around in your tuple.
                """
                msg = textwrap.dedent(msg)
                raise ValueError(msg)

            # embedding bags only want to work with 2d tensors of
            # batches then bags.
            # Okay, we will GIVE them 2d tensors, then unflatten them!

            original_shape = vocab_indices.shape[:-1]
            vocab_indices = vocab_indices.flatten(0, -2)
            probabilities = probabilities.flatten(0, -2)
            outcome = self.embedding_bag(vocab_indices, per_sample_weights=probabilities)
            return outcome.unflatten(dim=0, sizes=original_shape)

        raise ValueError(f"Unsupported input data: {input_data}")

    def create_distribution(self, embeddings: torch.Tensor, schema = None) -> torch.Tensor:
        """
        Converts embeddings into a distribution using the logits projection layer.

        :param embeddings: A tensor of shape (batch, ..., embedding_dim).
        :return: A tensor of logits of shape (batch, ..., vocabulary_size).
        """
        return self.logits(embeddings)

@registry.register_decorator(name="rms_image_adapter",
                             config_spec={"input_channels" : int,
                                          "embedding_dim" : int,
                                          "normalize" : bool,
                                          "max_image_value": float
                                          })
class RMSImageIOAdapter(IOAdapter):
    """
    A subclass of IOAdapter designed for handling image-like, channel-based data.
    This adapter supports normalization and linear projection to and from embeddings.

    The `RMSImageIOAdapter` can normalize the data over a specified maximum value and then
    project the channels into a specified embedding dimension. It also provides a reverse operation
    to decode, by direct projection, back into logit based format.
    """

    def __init__(self,
                 embedding_proj: nn.Linear,
                 distribution_proj: nn.Linear,
                 normalize: bool,
                 max_image_value: float):
        """
        Initializes the SimpleImageIOAdapter with linear projection layers for embedding and distribution,
        and optional normalization.

        :param embedding_proj: The linear projection layer for converting channels to embeddings.
        :param distribution_proj: The linear projection layer for converting embeddings back to channels.
        :param normalize: Whether to apply normalization to the input data.
        :param max_image_value: The maximum value for normalization (e.g., 255.0 for 8-bit image data).
        """
        super(RMSImageIOAdapter, self).__init__()
        self.embedding_proj = embedding_proj
        self.distribution_proj = distribution_proj
        self.normalize = normalize
        self.max_image_value = max_image_value

    @classmethod
    def setup(cls,
              input_channels: int,
              embedding_dim: int,
              normalize: bool = False,
              max_image_value: float = 255.0) -> 'RMSImageIOAdapter':
        """
        Sets up the SimpleImageIOAdapter by creating the necessary layers and configurations.

        :param input_channels: Number of input channels (e.g., 3 for RGB).
        :param embedding_dim: Dimension of the target embedding.
        :param normalize: Whether to apply normalization to the input data.
        :param max_image_value: The maximum value for normalization (e.g., 255.0 for 8-bit image data).
        :return: An instance of SimpleImageIOAdapter with the layers properly set up.
        """
        embedding_proj = nn.Linear(input_channels, embedding_dim)
        distribution_proj = nn.Linear(embedding_dim, input_channels)

        return cls(embedding_proj=embedding_proj, distribution_proj=distribution_proj, normalize=normalize,
                   max_image_value=max_image_value)

    def embed_input(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Converts incoming image data into a set of embeddings using a linear projection layer.

        :param input_data: A tensor of shape (batch, ..., channels), where each element is in the range [0, max_image_value].
        :return: A tensor of embeddings of shape (batch, ..., embedding_dim).
        """
        if self.normalize:
            input_data = input_data / self.max_image_value

        return self.embedding_proj(input_data)

    def create_distribution(self, embeddings: torch.Tensor, schema=None) -> torch.Tensor:
        """
        Converts embeddings into a distribution (e.g., image channels) using a linear projection layer.

        :param embeddings: A tensor of shape (batch, ..., embedding_dim).
        :return: A tensor of shape (batch, ..., channels), where each element is in the range [0, max_image_value].
        """
        output = self.distribution_proj(embeddings)

        if self.normalize:
            output = output * self.max_image_value

        return output

setup_spec = {"embedding_dim" : int,
              "logit_size" : int,
              "schemas" : Dict[str, torch.Tensor]}
@registry.register_decorator("controller_adapter",
                             setup_spec
                             )
class ControllerIOAdapter(IOAdapter):
    """
    A subclass of IO adapter designed for interfacing
    and manipulating data to control mode selection,
    image shape, and other such important details.

    The control distribution produced is associated with two
    details. One is a control mode, which will be an integer
    addressing the available schemas. Of note, the first
    schema is always associated with selecting a mode.

    Second, there will be the actual predictions, per dimension,
    to create and the mask of unused locations.
    """

    @classmethod
    def setup(cls,
              embedding_dim: int,
              logit_size: int,
              schemas: Dict[str, torch.Tensor]
              )->"ControllerIOAdapter":
        """
        Sets up a ControllerIOAdapter for the specified modelling
        process with the specified schemas. Checks along the way that
        everything is compatible.

        :param logit_size: The size of the logit layer to create for control purposes
        :param schemas: A mapping of each mode name to the schema for it's shape
        :return: A ControllerIOAdapter usable for the control process.
        """

        # Validate a control schema exists. Make sure the control schema
        # is at the front of the dictionary.

        assert "control"  in schemas, "Keyword 'control' must be in schemas"
        assert schemas["control"][0] == torch.tensor([len(schemas)-1]), "control must encode the number of mode schemas"
        assert next(iter(schemas.keys())) == "control", "control schema needed to be at front of the dictionary."

        # Validate shape information, and verify that sufficient logits exist
        control_length = schemas["control"].shape[0]
        for schema in schemas.values():
            if control_length != schema.shape[0]:
                msg = f"""\
                Schema with dimensions of length '{schema.shape[0]}' found. However, this 
                does not match control length of '{control_length}'. Make sure to pad all
                to common length!
                """
                msg = textwrap.dedent(msg)
                raise ValueError(msg)
            if schema.prod() > logit_size:
                msg = f"""\
                Schema not compatible with logit size. Schema of 
                shape '{schema}' requires '{schema.prod()}' number of
                logit slots. However, logit slots only go up to {logit_size}
                """
                msg = textwrap.dedent(msg)
                raise ValueError(msg)

        # Create the instance
        num_modes = len(schemas)
        schema_reference = torch.stack([schema for schema in schemas.values()], dim=0)
        embedding_layer = nn.Embedding(num_modes, embedding_dim)
        logits_layer = nn.Linear(embedding_dim, logit_size)
        logit_separator = LogitSeparator()
        return cls(embedding_layer, logits_layer, logit_separator, schema_reference)

    def __init__(self,
                 embedding_layer: nn.Embedding,
                 logit_projector: nn.Linear,
                 logit_separator: nn.Module,
                 schema_reference: torch.Tensor,
                 ):
        """

        :param embedding_size: The size of the embeddings to accept
        :param logit_projector: The logit projector
        :param logit_separator: Capable of separating layers based on the schema specification
        :param schema_reference: A tensor that associates integers with particular schemas.
        """
        super().__init__()
        self.dimensions = schema_reference.shape[-1]
        self.embedding_layer = embedding_layer
        self.logit_projector = logit_projector
        self.schema_reference = schema_reference
        self.logit_separator = logit_separator

    def embed_input(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Embedding is accomplished by calling into the embedding
        vocabulary. The vocabulary is sized to be able to operate
        in all the modes.

        :param tensor: A tensor of integers representing control modes
        :return: The embedded input
        """
        return self.embedding_layer(tensor)

    def create_distribution(self,
                            embeddings: torch.Tensor,
                            schema: torch.Tensor
                            ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Creates a logit distribution with D dimensions
        where D is the number of logit dimensions we need to worry
        about, and a mask indicating what entries were actually used

        :param embeddings:
            * The embeddings to create the distribution from.
            * Shape: (batch, items, embeddings)
        :parame schema:
            * The schema index each item is generated under
            * index zero means a mode select task, others
              refer to specific modes.
            * Shape (batch, items)
        :return: A distribution based on the logit placed in the specified
                 schema distribution. The mask will be true only for
        """

        logits = self.logit_projector(embeddings)
        schemas = self.schema_reference.index_select(dim=0, index = schema)
        logits, mask = self.logit_separator(schemas, logits)
        return logits, mask


class LogitSeparator(nn.Module):
    """
    Separates logits in Shared Logit Space into their
    per-dimension format.
    """

    def __init__(self):
        super().__init__()
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

        logit_indices = torch.arange(logits.shape[-1], device=logits.device)
        while logit_indices.dim() < logits.dim():
            logit_indices = logit_indices.unsqueeze(0)

        # Unsqueeze for interaction
        logit_indices = logit_indices.unsqueeze(-2)
        start_of_zone = start_of_zone.unsqueeze(-1)
        end_of_zone = end_of_zone.unsqueeze(-1)

        # Create mask

        mask = torch.logical_and(logit_indices >= start_of_zone, logit_indices < end_of_zone)
        return mask

    def forward(self, schemas: torch.Tensor, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass, which actually takes the schemas and logits
        :param schemas: A schema for each logit. Shape (..., D). D is dimensions
        :param logits: A logit tensor. Shape (..., L). L is logit lenght.
        :return: A tensor of shape (..., D, L). The logit has been broken up to
                 associate portions with the relevant dimensions.
        :return: A bool tensor of shape (..., D, L). The tensor indicates whether an element
                 was not padding, with True meaning not padding.
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
        mask = separation_mask.gather(dim=-1, index=indices)

        return logits, mask
