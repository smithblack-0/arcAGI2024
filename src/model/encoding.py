"""
A loader, and also saving, mechanism. The data converter module contains things designed to "tokenize"
or "detokenize" the multimodal content which is being presented. Content is transformed in such a way
that downstream it is straightforward to recover, for each piece of data being processed, what the context
is based on the metadata entries on each tensor


"""

import torch
from typing import Any, List, Dict, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.model.datastructures import TensorChannelSpec


## Registries, for zones, schemas, etcetera


# Block builder mechanism


@dataclass
class BlockMetadata:
    """
    A small dataclass, this exclusively is designed to contain
    metadata about the block. Note that although we call this
    metadata, some of it WILL be visible to the model.
    """
    mode: int
    zone: int
    sequence: int

## Token channel conversion

def create_header
class TokenChannelConverter:
    """
    The token channel encoder is designed to convert to
    or from metadata token channel format.

    It must be provided a BlockMetadata instance, and may then be invoked
    with a tensor. It will insert this tensor into a metadata context for
    downstream decoding.
    """

    # Helper functions

    def create_header(self)->torch.Tensor:
        """Creates a header sequence appropriate for putting in front of a block"""

    def get_shape_details(self,
                          schema: List[int],
                          tokens: torch.Tensor
                          )->Tuple[Tuple[int, ...], int]:
        """Do some basic sanity checking, then get the shape and number of elements."""
        # Sanity check based on the block schema.
        assert tokens.dim() == len(schema)
        assert all(dim <= schema_dim for dim, schema_dim in zip(tokens.shape, schema))

        # Return the shape, and the number of elements
        return tokens.shape, tokens.numel()

    def extend_metadata_tensors(self, metadata: BlockMetadata, length: int)->Dict[str, torch.Tensor]:
        """ Extends the metadata information tensors to match the length """


    def __init__(self,
                 schema: List[int],
                 metadata: BlockMetadata,
                 channel_spec: TensorChannelSpec,
                 ):
        self.metadata = metadata
        self.spec = channel_spec
    def encode(self,
                 schema: List[int],
                 metadata: BlockMetadata,
                 tokens: torch.Tensor
                 )->torch.Tensor:


        shape, numel = self.get_shape_details(tokens, schema)
        channel_tensors = self.extend_metadata_tensors(meta)

        # Create the index mesh and shape mesh
        index_mesh = torch.meshgrid(*[torch.arange(dim) for dim in shape], indexing="ij")
        index_mesh = torch.stack(index_mesh, dim=0)

        # Flatten both tokens and index mesh
        tokens = tokens.flatten().unsqueeze(-1)
        index_mesh = index_mesh.flatten(1)

        # Create metadata.

class BlockTensorBuilder:
    """
    Used to collect and build a tokenized block of
    content by collecting things like mode, shape,
    data, etc as we go.
    """

    def ___init__(self):

        block_contents = {}

        # Define some important meta information.
        block_contents["mode"] = None
        block_contents["zone"] = None
        block_contents["sequence"] = None

        #Define

# Main starts
class AbstractProcessor(ABC):
    """
    The abstract processor object exists to make life easier for
    those implementing a processing strategy by automatically accounting
    for and incorporating metadata info based off of a converted entry.

    It contains several methods that must be implemented by a subclass,
    and is subsequently capable
    """

    # Define the abstract methods
    @abstractmethod
    def encoding_implementation(self, content: Any)->torch.Tensor:
        """
        This must be implemented in a subclass. It should
        accept the incoming content which will be directly
        from the payload feature, and return a tensor with
        a particular shape.

        :param content: The raw data off the payload feature
        :return: A tensor with a particular shape.
        """

    @abstractmethod
    def subclass_decode(self, content: torch.Tensor)->Any:

    def __init__(self,):
        pass

    # Define the public methods
    def encode(self,
               block: Any
               )->torch.Tensor:

        # Encode the content
        encodings = self.encoding_implementation(block)

        # Define the shape section of the data
        shape = encodings.shape



class Processor:
    """
    An object capable of encoding
    """

    def encode(self, intake: Any):


class MultimodalTokenizer()

tokenizer = T5Tokenizer.from_pretrained("t5-small")