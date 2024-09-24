"""
Mechanisms associated with encoding or decoding a particular data block into
integer-based token form.


"""
import torch
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedTokenizer
from channel_bound_tensors import CBTensor, CBTensorSpec
from typing import Dict, List, Any, Callable, Tuple
from abc import ABC, abstractmethod

class AbstractBlockDataProcessor(ABC):
    """
    Encodes an entire block's data.
    Does not worry about handling headers
    """

    @abstractmethod
    def encode_helper(self, payload: Any)->torch.Tensor:
        """
        Abstract method. Encodes a payload into a tensor.

        :param payload: The payload that needs to be encoded
        :return: Tensor of integer encodings
            - The tensor should bave shape (..., channels), where ... indicates
              the dimensions and each element is an int.
            - One example would be (items, 1) for text
            - Another would be (x_axis, y_axis, patches) for an image
        """

    @abstractmethod
    def decode_helper(self, tensor: torch.Tensor)->Any:
        """
        One of the required helper functions. Decodes a tensor
        into a native representation, whatever that means. This
        could mean text, image, list of ints, whatever.

        :param tensor: The tensor to decode
            - Not batched
            - Has a fixed shape of (..., channels) that must be interpreted
            - channels depends on how many data channels we allow our content to use
            - Will be integer grid by this point.
        :return: The decoded tensor.
        """

    def encode(self,
               payload: Any,
               mode: int,
               submode: int
               )->Tuple[CBTensor, Tuple[int, ...]]:
        """
        Encodes a payload into a CBTensor, with set information for
        index, shape, mode, submode, and data. Any extra channels are
        extended before combination.

        :param payload: The payload to encode
        :param mode: The mode to mark it encoded as
        :param submode: The submode to mark it encoded as
        :return:
            - The CBTensor.
            - The nonchannel shape.
                - Used downstream, and we avoid fishing it out of the CBTensor this way
        """

        # Define the shape and data feature.
        tensor = self.encode_helper(payload)
        nonchannel_shape = torch.tensor(tensor.shape[:-1], device=tensor.device)
        data = tensor.flatten(0, -2)

        # Define the index feature.

        indexes = [torch.arange(dimension, device=tensor.device) for dimension in shape]
        index_mesh = torch.cartesian_product(*indexes)

        # Construct our encoding
        constructor = {
            "data" : data,
            "shape" : nonchannel_shape,
            "index" : index_mesh,
            "mode" : torch.tensor([mode], device=tensor.device),
            "submode" : torch.tensor([submode], device=tensor.device)
        }

        # Apply whatever padding might be needed to bring us into compliance
        # with the provided spec.

        for channel in constructor:
            feature = constructor[channel]
            spec_length = self.spec.channel_widths[channel]
            feature = F.pad(feature, (0, spec_length - feature.shape[-1]))
            constructor[channel] = feature

        # Construct and return
        tensor = CBTensor.create_from_channels(constructor)
        tensor = tensor.rebind_to_spec(self.spec)
        return tensor, nonchannel_shape
    def decode(self, tensor: CBTensor)->Any:
        """
        Decodes a CBTensor into original content, whatever
        that might end up being.

        :param tensor:
            - The CBTensor to decode.
            - Should ONLY contain block data by this point.
            - Should NOT be batched by this point.
            - Shape (items, total_channels)
        :return: The content.
        """

        # Extract the shape and data information.
        #
        # This will likely contain extra padding.

        tensor = tensor.gather_channels(["shape", "data"])
        data = tensor.gather_channels("data").get_tensor()
        shape_info = tensor.gather_channels("shape").get_tensor()
        shape = shape_info[0]

        # Remove that extra padding.

        is_shape_padding = shape == 0
        shape = shape.masked_select(is_shape_padding)
        data = data[..., :self.mode_data_width]

        # Finalize the shape. Insert the needed channels dimensions
        shape = list(shape) + [self.mode_data_width]

        # Resize, evaluate, and return
        data = data.reshape(shape)
        return self.decode_helper(data)

    def __init__(self,
                 spec: CBTensorSpec,
                 modality: str,
                 mode_data_width: int
                 ):
        self.spec = spec
        self.modality = modality
        self.mode_data_width = mode_data_width

class TextBlockDataProcessor(AbstractBlockDataProcessor):
    """
    Tokenizes and detokenizes text blocks. The payload
    in these blocks will always be strings. Can use
    various different tokenizers.
    """

    def encode_helper(self, payload: Any) ->torch.Tensor:
        assert isinstance(payload, str)
        tokens = self.tokenizer.encode(payload)
        tokens = torch.tensor(tokens, dtype=torch.long, device=self.device)
        return tokens

    def decode_helper(self, tensor: torch.Tensor) -> Any:
        text = self.tokenizer.decode(tensor)
        return text

    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 spec: CBTensorSpec,
                 device: torch.device = torch.device("cpu")
                 ):
        self.tokenizer = tokenizer
        self.device = device
        super().__init__(spec, "text", 1)

class IntGridBlockDataProcessor(AbstractBlockDataProcessor):
    """
    Converts 'intgrids' - grids of integer data which might
    be pieces on a chessboard, arc-agi data, etc -
    into a flattened tensor representation. Will always be
    an N x M grid
    """
    def __init__(self,
                 spec: CBTensorSpec,
                 device: torch.device = torch.device("cpu")
                 ):
        super().__init__(spec, "intgrid", 2)
        self.device = device
    def encode_helper(self, payload: Any) ->torch.Tensor:
        return torch.tensor(payload, device=self.device, dtype=torch.long)
    def decode_helper(self, tensor: torch.Tensor) -> Any:
        return tensor.tolist()

class BlockProcessor:
    """
    Constructs the headers for blocks.
    These are needed in order to ensure the main
    has a way
    """
    def __init__(self,
                 ):

