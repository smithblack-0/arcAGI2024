import textwrap
import torch
from typing import List, Dict, Optional, Tuple, TypeVar, Any, Union, Callable

from src.old.CBTensors.channel_bound_spec import CBTensorSpec

NestedList = TypeVar('NestedList')


class CBTensor:
    """
    The `CBTensor` class is designed to manage tensors that associate specific channels
    with designated widths, enabling efficient manipulation of channel-specific data
    within a tensor. The final dimension of each tensor is reserved for storing
    information related to these channels.

    This class provides an interface for managing channel assignments, modifying channel
    data, gathering channel subsets, and ensuring proper broadcasting when combining
    tensors with different shapes along non-channel dimensions.

    ---- Purpose ----

    The primary purpose of `CBTensor` is to provide structured access to a tensor's
    channels and the ability to perform operations such as setting, gathering,
    rebinding, and broadcasting across multiple dimensions while maintaining the
    integrity of the channel layout.

    ---- Properties ----

    * `channels`: A list of channel names representing the available channels in the
      tensor.

    * `channel_widths`: A dictionary mapping channel names to their respective widths
      (i.e., the number of elements allocated to each channel in the final tensor
      dimension).

    * `total_channel_width`: The total width of the final dimension in the tensor,
      which is the sum of all channel widths.

    * `slices`: A dictionary providing the slice object for accessing each channel's
      data in the final dimension of the tensor.

    * `channel_start_index`: The starting index (position) of each channel in the
      tensor's final dimension.

    * `channel_end_index`: The ending index (exclusive) of each channel in the tensor's
      final dimension.

    * `shape`: A property that returns the shape of the tensor excluding the channel dimension.
      The tensor behaves as though it does not have a final "channel" dimension when accessing
      its shape, which is useful for many operations in neural networks.

    ---- Shape Handling ----

    The `CBTensor` object behaves as if the final dimension, which represents the channels,
    is ignored when querying or slicing the tensor. When you access the `shape` of the
    tensor, you only get the dimensions excluding the channel dimension. For example, if the
    underlying tensor shape is `(batch_size, sequence_length, total_channel_width)`,
    accessing the `.shape` property will return `(batch_size, sequence_length)`.

    ---- Indexing Behavior ----

    The `__getitem__` method is designed to allow you to index the tensor as though the channel
    dimension does not exist. When you slice a `CBTensor`, the final channel dimension is
    automatically handled and preserved, so you do not need to worry about it when performing
    slicing operations on the non-channel dimensions. The tensor automatically adds a slice to
    include all channel elements, behaving as if the tensor has no channel dimension for indexing
    purposes.

    For example, indexing `tensor[0, 0]` will return a `CBTensor` containing all the channels
    for the first item of the batch, with the channels maintained in the final dimension.

    ---- Usage Scenarios ----

    * Managing channel-specific data for tensors used in multimodal neural networks.
    * Efficiently setting or gathering channel data without altering the rest of the
      tensor.
    * Broadcasting smaller tensors across larger batch sizes in a way that is
      compatible with both channel widths and non-channel dimensions.
    * Rebinding an existing `CBTensor` to a new specification (i.e., adding new
      channels or changing the order of channels).

    ---- Key Methods ----

    * `set_channels`: Allows you to set the values of specific channels in a tensor,
      broadcasting data from one tensor to another as needed.

    * `gather_channels`: Allows extraction of specific channels from a tensor to
      create a new `CBTensor` with just those channels.

    * `create_from_channels`: A class method that creates a `CBTensor` from a
      dictionary of individual tensors representing different channels.

    * `rebind_to_spec`: Rebinds an existing `CBTensor` to a new channel specification,
      allowing you to expand the tensor to include new channels or reorder existing
      ones.

    * `shape`: Returns the shape of the tensor excluding the final channel dimension.
      This is useful when working with the tensor in multimodal networks where the channel
      dimension is handled separately from the other dimensions.
    """
    ##
    # Define channel specific properties
    ##

    @property
    def channels(self) -> List[str]:
        return self.spec.channels

    @property
    def channel_widths(self) -> Dict[str, int]:
        return self.spec.channel_widths

    @property
    def total_channel_width(self) -> int:
        return self.spec.total_width

    @property
    def channel_start_index(self) -> Dict[str, int]:
        return self.spec.start_index

    @property
    def channel_end_index(self) -> Dict[str, int]:
        return self.spec.end_index

    @property
    def slices(self) -> Dict[str, slice]:
        return self.spec.slices

    @property
    def shape(self) -> Tuple[int]:
        """
        Returns the shape of the tensor, excluding the channel dimension.

        :return: Shape tuple excluding the final channel dimension.
        """
        return self.tensor.shape[:-1]


    @property
    def dtype(self)->torch.dtype:
        return self.tensor.dtype

    @property
    def device(self)->torch.device:
        return self.tensor.device


    def dim(self)->int:
        return self.tensor.dim()-1

    ##
    # Define magic methods
    ##
    IndexItems = Tuple[int, slice, type(Ellipsis)]
    IndexTuple = Tuple[IndexItems,...]
    def __getitem__(self,
                    key: Union[IndexItems, IndexTuple]):

        # Handle the ellipses trick, adding a slice
        # at the end. This ensures ellipses are handled
        # correctly. It also will result in errors being thrown
        # if you try to overindex.

        if not isinstance(key, tuple):
            key = (key,)
        key += (slice(None),)

        # Perform the indexing
        return CBTensor(self.spec, self.tensor[key])

    def __setitem__(self,
                    key: Union[IndexItems, IndexTuple],
                    value: 'CBTensor'):
        assert isinstance(value, CBTensor)
        assert self.spec == value.spec, "Cannot set tensors with different specs"

        # Handle the ellipses trick, adding a slice
        # at the end. This ensures ellipses are handled
        # correctly.
        if not isinstance(key, tuple):
            key = (key,)
        key = key + (slice(None),)

        # Perform the assignment, ensuring we operate over the proper slice
        self.tensor[key] = value.tensor

    ##
    # Define validation and helper functions
    ##

    def validate_cb_tensor(self,
                           tensor: Any
                           ):
        if not isinstance(tensor, CBTensor):
            raise ValueError(f'Expected a CBTensor, got {type(tensor)}')

    @staticmethod
    def validate_channels_exist(source_channels: List[str],
                                destination_channels: List[str],
                                ):
        """
        Validates that the channels defined in source tensor exist at destination
        tensor.

        :param source_tensor: The source tensor to get channel information from
        :param destination_tensor: The destination tensor whose channels we need to match
        :raises ValueError: If this fails to be true
        """

        for channel_name in source_channels:
            if channel_name not in destination_channels:
                msg = f"""
                Interaction failed. Source tensor was not channel compatible
                with destination tensor. 
                
                Source tensor had channel named '{channel_name}'
                However, destination tensor only supported '{destination_channels}'
                """
                msg = textwrap.dedent(msg)
                raise ValueError(msg)

    @staticmethod

    def validate_common_channel_widths(source_widths: Dict[str, int],
                                       destination_widths: Dict[str, int]
                                       ):
        """
        Validates that the channels in the source tensor have the same length as
        those in the destination tensor.s

        :param source_widths: The per-channel widhts of the source
        :param destination_tensor: The per-channel widths of the destination.
        :raises ValueError: If the shapes differ
        """
        for channel_name in source_widths.keys():
            if source_widths[channel_name] != destination_widths[channel_name]:
                msg = f"""
                Channel widths are mismatched for channel of name '{channel_name}'
                
                The destination tensor has width of {destination_widths[channel_name]} 
                However, the source has width of {source_widths[channel_name]} 
                """
                msg = textwrap.dedent(msg)
                raise ValueError(msg)

    @staticmethod
    def validate_broadcastable(destination_shape: Tuple[int, ...],
                               source_shape: Tuple[int, ...],
                               ):
        """
        Validates that the provided nonchannel dimensions are broadcastable with
        one another, and raises an error if not.

        :param destination_shape: The shape of the destination
        :param source_shape: The shape of the source
        :raises ValueError: If the two cannot be broadcast.
        """

        try:
            broadcast_shape = torch.broadcast_shapes(source_shape, destination_shape)
        except RuntimeError as e:
            msg = f"""
            Could not broadcast together source and destination tensors on the nonchannel
            dimensions. 
            """
            msg = textwrap.dedent(msg)
            raise ValueError(msg) from e

    # Construction actions

    @classmethod

    def create_from_channels(cls,
                             tensor_channels: Dict[str, torch.Tensor]
                             ) -> 'CBTensor':
        """
        Creates a new `CBTensor` from a dictionary of tensors representing individual
        channels. The new `CBTensor` will contain all specified channels and will broadcast
        the channel tensors as needed to ensure compatibility across non-channel dimensions.

        ---- Parameters ----
        :param tensor_channels:
            - A dictionary where each key is a channel name and each value is a tensor
              corresponding to that channel's data.
            - Each tensor must have a final dimension that corresponds to the channel's
              width, but the other dimensions (batch size, etc.) can differ and will be
              broadcast if needed.

        ---- Returns ----
        :return:
            - A `CBTensor` where the final dimension contains the combined width of all
              provided channels.
            - Any missing channels will be filled with zeros.

        ---- Raises ----
        - `ValueError`: Raised if the channel tensors cannot be broadcast to a common
          shape across non-channel dimensions.
        """


        # Create the spec.
        spec = {channel : tensor_channels[channel].shape[-1] for channel in tensor_channels.keys()}

        # Get the broadcast shape for the nonchannel portion, by combining the shapes of
        # the pieces.
        broadcast_shape = [1]
        for tensor in tensor_channels.values():
            nonchannel_shape = tensor.shape[:-1]

            if __debug__:
                cls.validate_broadcastable(broadcast_shape, nonchannel_shape)

            broadcast_shape = torch.broadcast_shapes(tensor.shape[:-1], broadcast_shape)
        broadcast_shape = list(broadcast_shape)

        # Construct the tensor core. Broadcast as needed.
        output = []
        for channel in tensor_channels.keys():
            case = tensor_channels[channel]
            required_shape = broadcast_shape + [case.shape[-1]]
            case = case.broadcast_to(required_shape)
            output.append(case)
        tensor = torch.cat(output, dim=-1)
        return CBTensor(spec, tensor)
    def __init__(self,
                 spec: Union[CBTensorSpec, Dict[str, int]],
                 tensor: Optional[torch.Tensor],
                 ):
        # Data standardization
        if isinstance(spec, dict):
            spec = CBTensorSpec(spec)
        if tensor is None:
            tensor = torch.zeros([spec.total_width])


        assert tensor.shape[-1] == spec.total_width, "tensor and spec mismatch"

        self.spec = spec
        self.tensor = tensor

    ##
    # Cross CB tensor logic. We are given and get back CB tensors
    ##

    def rebind_to_spec(self,
                       spec: Union[CBTensorSpec, Dict[str, int]],
                       allow_channel_expansion: bool = False,
                       allow_channel_pruning: bool = False)->'CBTensor':
        """
        Rebinds the current `CBTensor` to a new specification, which can add or reorder
        channels based on the provided flags.

        This method expands the current tensor to include channels specified in the new
        spec, or reorders existing channels as needed. Any new channels in the new spec
        that are not present in the original tensor will be initialized with zeros.

        ---- Parameters ----
        :param spec:
            - The new `CBTensorSpec` or dictionary that defines the new channel structure.
            - The new spec must include all channels from the original tensor unless
              `allow_channel_pruning` is set to True. Additional channels can be added if
              `allow_channel_expansion` is set to True.

        :param allow_channel_expansion:
            - If True, new channels can be added to the new spec, and they will be initialized with zeros.
            - If False, a ValueError is raised if the new spec includes extra channels.

        :param allow_channel_pruning:
            - If True, channels from the original tensor that are not in the new spec will be dropped.
            - If False, a ValueError is raised if the new spec is missing channels from the original tensor.

        ---- Returns ----
        :return:
            - A new `CBTensor` with the same data as the original tensor, expanded or reordered according
              to the new spec.

        ---- Raises ----
        - `ValueError`: Raised if the new spec is incompatible with the original tensor, such as missing
          channels or adding extra channels in violation of the expansion/pruning rules.
        """

        # Standardization
        if isinstance(spec, dict):
            spec = CBTensorSpec(spec)

        # Validation based on flags
        if __debug__:
            if allow_channel_expansion:
                self.validate_channels_exist(self.channels, spec.channels)
                self.validate_common_channel_widths(self.channel_widths, spec.channel_widths)
            elif allow_channel_pruning:
                self.validate_channels_exist(spec.channels, self.channels)
                self.validate_common_channel_widths(spec.channel_widths, self.channel_widths)
            else:
                self.validate_channels_exist(self.channels, spec.channels)
                self.validate_channels_exist(spec.channels, self.channels)
                self.validate_common_channel_widths(spec.channel_widths, self.channel_widths)

        # Create new CBTensor
        shape = list(self.tensor.shape[:-1]) + [spec.total_width]
        tensor = torch.zeros(shape, dtype =self.tensor.dtype, device=self.tensor.device)
        tensor = CBTensor(spec, tensor)

        # Gather relevant channels (if pruning, gather only the channels in the new spec)
        if allow_channel_pruning:
            relevant = self.gather_channels(spec.channels)
        else:
            relevant = self

        # Insert and return
        return tensor.set_channels(relevant)


    def set_channels(self, tensor: 'CBTensor')->'CBTensor':
        """
        Sets the values of channels from the provided `CBTensor` into the current tensor,
        broadcasting as needed to ensure compatibility along non-channel dimensions. In
        the case a channel is present in this tensor that is not in the provided source,
        that channel remains unchanged.

        This method modifies the channels in the destination tensor by broadcasting the
        source tensor (if necessary) along non-channel dimensions. It ensures that channel
        widths match between the source and destination tensors and raises an error if they
        do not.

        ---- Parameters ----
        :param tensor:
            - A `CBTensor` object containing the source tensor to transfer channel data
              from.
            - The source tensor must be broadcast-compatible with the destination tensor
              along all dimensions except the last (channel dimension), which must match
              in width.

        ---- Returns ----
        :return:
            - A new `CBTensor` with the specified channels set to the values from the
              source tensor.
            - The non-channel dimensions of the source tensor are broadcast to match the
              destination tensor if needed.

        ---- Raises ----
        - `ValueError`:
            - Raised if the source and destination tensors do not have compatible channel
              widths.
            - Raised if the source tensor cannot be broadcast to the destination tensor's
              non-channel dimensions.

        """

        # Validation
        if __debug__:
            self.validate_cb_tensor(tensor)
            self.validate_channels_exist(tensor.channels, self.channels)
            self.validate_common_channel_widths(tensor.channel_widths, self.channel_widths)
            self.validate_broadcastable(tensor.tensor.shape[:-1], self.tensor.shape[:-1])

        ##
        # Note:
        #
        # It is more memory and speed efficient to construct and modify a list than to
        # set to a tensor, presumably because torch stores the entire tensor
        # for gradient calculation and this just updates a pointer.
        ##

        destination_sections = self.separate_into_channels()
        source_sections = tensor.separate_into_channels()
        outcome = []
        for channel in destination_sections:
            case = source_sections[channel] if channel in source_sections else destination_sections[channel]
            case = torch.broadcast_to(case, destination_sections[channel].shape)
            outcome.append(case)
        outcome = torch.cat(outcome, dim=-1)
        return CBTensor(self.spec, outcome)

    def gather_channels(self, channels: Union[str,List[str]])->'CBTensor':
        """
        Gathers a subset of channels from the current tensor and returns a new `CBTensor`
        containing only the selected channels.

        This method allows extraction of a specific subset of channels from the current
        tensor. The resulting `CBTensor` will contain only the selected channels, in the
        specified order, while retaining the same non-channel dimensions as the original
        tensor.

        ---- Parameters ----
        :param channels:
            - A single channel name (string) or a list of channel names to gather.
            - The channels must exist in the current `CBTensor`, and they will be gathered
              in the order specified.

        ---- Returns ----
        :return:
            - A new `CBTensor` containing only the gathered channels.
            - The shape of the resulting tensor will have the same non-channel dimensions
              as the original, with a final dimension equal to the combined width of the
              gathered channels.

        ---- Raises ----
        - `ValueError`: Raised if any of the specified channels do not exist in the
          current `CBTensor`.

        """

        # Standardize
        if isinstance(channels, str):
            channels = [channels]

        # Validate
        if __debug__:
            self.validate_channels_exist(channels, self.channels)

        # Spec the outcome
        # Get the tensors
        spec = {channel : self.channel_widths[channel] for channel in channels}
        spec = CBTensorSpec(spec)
        tensors = [self.tensor[..., self.slices[channel]] for channel in channels]
        tensor = torch.cat(tensors, dim=-1)

        # Return the outcome
        return CBTensor(spec, tensor)

    ##
    # Direct, tensor based gathering and setting
    ##

    def set_tensor(self, tensor: torch.Tensor)->'CBTensor':
        """
        Sets the entire tensor for the current CBTensor
        representation. We must match the current total channel width

        :param tensor:
            - The tensor to set
            - Shape (..., total_channel_width)
        :return: A CBTensor, with the same spec but different tensor bindings
        """
        return CBTensor(self.spec, tensor)
    def get_tensor(self)->torch.Tensor:
        """
        Gets the currently managed tensor
        :return: The tensor
        """
        return self.tensor

    def separate_into_channels(self)->Dict[str, torch.Tensor]:
        """
        Separates a given tensors into a dictionary composing
        its various channels.

        :return: A dictionary of tensors, one per channel
        """
        return {name : self.tensor[..., self.slices[name]] for name in self.channels}

    def clone(self)->'CBTensor':
        return CBTensor(self.spec, self.tensor.clone())

    ###
    # Implement some necessary torch functions. We will likely implement more as we work
    ##
    supported_operators: Dict[Callable, Callable] = {}
    @classmethod
    def register_operator(cls, torch_functions: Callable | List[Callable])->Callable:
        """
        A decorator to register an operator as associated with a torch function. For example,
        you could registers torch.cat as associated with a custom CBTensor cat operation.

        :param torch_functions: The torch function or functions to register as associated.
        :return: A decorater designed to be applied to the function
        """

        # Standardize
        if not isinstance(torch_functions, list):
            torch_functions = [torch_functions]

        # Provide decorator
        def decorator(implementation: Callable)->Callable:
            for torch_function in torch_functions:
                cls.supported_operators[torch_function] = implementation
            return implementation
        return decorator

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None)->'CBTensor':
        """
        Implements a torch function handling passthrough. The primary thing we have
        to do to make torch functions compatible with the CBTensor is ensure we successfully
        hide away the channels dimension. This means intercepting any attempts to use a
        dim feature, and modifying the index if needed.

        One additional complication is that

        :param func: The funciton being invoked. Important.
        :param types: Unused
        :param args: This may have dim info in it that needs to be modified
        :param kwargs: This may have dim info in it that needs to be modified.
        :return:
        """
        if kwargs is None:
            kwargs = {}

        if func not in cls.supported_operators:
            raise ValueError(f"Torch function was not supported: {func}")
        return cls.supported_operators[func](*args, **kwargs)













