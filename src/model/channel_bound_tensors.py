import textwrap

`import torch
from functools import cached_property
from torch.nn import functional as F
from typing import List, Dict, Optional, Tuple, TypeVar, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
NestedList = TypeVar('NestedList')


@dataclass(frozen=True)
class CBTensorSpec:
    """
    The CB tensor spec is designed to hold channel binding (CB) data
    for a channel bound tensor. This means tracking both the channels, and the
    widths. We also provide a bunch of informative statistics.

    ---- properties ----

    channels: The channels that are being represented, and in what order.
    channel_width: For each channel name, how many elements wide that channel is
    total_width: The width of all the channels put together. The sum of the individual lengths
    start_index: For each channel, what the start index for the channel is.
    end_index: For each channel, what the end index for the channel is.
    slices: For each channel, what the slice addressing that channel would be.
    """

    spec: Dict[str, int]

    @property
    def channels(self)->List[str]:
        return list(self.spec.keys())

    @property
    def channel_widths(self)->Dict[str, int]:
        return self.spec.copy()

    @property
    def total_width(self)->int:
        return sum(self.spec.values())

    @cached_property
    def start_index(self)->Dict[str, int]:
        position = 0
        output = {}
        for name, length in self.spec.items():
            output[name] = position
            position += length
        return output

    @cached_property
    def end_index(self)->Dict[str, int]:
        position = 0
        output = {}
        for name, length in self.spec.items():
            position += length
            output[name] = position
        return output

    @cached_property
    def slices(self)->Dict[str, slice]:
        output = {}
        for name in self.channels:
            output[name] = slice(self.start_index[name], self.end_index[name])
        return output

    def __contains__(self, item: str)->bool:
        return item in self.spec

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
        for channel_name in destination_widths.keys():
            if source_widths[channel_name] != destination_widths[channel_name]:
                msg = f"""
                Channel widths are mismatched for channel of name{channel_name}
                
                The destination tensor has width of {destination_widths[channel_name]} 
                However, the source has width of {source_widths[channel_name]} 
                """
                msg = textwrap.dedent(msg)
                raise ValueError(msg)

    def validate_tensor_channel_width(self, channels: List[str], tensor_width: int):
        """
        Validates that a given collection of channels will match up with the given channel
        width

        :param channels: The channels being targeted
        :param channel_width: The width of the provided
        """
        target_width = sum(self.channel_widths[channel] for channel in channels)
        if target_width != tensor_width:
            msg = f"""
            The tensor is not compatible with the channels. 
            
            The channels selected require a width of {target_width}.
            However, actually got {target_width} instead.
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


        # Create the spec and get the tensor to broadcast match
        spec = {channel : tensor_channels[channel].shape[-1] for channel in tensor_channels.keys()}
        highest_key = max(tensor_channels.keys(), key = lambda channel : tensor_channels[channel].dim())
        highest_dimensional_tensor = tensor_channels[highest_key]
        dtype = highest_dimensional_tensor.dtype
        device = highest_dimensional_tensor.device
        broadcast_shape = list(highest_dimensional_tensor.shape[:-1])

        # Construct the tensor core. Broadcast as needed.
        output = []
        for channel in tensor_channels.keys():
            required_shape = broadcast_shape + [case.shape[-1]]
            case = tensor_channels[channel]
            case = case.broadcast_to(required_shape)
            output.append(case)
        tensor = torch.cat(output, dim=-1)
        return CBTensor(spec, tensor)
    def __init__(self,
                 spec: CBTensorSpec | Dict[str, int],
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

    def rebind_to_spec(self, spec: CBTensorSpec | Dict[str, int])->'CBTensor':
        """
        Rebinds the current `CBTensor` to a new specification, which can add or reorder
        channels.

        This method expands the current tensor to include channels specified in the new
        spec, or reorders existing channels as needed. Any new channels in the new spec
        that are not present in the original tensor will be initialized with zeros.

        ---- Parameters ----
        :param spec:
            - The new `CBTensorSpec` or dictionary that defines the new channel structure.
            - The new spec must include all channels from the original tensor, and may
              include additional channels.

        ---- Returns ----
        :return:
            - A new `CBTensor` with the same data as the original tensor, but expanded or
              reordered according to the new spec.

        ---- Raises ----
        - `ValueError`: Raised if the new spec is incompatible with the original tensor,
          such as missing channels from the original tensor.

        """

        # Standardization
        if isinstance(spec, dict):
            spec = CBTensorSpec(spec)

        # Validation
        if __debug__:
            self.validate_channels_exist(spec.channels, self.channels)
            self.validate_common_channel_widths(spec.channel_widths, self.channel_widths)

        # Create new CBTensor
        shape = list(self.tensor.shape[:-1]) + [spec.total_width]
        tensor = torch.zeros(shape, dtype =self.tensor.dtype, device=self.tensor.device)
        tensor = CBTensor(spec, tensor)

        # Insert and return
        return tensor.set_channels(self)


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

    def gather_channels(self, channels: str | List[str])->'CBTensor':
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
            self.validate_channels_exist(self.channels, channels)

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



class BatchMagic:
    """
    A special class responsible for knowing how to batch, unbatch,
    and generally work with batches of MTC tensors.
    """

    def filter(self,
               tensor: torch.Tensor,
               nonpadding_mask: torch.Tensor,
               channel: str,
               filter: torch.Tensor
               )->Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs filtration pf the indicated MTC tensor by looking at the indicated channel,
        and only retaining the item if it has an integer in it that matches something in filter.

        :param tensor:
            - The MTC tensor to filter. Shape (..., items, channels)
        :param nonpadding_mask:
            - A nonpadding tensor.
            - Filtration is usually most useful on batched entities
            - Shape (..., items)
        :param channel:
            - The channel to select then manipulate.
            - Must be within our selected channels
        :param filter:
            - The integer values to filter with
            - Only integers that match are included
            - Shape: (filters, L)
            - L matches channel length
        :return:

        """
        # Perform sanity checks and assertions
        assert tensor.device == filter.device
        assert tensor.device == nonpadding_mask.device
        assert channel in self.channel_allocs
        assert tensor.shape[:-1] == nonpadding_mask.shape
        assert filter.dim() == 2
        assert filter.shape[-1] == self.channel_spec[channel]

        # Construct a boolean mask that will associate each element in the channel with one of the
        # active channels, or nothing

        channel_tensors = self.extract(tensor, channel) # (..., items, L)
        channel_tensors = channel_tensors.unsqueeze(dim=-2) #(..., items, 1, L)
        channels_mask = (channel_tensors == filter) #(..., items, filter, L)
        channels_mask = torch.all(channels_mask, dim=-1) #(..., items, filter)
        channels_mask = torch.any(channels_mask, dim=-1) #(..., items)

        # Figure out the maximum number of items to be included, and thus the
        # needed padding size. Then create destination tensors

        max_final_items = int(channels_mask.sum(dim=-1).max())
        shape = list(tensor.shape)
        shape[-2] = max_final_items

        destination_tensor = torch.full(shape, 0, device=tensor.device, dtype=tensor.dtype)
        destination_padding = torch.full(shape[:-1], False, device=tensor.device, dtype=nonpadding_mask.dtype)

        # Adjust the modes mask to get a destination modes mask that will land the
        # pieces in place. We do this by sorting to keep the true mask entries.
        # This ensures that all active elements are in place starting at zero and
        # getting higher.
        #
        # Then move everything into place using the masks.
        destination_mask, _ = torch.sort(channels_mask, dim=-2, descending=True, stable=True)
        destination_mask = destination_mask[..., :max_final_items]

        destination_tensor[destination_mask.unsqueeze(-1)] = tensor[channels_mask.unsqueeze(-1)]
        destination_padding[destination_mask] = nonpadding_mask[channels_mask]

        # Return
        return destination_tensor, destination_padding

    def batch(self,
              tensors: torch.Tensor | NestedList[torch.Tensor]
              )->Tuple[torch.Tensor, torch.Tensor]:
        """
        Batches a collection of tensors, potentially organized in nested lists, by adding
        necessary padding (with zeros) to align the dimensions. This method recursively processes
        nested lists of tensors to ensure that all elements are padded to a uniform shape.

        Returns a batched tensor and a corresponding non-padding mask that indicates which
        elements in the resulting batched tensor are valid (i.e., not padding).

        ---- Parameters ----
        :param tensors:
            A list or nested list of tensors that are to be batched.
            Each tensor in the list must have the same number of dimensions, even if the size
            of the dimensions (lengths) may vary. Tensors should all have the same last dimension
            (the channels dimension).

            Example:
                If the tensors are of shape (items, channels), you may have:
                    - A nested list like: [[tensor_a, tensor_b], [tensor_c]]
                    - With tensor_a.shape == (10, 5), tensor_b.shape == (12, 5), tensor_c.shape == (8, 5)

        ---- Returns ----
        :return:
            A tuple containing:
            - batched_tensor: A tensor with shape `(..., max_items, channels)`,
              where batch_size reflects the structure of the nested list and max_items is
              the length after padding.

            - nonpadding_mask: A mask with shape `(..., max_items)`, where `True`
              indicates a valid (non-padded) entry and `False` indicates padding. This mask
              can be used to track and later remove padded values.

        ---- Example Workflow ----
        If provided a nested list of tensors:
            tensors = [[tensor_a, tensor_b], [tensor_c]]

        Where:
            - tensor_a.shape == (10, 5)
            - tensor_b.shape == (12, 5)
            - tensor_c.shape == (8, 5)

        The result would be:
            - batched_tensor.shape == (2, 2, 12, 5)  # Batch of 2, max items 12, and 5 channels
            - nonpadding_mask.shape == (2, 2, 12)    # Same structure, marking valid vs. padded elements.
        """

        # Create the main processing arrays.
        #
        # We need to gather the tensors together that will end up in a batch,
        # alongside gathering the width to pad to on each of the dimensions.


        tensors_requiring_padding: List[Tuple[torch.Tensor, torch.Tensor]] = []
        max_dimension_length = None
        for item in tensors:
            if isinstance(item, torch.Tensor):
                # Base case encountered. We store the tensor and make a padding
                # mask that is filled entirely with true, indicating all elements
                # active.
                assert item.shape[-1] == self.channel_length
                dimension_lengths = torch.tensor(item.shape[:-1], device=item.device)

                # This logic gathers information on what to pad to for each dimensions
                if max_dimension_length is None:
                    max_dimension_length = dimension_lengths
                else:
                    # We detect when you are trying to combine tensors of differing number of dimensions.
                    #
                    # Since it then becomes ambiguous on how to pad and combine, we throw. Otherwise, we keep
                    # the largest of the dimensions to pad to.
                    assert max_dimension_length.shape[-1] == item.dim()
                    max_dimension_length = torch.maximum(max_dimension_length, dimension_lengths)

                entry = (item, torch.ones_like(item, dtype=torch.bool))
                tensors_requiring_padding.append(entry)
            else:
                # Item is a list. It will need to be recursively processed
                tensors_requiring_padding.append(self.batch(item))
        assert max_dimension_length is not None

        tensor_stack = []
        paddings_stack = []
        for tensor, padding in tensors_requiring_padding:
            assert tensor.shape[:-1] == padding.shape

            # Construct the padding operator. This will be based on
            # padding the dimensions other than the channel dimension
            # to the maximum length

            padding_amount = [target - actual for target, actual in zip(max_dimension_length, tensor.shape)]
            padding_operator = []
            for pad_length in reversed(padding_amount):
                padding_operator.extend([0, pad_length])

            # Pad the padding tensor to match
            padding = F.pad(padding, padding_operator, value=False)

            # Pad the tensor to match. Do not pad the channels dimension
            tensor = F.pad(tensor, [0, 0] + padding_operator)

            # Append
            tensor_stack.append(tensor)
            paddings_stack.append(padding)

        batched_tensor = torch.stack(tensor_stack, dim=0)
        batched_padding = torch.stack(paddings_stack, dim=0)

        return batched_tensor, batched_padding

    def unbatch(self,
                tensors: torch.Tensor,
                nonpadding_mask: torch.Tensor
                ) -> torch.Tensor | NestedList[torch.Tensor]:
        """
        Recursively unbatches a tensor into individual tensors or nested lists, removing padding along the way.

        The recursion stops when the input tensor reaches 2D (items, channels), and at that
        point, the padding is removed using the non-padding mask, and the valid entries
        are returned. If the input is higher-dimensional, it will return a nested list.

        :param tensors:
            - A batched tensor of shape (..., items, channels).
        :param nonpadding_mask:
            - A mask of shape (..., items) indicating valid entries.
        :return:
            - A tensor with padding removed or a nested list, preserving the structure of the input.
        """
        # Recursive case: unbind until we reach the 2D base case
        if tensors.ndim > 2:
            return [self.unbatch(subtensor, submask) for subtensor, submask in
                    zip(tensors.unbind(0), nonpadding_mask.unbind(0))]

        # Base case: tensors of shape (items, channels)
        # Unsqueeze the nonpadding_mask to match the dimensions of the tensor for indexing
        valid_tensor = tensors[nonpadding_mask.bool().unsqueeze(-1)]

        # Return the unpadded tensor
        return valid_tensor
