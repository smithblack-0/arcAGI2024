import textwrap

import torch
from functools import cached_property
from torch.nn import functional as F
from typing import List, Dict, Optional, Tuple, TypeVar, Any
from abc import ABC, abstractmethod
NestedList = TypeVar('NestedList')



class TensorChannelSpec:
    """
    A data class that defines the channels
    and provides information about them
    """
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
    def slices(self)->Dict[str, slice]:
        position = 0
        output = {}
        for name, length in self.spec.items():
            output[name] = slice(position, position+length)
            position += length
        return output

    def __init__(self, channel_spec: Dict[str, int]):
        self.spec = channel_spec

class MTCTensorManager:
    """
    A helper class designed to be bound to
    and promise to edit values of a particular
    tensor channel spec.

    ---- MTC tensor ----

    TODO: Brief description

    ---- methods ----
    TODO
    """
    ###
    # Validation mechanisms.
    ##
    def validate_channels(self, channels: List[str]):
        """Validates that a given channel list is sane and addresses actual channels"""
        if len(set(channels)) != len(channels):
            raise ValueError('Channels must have unique values')
        for channel_name in channels:
            if channel_name not in self.channel_spec.channels:
                msg = f"Channel of name '{channel_name}' is not among '{self.channel_spec.channels}}'"
                raise ValueError(msg)

    def validate_channel_width(self, channels: List[str], channel_width: int):
        """Validates that a given channel collection is compatible with a given channel width"""
        required_length = sum(self.channel_spec.channel_widths[channel] for channel in channels)
        if required_length != channel_width:
            msg = f"""
            The selected channels were: {channels}
            These were expected to be associated with tensors of width: {required_length}
            However, we actually got a tensor of width {channel_width}
            """
            msg = textwrap.dedent(msg)
            raise ValueError(msg)


    ## Init ##

    def __init__(self,
                 channel_spec: TensorChannelSpec
                 ):
        self.channel_spec = channel_spec
    def create_mtc_tensor(self,
               channels: str | List[str],
               values: torch.Tensor,
               )->torch.Tensor:
        """
        Populates the given channels with the given values.
        Everything else is padded out to be zero. The channel information
        is expected to be provided in values along the last dimension in
        the specified channel order.

        :param channels:
            - The channels to insert into.
            - Order matters.
            - Do not repeat channels
            - Information must be associated in that order over in values.
        :param values:
            - The values to create the MTC tensor out of.
            - The channels that are active should be defined within this
        :return:
            - An MTC tensor.
        """

        # Input standardization
        if not isinstance(channels, list):
            channels = [channels]

        # Validation
        if __debug__:
            self.validate_channels(channels)
            self.validate_channel_width(channels, values.shape[-1])

        # Separate the values into sections associated with each individual channel
        split_indices = [0] + [self.channel_spec.channel_widths[channel] for channel in channels]
        split_indices = split_indices[:-1]
        split_indices = torch.cumsum(torch.tensor(split_indices), dim=0)
        sections = torch.split(values, split_indices, dim=-1)

        # Create the output tensor
        shape = list(values.shape)
        shape[-1] = self.channel_spec.total_width
        output = torch.zeros(shape, dtype=values.dtype, device=values.device)

        # Commit the values to the proper locations, in sequence
        for channel, section in zip(channels, sections):
            target_slice = self.channel_spec.slices[channel]
            output[..., target_slice] = section

        return output



    def set_mtc_tensor(self,
                   mtc_tensor: torch.Tensor,
                   channels: str | List[str],
                   values: torch.Tensor
                   ):
        """
        :param mtc_tensor:
         - The MTC tensor we are managing. We plan on setting to some of the channels
        :param channels:
        :param values:
        :return:
        """
        # Input standardization
        if not isinstance(channels, list):
            channels = [channels]

        # Validation
        if __debug__:
            self.validate_channels(channels)
            self.validate_channel_width(channels, values.shape[-1])
            self.validate_mtc_tensor(mtc_tensor,)



    def extract_mtc_tensor(self,
                           channels: List[str]
                           )->Tuple['MTCTensorManager', torch.Tensor]:
        """

        :param channels:
        :return:
        """

class TensorChannelManager:
    """
    A helper class that manages tensor channels and allows structured, programmatic
    access to different features stored within the last dimension of a tensor.

    The purpose of this class is to avoid hardcoding channel indices by providing
    a centralized mechanism for combining and extracting features from a tensor's
    last dimension based on user-specified channel mappings.

    ---- Initialization ----

    This class must be initialized with a dictionary (`channel_spec`) that defines
    how many channels each feature (or purpose) occupies in the last dimension
    of a tensor. The keys of this dictionary are feature names (strings), and the
    values are integers representing the number of channels allocated to each feature.

    Example:
        channel_spec = {
            "position": 2,     # 2 channels for position
            "velocity": 3,     # 3 channels for velocity
            "acceleration": 1  # 1 channel for acceleration
        }

    This means tensors combined or extracted using this class must adhere to the
    specified format, with `position` represented by the first two channels,
    `velocity` by the next three channels, and `acceleration` by the last channel.

    ---- Properties ----

    * `channel_allocs`: Returns a list of feature names (in the order defined in the
      `channel_spec`) that maps each feature to its allocated channels.

    * `indexes`: Returns a dictionary that maps feature names to the corresponding
      index positions in the channel dimension. This allows easy access to the
      index positions for each feature when working with tensors.

    * `channel_length`: Returns the total number of channels across all features,
      as specified by the `channel_spec`.

    ---- Usage ----

    * `combine`:
        Accepts a dictionary of tensors and combines them along the last dimension
        in the order specified in the `channel_spec`. Each tensor must correspond
        to a feature, and the tensors must have matching shapes in all dimensions
        except the last. The last dimension of each tensor must match the number
        of channels allocated to the corresponding feature in `channel_spec`.

        Example:
            Given the `channel_spec` from above, the `combine` method would accept
            a dictionary like this:

            tensors = {
                "position": torch.tensor(...),   # Shape: (..., 2)
                "velocity": torch.tensor(...),   # Shape: (..., 3)
                "acceleration": torch.tensor(...) # Shape: (..., 1)
            }

            The `combine` method will concatenate these tensors along the last
            dimension, producing a tensor of shape `(..., 6)`.

    * `extract`:
        Accepts a tensor and a string (the name of the feature to extract), and
        returns the portion of the tensor corresponding to that feature. The input
        tensor must have a last dimension that matches the total number of channels
        defined in the `channel_spec`. The method extracts the channels associated
        with the requested feature and returns them as a tensor.

        Example:
            Given the combined tensor of shape `(..., 6)`, calling
            `extract(tensor, "velocity")` would return the slice corresponding to
            the velocity feature, which would have shape `(..., 3)`.

    ---- Example Workflow ----

    1. Initialize a `TensorChannelSpec` with a channel specification:

       spec = TensorChannelSpec({
           "position": 2,
           "velocity": 3,
           "acceleration": 1
       })

    2. Combine feature tensors into a single tensor:

       combined_tensor = spec.combine({
           "position": position_tensor,       # Shape: (..., 2)
           "velocity": velocity_tensor,       # Shape: (..., 3)
           "acceleration": acceleration_tensor # Shape: (..., 1)
       })

    3. Extract specific features from the combined tensor:

       position_only = spec.extract(combined_tensor, "position")  # Shape: (..., 2)
       velocity_only = spec.extract(combined_tensor, "velocity")  # Shape: (..., 3)
    """
    @property
    def channel_allocs(self)->List[str]:
        return list(self.channel_spec.keys())

    @property
    def slices(self)->Dict[str, slice]:
        position = 0
        output = {}
        for name, length in self.channel_spec.items():
            output[name] = slice(position, position+length)
            position += length
        return output

    @property
    def channel_length(self) -> int:
        return sum(self.channel_spec.values())

    @property
    def channel_width(self)->Dict[str, int]:
        return self.channel_spec

    ##
    # Validation logic.
    #
    # All error messages MUST be thrown as originating
    # from something in here. Anything else is a bug
    ##
    def validate_channel_selections(self,
                                    channels: List[str],
                                    ):
        """
        Validation for whether the given channels exists.
        :param channels: A list of the channels to check
        :raises: ValueError if one of the provided channels does not exist
        """
        # Throws an exception if any channel in the provided list
        # was not actually a valid channel.
        for channel in channels:
            if channel not in self.channel_spec:
                msg = f"""
                Channel Validation Error
                
                Channel of name '{channel}' was interacted with, but is not defined.
                The only defined channels are {self.channel_allocs}.
                """
                msg = textwrap.dedent(msg)
                raise ValueError(msg)

    def validate_channel_width(self,):


    def __init__(self, channel_spec: Dict[str, int]):
        self.channel_spec = channel_spec
    def get_common_shape(self,
                         tensors: Dict[str, Optional[torch.Tensor]]
                         )->Tuple[Tuple[int, ...], torch.dtype, torch.device]:
        """Gets the shape of the tensors, and throws if they are not common throughout"""
        common_shape = None
        dtype = None
        device = None
        for channel_name, tensor in tensors.items():
            if tensor is None:
                continue
            if common_shape is None:
                common_shape = tensor.shape[:-1]
                dtype = tensor.dtype
                device = tensor.device
            assert tensor.shape[:-1] == common_shape, f"Shape mismatch: {tensor.shape[:-1]} != {common_shape}"
            assert tensor.dtype == dtype, f"Type mismatch: {tensor.dtype} != {dtype}"
            assert tensor.device == device, f"Device mismatch: {tensor.device} != {device}"
        assert common_shape is not None
        return common_shape, dtype, device
    def combine(self, tensors: Dict[str, Optional[torch.Tensor]])->torch.Tensor:
        """
        Combines together various tensors in the proper order to be
        processed by elements located downstream. It expects all
        channels to be defined, and all tensors to have common shape
        up to the last dimension, at which point the shape must be equal
        to the specified channel spec length.

        :param tensors:
            - A dictionary of tensors with each tensor matching a channel
            - All channels must be matched.
            - The tensors must be stackable along the last dimension
            - The shape of the last dimension must match the channel spec.
            - If a tensor has a value of none, it is filled with zeros
        :return: A combined tensor in the proper order.
        """
        # Perform sanity checking
        assert set(tensors.keys()) == set(self.channel_allocs), "Tensors and channels do not match"
        nonchannel_shape, dtype, device = self.get_common_shape(tensors)

        # Primary processing

        ordered_tensors = []
        for channel_name, length in self.channel_spec.items():
            # Get the tensor
            tensor = tensors[channel_name]

            if tensor is None:
                # Handle cases where we are suppose to fill in with padding
                tensor = torch.zeros([*nonchannel_shape, length],
                                     dtype=dtype,
                                     device=device)

            # Assert and append

            assert tensor.shape[:-1] == nonchannel_shape, f"Tensor of name {channel_name} not compatible"
            assert tensor.shape[-1] == length, f"Tensor of name {channel_name} had different channel length"
            ordered_tensors.append(tensor)

        return torch.cat(ordered_tensors, dim=-1)

    def extract(self,
                tensor: torch.Tensor,
                extractions: str | List[str]
                )->torch.Tensor:
        """
        "tensor" is assumed to be in channel spec format. This
        will extract a particular feature out of the channel spec
        :param tensor: The tensor in channel spec format
        :param extraction: A string indicating the element to extract, or a list to extract
        :return: The extracted tensor. It will still include the channel dim
        """
        if not isinstance(extractions, list):
            extractions = [extractions]

        # Validate
        assert all(key in self.channel_allocs for key in extractions), "Attempted to extract a channel that does not exist"
        assert tensor.shape[-1] == self.channel_length, "Tensor could not have been a channelspec."

        # Extract
        output = []
        for channel_name in extractions:
            tensor_slice = self.slices[channel_name]
            selected_tensor = tensor[..., tensor_slice]
            output.append(selected_tensor)
        return torch.cat(output, dim=-1)

    def replace(self,
                tensor: torch.Tensor,
                channel_names: str | List[str],
                replacement: torch.Tensor
                )->torch.Tensor:
        """
        Replace a channel feature in tensor with the given replacement. Does
        not overwrite but makes a new copy.

        :param tensor:
            - The tensor to start with
            - Shape (..., items, channels)
        :param channel_names:
            - The features to replace
            - Must be in channel allocs
            - Must be the case
        :param replacement:
            - The thing to replace with.
            - Must be shape (..., items, D), with D matching alloc with
        :return:
            - A tensor with the elements replaced
            - Shape (..., items, channels)
        """

        # Input standardization
        if not isinstance(channel_names, list):
            channel_names = [channel_names]

        # Validation
        if __debug__:
            self.validate_channel_selections(channel_names)
            self.validate_valid_channel_width(channel_names, replacement.shape[-1])

        assert channel_name in self.channel_allocs
        assert replacement.shape[-1] == self.channel_spec[channel_name]
        assert replacement.shape[:-1] == tensor.shape[:-1]

        tensor_slice = self.slices[channel_name]
        tensor = tensor.clone()
        tensor[..., tensor_slice] = replacement



class BoundChannels:
    """
     A utility class that binds to a particular channel within a Multimodal Token Channels (MTC) tensor.
     This class provides methods to interact with the bound channel, allowing setting, extracting, and
     creating MTC tensors for that channel.

     **Purpose**:
     To streamline access and modification of individual channels in the MTC tensor, where each
     channel represents a different feature in the data (e.g., mode, data, zone). The `BoundChannel`
     acts as an adapter for working with the specific channel and handles broadcasting and dimensional
     alignment when setting values.
     """

    def __init__(self,
                 channel: str,
                 channel_spec: TensorChannelManager
                 ):
        assert channel in channel_spec.channel_allocs
        self.channel = channel
        self.channel_spec = channel_spec

    def set(self, mtc_tensor: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """
        Set the specified channel in the MTC (Multimodal Token Channels) tensor to the provided values.

        This method assigns the provided values to the channel bound during the initialization of
        the `BoundChannel`. The `values` tensor must match the channel width, and it will be
        broadcasted to align with the non-channel dimensions of `mtc_tensor`.

        ---- Parameters ----
        :param mtc_tensor:
            - The MTC tensor to modify.
            - Shape: `(..., total_channel_width)` where `total_channel_width` is the combined width of all channels.

        :param values:
            - The values to assign to the channel.
            - Shape: `(..., channel_width)`, where `channel_width` matches the width of the bound channel.
            - The dimensionality of `values` should not exceed that of `mtc_tensor`.
            - It will be broadcasted to match `mtc_tensor` in all non-channel dimensions.

        ---- Returns ----
        :return:
            - A modified copy of `mtc_tensor` with the values set in the bound channel.
            - Shape: Same as `mtc_tensor` (`(..., total_channel_width)`).
        """
        # Ensure the last dimension of `values` matches the channel width in the MTC tensor
        assert values.shape[-1] == self.channel_spec.channel_spec[self.channel]

        # Ensure the dimensionality of `values` does not exceed that of `mtc_tensor`
        assert values.dim() <= mtc_tensor.dim()

        # Prepare the broadcast shape for `values` to match `mtc_tensor`
        broadcast_shape = list(mtc_tensor.shape)
        broadcast_shape[-1] = values.shape[-1]
        replacement = values.broadcast_to(broadcast_shape)

        # Use the channel spec to replace the bound channel in `mtc_tensor` with the broadcasted `values`
        return self.channel_spec.replace(mtc_tensor, self.channel, replacement)

    def create(self, value: torch.Tensor) -> torch.Tensor:
        """
        Creates a new MTC tensor with the provided values assigned to the bound channel.
        All other channels will be initialized to zeros.

        ---- Parameters ----
        :param value:
            - The value tensor to assign to the channel.
            - Shape: `(..., items, channel_width)`, where `channel_width` corresponds to the width of the bound channel.

        ---- Returns ----
        :return:
            - A new MTC tensor where the bound channel contains `value` and all other channels are filled with zeros.
            - Shape: `(..., items, total_channels)`, where `total_channels` corresponds to the combined width of all channels.
        """
        assert value.shape[-1] == self.channel_spec.channel_spec[self.channel]

        # Create a dictionary of None values for all channels, except for the bound channel.
        tensors = {name: None for name in self.channel_spec.channel_allocs}
        tensors[self.channel] = value

        # Combine the values into a new tensor where the bound channel is set and others are filled with zeros.
        return self.channel_spec.combine(tensors)

    def extract(self, mtc_tensor: torch.Tensor) -> torch.Tensor:
        """
        Extracts the values from the bound channel within the provided MTC tensor.

        ---- Parameters ----
        :param mtc_tensor:
            - The MTC tensor to extract from.
            - Shape: `(..., total_channel_width)`.

        ---- Returns ----
        :return:
            - A tensor containing the values from the bound channel.
            - Shape: `(..., channel_width)`, where `channel_width` corresponds to the width of the bound channel.
        """
        return self.channel_spec.extract(mtc_tensor, self.channel)


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




class ModeSpec:
    """
    The mode spec keeps track of the various modes that
    are currently active, and what integers are associated
    with what modes. It also abstracts away the process of
    checking for modesc
    """
    def __init__(self,
                 channel_name: str,
                 modes: str,
                 channel_spec: TensorChannelManager):
        assert channel_name in channel_spec.channel_allocs
        self.channel_name = channel_name
        self.modes = modes
        self.channel_spec = channel_spec
    def set_mode(self, tensor: torch.Tensor, mode: str)->torch.Tensor:
        """
        Sets a MTC tensor to exclusively be assigned to the given mode.
        :param tensor: The MTC tensor to a
            - The MTC tensor to assign
            - Shape (..., items, channels)
        :param mode:
            - The mode to assign
            - Must be one of the modes of the spec.
        :return:
            - A tensor of the same shape as the input
            - the mode channel will be set.
        """
        # Sanity check
        assert mode in self.modes

        # Setup and commit replacement
        index = self.modes.index(mode)
        replacement = torch.full(tensor.shape[:-1], index, device=tensor.device, dtype=tensor.dtype)
        return self.channel_spec.replace(tensor, self.channel_name, replacement)

    def filter_by_modes(self,
                        tensor: torch.Tensor,
                        nonpadding_mask: torch.Tensor,
                        modes: str | List[str]
                        )->Tuple[torch.Tensor, torch.Tensor]:
        """
        Filters the entries of the given MTC tensor to only
        keep around entries that belong to one of the given modes.
        Padding is also adjusted as needed.

        :param tensor:
            - The MTC tensor to filter. Filtration will occur along the 'items' dimension
            - Shape (..., items, channels)
        :param nonpadding_mask:
            - A mask which indicates whether an item is active, or only included as padding
            - shape (..., items)
        :param modes:
            - The modes to keep. May be just a string, or a list of strings.
            - All modes must actually have been defined
            - Items are returned in the order they occurred.
        :return:
            - tensor:
                - A MTC tensor where all items elements must have a mode in filtered mode
                - Shape (..., less_items, channels)
            - nonpadding_mask:
                - The new nonpadding mask
                - Shape (..., less_items)
        """
        if not isinstance(modes, list):
            modes = [modes]

        assert all(key in self.modes for key in modes)
        assert len(set(modes)) == len(modes)
        assert nonpadding_mask.shape == tensor.shape[:-1]

        # Construct a boolean mask that will associate each mode with one of the
        # active channels, or nothing

        mode_tensor = self.channel_spec.extract(tensor, self.channel_name) #(..., items, 1)
        modes_targets = torch.tensor([self.modes.index(mode) for mode in modes], device=tensor.device) #len(modes)
        modes_mask = (mode_tensor == modes_targets).any(dim=-1)

        # Figure out the maximum number of items to be included, and thus the
        # needed padding size. Then create destination tensors

        max_final_items = modes_mask.sum(dim=-1).max()
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
        destination_mask, _ = torch.sort(modes_mask, dim=-2, descending=True, stable=True)
        destination_mask = destination_mask[..., :max_final_items]

        destination_tensor[destination_mask.unsqueeze(-1)] = tensor[modes_mask.unsqueeze(-1)]
        destination_padding[destination_mask] = nonpadding_mask[modes_mask]

        # Return
        return destination_tensor, destination_padding
class ZoneSpec:
    """
    The `ZoneSpec` class is responsible for inserting and managing zone-specific metadata
    within a multimodal token channel (MTC) tensor. A "zone" refers to a group of tensor blocks
    related to a specific task or topic, such as a "Hypothesis" zone or a "Test" zone. The model's
    behavior and generation strategy vary significantly depending on the active zone, with zones
    influencing how content is generated and processed within the model.

    Zone information is stored as metadata in a dedicated channel of the MTC tensor, represented
    as integers that are mapped one-to-one to zone names. This metadata helps condition the model's
    generation strategy and can sometimes influence loss calculation or communication between zones.

    **Purpose**:
    The primary function of `ZoneSpec` is to insert zone information into the correct channel
    of an MTC tensor, based on the provided `TensorChannelSpec`. The class abstracts this insertion
    process, allowing for dynamic interaction with zone information during different stages of model
    training, evaluation, and inference.

    **Usage**:
    The zone list, which is static after construction, defines the mapping between zone names and
    their corresponding integer representation. This list is a hyperparameter of the model and remains
    consistent across stages of the model's lifecycle.

    **Initialization**:
    `ZoneSpec` requires both a `TensorChannelSpec` and a list of strings representing the zone names
    upon initialization. The zone names define the structure of the zone-to-integer mapping.

    :param channel_spec: The `TensorChannelSpec` that defines the channel allocations, including the zone channel.
    :param zones: A list of strings representing the zone names to be mapped to integers.

    Example:
    ```python
    channel_spec = TensorChannelSpec(...)
    zones = ["Hypothesis", "Test"]
    zone_spec = ZoneSpec(channel_spec, zones)
    ```

    Methods:
    --------
    - `insert_zone`: Inserts zone-specific metadata into the appropriate channel of the MTC tensor.
    - `get_zone_info`: Retrieves the zone information for a particular tensor block, if needed.
    """
    @property
    def num_zones(self)->int:
        return len(self.zones)

    def __init__(self,
                 zone_channel: str,
                 channel_spec: TensorChannelManager,
                 zones: List[str]
                 ):
        assert zone_channel in channel_spec.channel_allocs
        self.zone_channel = zone_channel
        self.channel_spec = channel_spec
        self.zones = zones
    def attach_zone_metadata(self, tensor: torch.Tensor, zone: str)->torch.Tensor:
        """
        Insert metadata assigning the particular tensor to be part of the indicated
        zone. Returns that new tensor. Used primarily while encoding or generating

        :param tensor:
            - A tensor in MTC format which we can insert information into
            - Shape (..., items, channels)
        :param zone:
            - A string indicating the zone to indicate attachment to
            - Must be in provided initial zones
        :return:
            - A tensor matching the original tensor shape
            - Zone feature will be filled
        """
        # Check we are being asked to sanely attach a zone
        assert zone in self.zones

        # Create zone replacement feature based on tensor shape
        index = self.zones.index(zone)
        shape = tensor.shape[:-1]
        replacement = torch.full(shape, index, dtype=tensor.dtype, device=tensor.device)
        replacement = replacement.unsqueeze(-1)

        # Attach it and return
        return self.channel_spec.replace(tensor, self.zone_channel, replacement)
    def filter_tensors_by_zones(self,
                                tensor: torch.Tensor,
                                zones: str | List[str]
                                )->Dict[str, torch.Tensor]:
        """
        Accepts a collection of
        :param tensor:
        :param zones:
        :return:
        """


    def define_zone_extraction_mask(self, tensor: torch.Tensor)->torch.Tensor:
    def extract_zone_tensors(self, tensor: torch.Tensor)->Dict[str,torch.Tensor]:
        """
        Extracts the zones out of an unbatched
        based on the zones that are marked, into a dictionary that has
        one entry for each type of zone.

        :return:
            - A dictionary with one entry for each type of zone
            - Contents are unbatched
            - Entries have shape (items, channel), with items being related to the zone
        """
        assert tensor.ndim == 2

        # Identify what belongs to what zone
        zone_matching_bool = torch.arange()
        zone_ids = self.extract_zone_ids(tensor)
        zone_outcomes = self.extract_zone_out
    def extract_zone_tensors(self, tensor: torch.Tensor)->NestedList[Dict[str, torch.Tensor]]:
        """
        Extracts the zone tensors information from the provided channels. If batch
        dimensions exist, each batch will be processed separate

        :param tensor:
            - The MTC tensor to extract information from
            - Shape (..., items, channels)
        :return:
            - The zones.
            - Shape (..., items, 1)
        """
        # WARNING: monitor performance carefully. This may need vectorization.
        # Conceptually, the alternative would be to use boolean masks. Small
        # batch size also helps. You also might consider dispatching then
        # waiting on futures.

        # We recursively extract zones until we are left only with the
        # items, channels features. Then we extract each zone into a dictionary

        if tensor.dim() > 2:
            output = []
            for subtensor in tensor.unbind(0):
                output.append(self.extract_zones(subtensor))
        return self.extract_zone(tensor)
