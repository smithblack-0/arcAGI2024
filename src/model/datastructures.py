import torch
from torch.nn import functional as F
from typing import List, Dict, Optional, Tuple, TypeVar
NestedList = TypeVar('NestedList')


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
                channel_name: str,
                replacement: torch.Tensor
                )->torch.Tensor:
        """
        Replace a channel feature in tensor with the given replacement. Does
        not overwrite but makes a new copy.

        :param tensor:
            - The tensor to start with
            - Shape (..., items, channels)
        :param channel_name:
            - The feature to replace
            - Must be in channel allocs
        :param replacement:
            - The thing to replace with.
            - Must be shape (..., items, D), with D matching alloc with
        :return:
            - A tensor with the elements replaced
            - Shape (..., items, channels)
        """
        assert channel_name in self.channel_allocs
        assert replacement.shape[-1] == self.channel_spec[channel_name]
        assert replacement.shape[:-1] == tensor.shape[:-1]

        tensor_slice = self.slices[channel_name]
        tensor = tensor.clone()
        tensor[..., tensor_slice] = replacement

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
              tensors: NestedList[torch.Tensor]
              )->Tuple[torch.Tensor, torch.Tensor]:
        """
        Batches a collection of tensors, possibly recursively, by adding
        padding (0) in place where needed. Returns both a data tensor, and
        a nonpadding mask indicating what elements were not padding, and where.

        :param tensors:
            - A list or nested list of tensors to batch.
            - The tensors in the list must all have shape (items, channels)
            - The nested list structure does not have to have lists of the same length.
              Padding is automatically injected where needed, including for varying lengths of
              lists. However, the number of DIMENSIONS must always be the same, even if the length of
              the lists can vary.
        :return:
        """


        # Create the main processing arrays.


        tensors_requiring_padding: List[Tuple[torch.Tensor, torch.Tensor]] = []
        shape_checker = None
        for item in tensors:
            if isinstance(item, torch.Tensor):
                # Base case encountered. We store the tensor and make a padding
                # mask that is filled entirely with true, indicating all elements
                # active.
                assert item.shape[-1] == self.channel_length

                if shape_checker is None:
                    shape_checker = item.shape[:-1]
                else:
                    assert item.shape[:-1] == shape_checker

                entry = (item, torch.ones_like(item, dtype=torch.bool))
                tensors_requiring_padding.append(entry)
            else:
                # Item is a list. It will need to be recursively processed
                tensors_requiring_padding.append(self.batch(item))


        # Get the maximum length. Then pad all to match.
        maximum_length = max([item.shape[-2] for item, _ in tensors_requiring_padding])
        tensor_stack = []
        paddings_stack = []
        for tensor, padding in tensors_requiring_padding:
            # Pad tensor to match maximum length
            pad_op = (0, 0, 0, maximum_length - tensor.shape[-2])
            tensor = F.pad(tensor, pad_op)

            # Pad padding to match maximum length
            pad_op = (0, maximum_length-padding.shape[-2])
            padding = F.pad(padding, pad_op)

            # Append
            tensor_stack.append(tensor)
            paddings_stack.append(padding)

        batched_tensor = torch.stack(tensor_stack, dim=0)
        batched_padding = torch.stack(paddings_stack, dim=0)

        return batched_tensor, batched_padding

    from typing import List, Union, Tuple, TypeVar
    import torch

    # Define the type for nested lists of tensors
    T = TypeVar('T', bound=torch.Tensor)
    NestedList = Union[T, List['NestedList']]

    class TensorChannelManager:
        # (Previous methods, like batch...)

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


class SpecMap:




class HeaderSpec:
    """
    The `HeaderSpec` class abstracts the layout of data in a multimodal token channel (MTC) tensor,
    enabling the model to set or retrieve features needed for block generation without directly
    interacting with the underlying data structure. It specializes in managing the header encodings
    required for Block Multimodal Decoding, where various modes (e.g., image, text) and structural
    attributes (e.g., shape) must be predicted to facilitate flexible content generation.

    This class is designed to provide a clear interface for creating, manipulating, and extracting
    header tensors that contain essential metadata for the model. Unlike payload data, headers are
    metadata blocks that control how subsequent content in the tensor is generated, ensuring
    that mode and shape predictions are properly integrated into the decoding process. This is the
    class that will know what header goes where, and how to extract header data from a tensor.

    **Purpose**:
    The key purpose of this class is to decouple the model's interaction with headers from the
    underlying tensor layout. This allows for flexible data conversion and interaction with
    headers across various TensorChannelSpec schemas, without being tightly bound to a specific
    architectural implementation or task (e.g., inference vs. evaluation).

    **Usage Scenarios**:
    - Creating headers for embedding as input to large language models (LLMs) to predict modes
      (such as content type) and shapes during Block Multimodal Decoding.
    - Extracting specific header information (e.g., mode or shape) from block tensors, enabling
      modular and adaptable interaction with data layouts.
    - Discarding headers when transitioning from MTC blocks to content data during tasks like detokenization.

    **Instantiation**:
    The class is instantiated by binding it to a `TensorChannelSpec` and specifying a sequence of
    headers. It will generate or extract these headers in the given order, allowing flexible
    interaction with different tensor schemas.

    :param channel_spec: The `TensorChannelSpec` that defines the channel allocations for the headers.
    :param headers: A list of headers (as strings) that specify the sequence of header generation.

    Example:
    ```python
    channel_spec = TensorChannelSpec(...)
    headers = ["mode", "shape"]
    header_spec = HeaderSpec(channel_spec, headers)
    ```

    Methods:
    --------
    - `create_headers`: Generates a collection of header tensors for embedding.
    - `extract_header`: Extracts a specific header from a block tensor.
    - `discard_headers`: Removes all header information from a block tensor, preserving only the data content.
    """
    @property
    def headers_length(self)->int:
        return len(self.headers)
    def __init__(self,
                 channel_spec: TensorChannelManager,
                 headers: List[str]
                 ):
        """
        :param channel_spec: The channel spec to bind to
        :param headers: The headers to expect, and in the indicted order
        """
        assert all(header in channel_spec.channel_allocs for header in headers)
        self.channel_spec = channel_spec
        self.headers = headers
    def create_headers(self,
                             header_tensors: Dict[str, torch.Tensor]
                             )->torch.Tensor:
        """
        Creates a collection of "header" tensors based on the provided header
        tensor dictionary. Notably, to allow downstream generative logic discretion
        on operation if desired, header construction is cumulative such that
        the later headers will contain the information from earlier headers, plus
        new details.

        Additionally, header construction can be incomplete, to allow embedding during
        generative processes when we might be making the headings in the first place.
        In this situation, you may provide only a few of the headings - HOWEVER, these must
        be provided in order.

        :param header_tensors: Dictionary
            - Note: If incomplete, headers must be defined in order.
            - Key: Channel allocations matching headers associated with the spec
            - Values:  Tensors of shape (..., D) where D is the channel allocation width for
                       each kind, and ... is common among all tensors
        :return:
            - A tensor of shape (..., header_length, channel_length).
        """
        # Subselect the headers that are active, and validate
        headers = self.headers[:len(header_tensors)]
        assert set(header_tensors.keys())==set(headers), "header tensor keys missing or in wrong order"

        # Construct the headers
        header_constructor = {channel: None for channel in self.channel_spec.channel_allocs}
        output = []
        for header in headers:
            header_constructor[header] = header_tensors[header]
            output.append(self.channel_spec.combine(header_constructor))

        # Return
        return torch.stack(output, dim=-2)

    def extract_header(self, tensor: torch.Tensor, header: str)->torch.Tensor:
        """
        Extracts the value of a particular header tensor. The header
        can be further disassembled using a TensorChannelSpec to get the actual
        values if needed.

        :param tensor:
          - A tensor in channel spec form, which is presumed to have headers attached.
          - Shape (..., items, channels)
        :param header: The header to extract
        :return: A tensor with reduced dimensions, which comes from getting that header.
        """
        assert header in self.headers
        index = self.headers.index(header)
        return tensor.index_select(dim=-2, index=index)

    def discard_headers(self, tensor: torch.Tensor)->torch.Tensor:
        """
        Completely discards a section of header length from an incoming
        MTC encoded collection. This can be used to remove metadata from
        encodings before decoding

        :param tensor:
            - The tensor to remove the headings from
            - Shape (..., items, channels)
        :return:
            - A tensor with the header region removed
        """
        return tensor[..., self.headers_length:, :]

class ModeSpec:
    """
    The mode spec keeps track of the various modes that
    are currently active, and what integers are associated
    with what modes. It also abstracts away the process of
    checking for modes
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
