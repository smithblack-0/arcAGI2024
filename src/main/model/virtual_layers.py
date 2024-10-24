import textwrap

import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple, List, Dict, Any, Optional, Callable, Generator, Type, Union
from abc import abstractmethod, ABC
from src.main.model.base import StatefulCore, TensorTree, parallel_pytree_map
from dataclasses import dataclass

"""
The virtual_layers module is generally centered around selecting,
managing, and using parallel virtual layers, which were historically 
referred to as "banks."
"""


# Some general helper layers. Maybe I should put these elsewhere?
class DropoutLogits(nn.Module):
    """
    A dropout layer for logits, which applies dropout by masking logits to
    a large negative value (epsilon) to simulate "dropping out" certain logits.
    """

    def __init__(self, p: float = 0.5, epsilon: float = -1e9):
        """
        Initialize the DropoutLogits layer.
        :param p: The probability of dropping out a logit (default is 0.5).
        :param epsilon: The large negative value used for masking (default is -1e9).
        """
        super(DropoutLogits, self).__init__()
        self.p = p
        self.epsilon = epsilon

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply dropout to the logits by masking them to a large negative value (epsilon).
        :param logits: The input logit tensor.
        :return: The logits with dropout applied.
        """
        if self.training:
            # Create a dropout mask (0 = drop, 1 = keep)
            dropout_mask = torch.bernoulli(torch.full_like(logits, 1 - self.p))

            # Apply the dropout mask, setting dropped logits to epsilon
            logits = torch.where(dropout_mask == 1, logits, torch.full_like(logits, self.epsilon))

        return logits


@dataclass
class SelectionSpec:
    """
    A specification for selecting and weighting different virtual layers
    (previously referred to as "banks") in the context of a virtualized
    model architecture.

    This structure encapsulates both the indices of the virtual layers involved
    in the computation and the associated probabilities or weights that define
    their contributions. It is particularly useful when multiple virtual layers
    are selected and their contributions need to be combined, either through
    superposition or discrete selection.

    ---- Broadcast and Interpretation Behavior ----
    The `SelectionSpec` serves as a flexible way to express which virtual layers
    (or banks) should be used for a given operation. However, the actual **interpretation**
    of this selection—how it influences tensor operations—depends on the class that uses it
    (such as `VirtualBuffer`, `VirtualParameter`, etc.).

    These classes take the `SelectionSpec` and **interpret it dynamically**:
    - Some classes may add **extra dimensions** to the data (for instance, for batch-like features).
    - Others might combine selected layers by broadcasting over additional data dimensions.

    The selection and broadcasting behavior is not hardcoded into `SelectionSpec` itself but
    rather relies on the class that interprets it. This flexibility allows the same `SelectionSpec`
    to be reused across different operations, each adapting the selection mechanism according to
    its own needs.

    ---- Example: Batch and Data Dimensions ----
    When applied, the selection mechanism can target different combinations of batch and data dimensions.
    For example, a selection might apply across batch dimensions, but depending on the context, the selected
    layers can be **broadcasted over data dimensions** to form a complete tensor expression. This reduces
    the need for restructuring tensors and allows operations to adapt to multi-dimensional tensors efficiently.

    ---- Fields ----
    selection_index (torch.Tensor):
        An integer tensor containing the indices of the virtual layers selected
        for the current operation. Each index points to a specific layer in a
        collection of virtual layers.
        - Shape: (..., num_selected_virtual_layers).
        The shape can vary depending on how many dimensions the selection applies to.
        Batch dimensions are explicitly handled, while data dimensions are broadcast as needed.

    selection_probabilities (torch.Tensor):
        A tensor containing the probabilities or weights that determine how
        strongly each selected virtual layer contributes to the final result.
        This is used to calculate a weighted superposition of the selected layers.
        - Shape: (..., num_selected_virtual_layers).
        Like `selection_index`, this tensor's shape can accommodate different dimensions,
        and will broadcast over remaining data dimensions as required.
    """
    selection_index: torch.Tensor
    selection_probabilities: torch.Tensor


def virtual_state_select(state: torch.Tensor,
                         selection: SelectionSpec,
                         dim: int,
                         superposition: bool = True) -> torch.Tensor:
    """
    Selects and compresses a subset of virtual layers (also called "banks")
    from the provided `state` tensor along the specified dimension (`dim`).
    This operation either combines the selected layers into a weighted
    superposition or keeps them separate, depending on the `superposition` flag.

    ---- Superposition and Selection ----

    A "superposition" refers to a weighted combination of multiple virtual layers,
    where the selected layers are combined according to probabilities from
    the `SelectionSpec`. The final output is a tensor that represents the
    weighted sum of the selected layers.

    When `superposition=False`, the function gathers the selected layers
    and returns them separately along the specified dimension. It does not
    use the probabilities to weight them.

    ---- Parameters ----
    :param state:
        - A tensor representing the current state from which virtual layers
          are selected.
        - Shape: (...batch, ..., options, ...), where the `options` dimension
          contains the available virtual layers.
        - The `state` tensor may contain additional data dimensions, which
          will be broadcast automatically.

    :param selection:
        - A `SelectionSpec` object containing the selection indices and
          probabilities.
        - `selection_index`: The indices of the virtual layers to select
          (tensor of shape: (...batch, selected)).
        - `selection_probabilities`: Probabilities or weights for each
          selected virtual layer (tensor of shape: (...batch, selected)).

    :param dim:
        - The dimension of the `state` tensor that represents the "virtual layer"
          or "bank" dimension.
        - The operation gathers values along this dimension.

    :param superposition:
        - If `True`, the function reduces the selected layers into a single
          tensor by weighting them according to `selection_probabilities`.
        - If `False`, the selected layers are returned without combining,
          and the result will contain separate layers in the selected dimension.

    ---- Returns ----
    :return: A tensor containing the selected and optionally superposed features.
        - If `superposition=True`, returns a tensor of shape (...batch, ...),
          collapsing the selected layers.
        - If `superposition=False`, returns a tensor of shape (...batch, ...,
          selected, ...), keeping the selected layers separate.

    ---- Key Notes ----
    - The `state` tensor can include common batch dimensions as well as extra
      data dimensions.
    - The `SelectionSpec` defines how the selection and weighting of virtual
      layers occur, but the interpretation of the selected layers (and
      broadcasting) is handled within this function.
    """
    # Unpack selection tuple
    indices = selection.selection_index
    probabilities = selection.selection_probabilities

    # Move the selection dim to the end of the state tensor
    state = state.movedim(dim, -1)  # (...batch, ..., options)

    # Ensure indices has the correct number of dimensions
    while indices.dim() < state.dim():
        indices = indices.unsqueeze(-2)
        probabilities = probabilities.unsqueeze(-2)

    # Expand indices to match the state tensor, excluding the last dimension
    indices = indices.broadcast_to(*state.shape[:-1], indices.shape[-1])  # (...batch, ..., selections)

    # Perform gather on the last dimension (which used to be dim)
    selected_state = state.gather(-1, indices)  # (...batch, ..., selections)

    # If requested, reduce and form the superposition.
    # Else, restore original shape
    if superposition:
        selected_state = torch.sum(selected_state * probabilities, dim=-1)
    else:
        selected_state = selected_state.movedim(-1, dim)

    return selected_state


def virtual_state_scatter(state: torch.Tensor,
                          substate: torch.Tensor,
                          selection: SelectionSpec,
                          dim: int,
                          superposition: bool = True
                          ) -> torch.Tensor:
    """
    Inserts a `substate` tensor into a larger `state` tensor along the specified
    `dim` dimension. The insertion is guided by indices and interpolation
    probabilities provided in the `SelectionSpec`. This process interpolates
    between the original `state` values and the new `substate` values.
    You might think of it as "smearing" the substate across the selection
    weighted by the selection probabilities. A probability of 1.0 for a position
    results in a complete replacement of the element

    ---- Parameters ----
    :param state: The original state tensor to update.
        - If `superposition=False`: Shape (..., batch_size, ..., num_options, ...)
        - If `superposition=True`: Shape (..., batch_size, ...)
        - The base tensor that gets updated with interpolated values from `substate`.

    :param substate: The tensor holding new values to insert into `state`.
        - Shape: (..., batch_size, ..., num_selected, ...)
        - The values in this tensor are inserted into `state`, using the indices
          and probabilities from `selection`.

    :param selection: A `SelectionSpec` dataclass that specifies:
        - `selection_index`: Tensor containing indices of the positions in `state`
                             to update. Shape (..., batch_size, num_selected).
        - `selection_probabilities`: Tensor with interpolation weights for each
                                     selected index. Shape (..., batch_size,
                                     num_selected).
        - These control both the locations in `state` to update and the strength
          of the update from `substate`.

    :param dim: The dimension of `state` where the updates occur.
        - This is the axis in `state` where substate values will be inserted,
          based on the indices from `selection`.

    :param superposition: Boolean flag that indicates whether `state` was
        previously reduced by a superposition operation (i.e., dimensions
        collapsed or squeezed).
        - If `True`, the function restores the missing dimension in `state` to
          match `substate` shape before performing the update.

    ---- Returns ----
    :return: A tensor with the same shape as `state`, updated using interpolated
             values from `substate` at positions specified by `selection`.

    ---- Behavior ----
    - **Superposition Handling**: If `superposition=True`, the function first
      expands the missing dimension in `state` to align with `substate`, ensuring
      the update applies correctly.

    - **Gather and Interpolate**: The function gathers values from `state` at
      the specified indices, interpolates between them and `substate` using the
      probabilities, and scatters the updated values back into `state`.

    ---- Key Notes ----
    - The `state` tensor can contain extra batch or data dimensions, and this
      function automatically broadcasts to handle these.
    - The `SelectionSpec` defines the locations and weights for interpolation,
      but the actual dimensions are handled dynamically by this function based on
      the `state` and `substate`.
    """
    # Data standardization.
    if superposition:
        substate = substate.unsqueeze(dim)

    # Run basic error checking to check sanity of state and substate

    if state.dim() != substate.dim():
        raise ValueError("State and substate must have the same dimensionality. Broadcasted scatters not allowed")

    state_shape = list(state.shape)
    substate_shape = list(substate.shape)

    state_shape.pop(dim)
    substate_shape.pop(dim)

    if state_shape != substate_shape:
        raise ValueError("State and subshape must be the same shape except for dim.")

    # Unpack

    indices = selection.selection_index
    probabilities = selection.selection_probabilities

    # If in a superposition, the indicated dimension was actually squeeze away
    # earlier. Restore it. This is essentially data standardization.

    # Move the selection dim to the end of the state tensor for easier processing
    state = state.movedim(dim, -1)  # (...batch, ..., options)
    substate = substate.movedim(dim, -1)  #(...batch, ..., selections

    # Ensure indices has the correct number of dimensions for broadcasting
    while indices.dim() < state.dim():
        indices = indices.unsqueeze(-2)
        probabilities = probabilities.unsqueeze(-2)

    # Expand indices  to match the state tensor
    indices = indices.broadcast_to(*state.shape[:-1], indices.shape[-1])  # (...batch, ..., selections)

    # Gather the current state values at the specified indices
    gathered_state = state.gather(-1, indices)  # (...batch, ..., selections)

    # Perform interpolation between the gathered state and the substate using probabilities
    interpolated = (1 - probabilities) * gathered_state + probabilities * substate  # (...batch, ..., selections)

    # Scatter the interpolated values back into the original state tensor
    state = state.scatter(-1, indices, interpolated)

    # Restore the original dimension order by swapping back
    state = state.movedim(-1, dim)

    return state


class VirtualParameter(nn.Module):
    """
    The `VirtualParameter` class represents a parameter with a hidden "bank"
    dimension. This bank dimension allows the parameter to exist in multiple
    variations, called a parameter bank. When a `SelectionSpec` is provided,
    a superposition of the parameters is created by weighting and combining
    values across the bank dimension according to the given probabilities.

    This allows for dynamic access to different parameter configurations,
    making it suitable for models with virtual layers. It is used heavily in
    the upcoming wrapper class.

    It can either be initialized directly with a parameter — in which case
    the last dimension becomes the bank dimension — or one can use the
    `.create` method to make a parameter according to a given specification.
    """

    @classmethod
    def create(cls,
               bank_size: int,
               shape: Tuple[int, ...],
               device: Optional[torch.device] = None,
               dtype: Optional[torch.dtype] = None,
               init: Optional[Callable[[torch.Tensor], None]] = None
               ) -> 'VirtualParameter':
        """
        Creates and initializes a `VirtualParameter` with a bank of parameters,
        according to the given specification.

        :param bank_size: The number of different parameter configurations
                          in the bank.
        :param shape: The shape of each individual parameter in the bank.
        :param device: The device to store the parameter on. Defaults to
                       the current device.
        :param dtype: The data type of the parameter. Defaults to `None`,
                      which means the default dtype is used.
        :param init: An optional initialization function that takes a
                     `torch.Tensor` and initializes the parameter in place.
        :return: An instance of the `VirtualParameter` class.
        """
        parameter = torch.zeros([*shape, bank_size, ], device=device,
                                dtype=dtype)
        if init is not None:
            init(parameter)
        return cls(parameter)

    def __init__(self,
                 parameter: nn.Parameter):
        """
        Initializes a `VirtualParameter` with a bank of parameters.

        :param parameter: The parameter to initialize. The first dimension
                          will always be the bank dimension.
        """
        assert parameter.dim() > 0, "No bank dimension was specified."

        super().__init__()
        self.bank_size = parameter.shape[-1]
        self.shape = parameter.shape[:-1]
        self.device = parameter.device
        self.dtype = parameter.dtype
        self.parameter = nn.Parameter(parameter)

    def forward(self, bank_spec: SelectionSpec) -> torch.Tensor:
        """
        Returns a superposition of parameters based on the provided
        `bank_spec`.

        The `bank_spec` provides both the selected bank indices and their
        corresponding probabilities. The selected parameters are interpolated
        according to these probabilities to form a final output.

        :param bank_spec: A `SelectionSpec` dataclass containing:
            - `selection_index`: The indices of the selected banks. (..., selected)
            - `selection_probabilities`: The probabilities for weighting
                                         the selected banks. (..., selected)

        :return: A tensor containing the weighted combination of selected
                 parameters. Shape: (..., *shape)
        """
        # Set aside the bank dimension, measured from the end of the tensor with negative
        # style indexing. Extract the selected banks using advanced indexing
        parameter = self.parameter.movedim(-1, 0)  # (*shape, bank, )
        parameter = parameter[bank_spec.selection_index]  # (..., *shape,  bank_selected,)
        parameter = parameter.movedim(-self.parameter.dim(), -1)
        # Setup the probabilities. The probability tensor need to have
        # enough elements added to it to broadcast across the shape portions
        # of the parameter tensor. We add those.
        probabilities = bank_spec.selection_probabilities  # (..., bank_selected)
        for _ in self.shape:
            probabilities = probabilities.unsqueeze(-2)

        # Perform the kernel superposition, and return the result.
        parameter = torch.sum(parameter * probabilities, dim=-1)  # (..., *shape)
        return parameter


class VirtualBuffer(nn.Module):
    """
    This is an instance of a "virtual buffer",
    which is a collapsable feature using the
    virtual selection mechanism. A buffer is internally
    maintained across a "banks" dimension, and one can
    ask to instance a version of the buffer, or update
    the buffer again based on the instance.

    The bank dimension is internally maintained as the
    last dimension, which influences how the selection
    mechanism operates.
    """
    buffer: torch.Tensor

    @classmethod
    def create(cls,
               bank_size: int,
               shape: Tuple[int, ...],
               device: Optional[torch.device] = None,
               dtype: Optional[torch.dtype] = None,
               init: Optional[Callable[[torch.Tensor], None]] = None
               ) -> 'VirtualBuffer':
        """
        Creates and initializes a `VirtualBuffer` with a bank of buffer,
        according to the given specification.

        :param bank_size: The number of different parameter configurations
                          in the bank.
        :param shape: The shape of each individual parameter in the bank.
        :param device: The device to store the parameter on. Defaults to
                       the current device.
        :param dtype: The data type of the parameter. Defaults to `None`,
                      which means the default dtype is used.
        :param init: An optional initialization function that takes a
                     `torch.Tensor` and initializes the parameter in place.
        :return: An instance of the `VirtualBuffer` class.
        """
        # Bank dimension is now the last dimension of the buffer tensor
        buffer = torch.zeros([*shape, bank_size], device=device, dtype=dtype)
        if init is not None:
            init(buffer)
        return cls(buffer)

    def __init__(self, buffer: torch.Tensor):
        """
        Initialize a virtual buffer class. The buffer passed
        in has its last dimension declared as the bank dimension.
        Keep that in mind.
        :param buffer:
            - The buffer to initialize with
            - Keep in mind the shape (..., buffer_banks)
        """
        assert buffer.dim() > 0, "No bank dimension was specified."

        super().__init__()
        self.shape = buffer.shape[:-1]  # All dimensions except the last are the shape
        self.bank_size = buffer.shape[-1]  # Last dimension is the bank dimension
        self.device = buffer.device
        self.dtype = buffer.dtype
        self.register_buffer("buffer", buffer)  # (...batch, ...data, buffer_size)

    def express_buffer(self,
                       selection: SelectionSpec,
                       superposition: bool = True
                       ) -> torch.Tensor:
        """
        Expresses the buffer in a way that it becomes nonvirtual.

        If `superposition=True`, the output will be a weighted combination (superposition)
        of selected buffer values, collapsing the virtual banks into a single buffer
        based on the selection probabilities.

        If `superposition=False`, the buffer values will remain separated, with each
        buffer indexed by the selection indices.

        :param selection: A `SelectionSpec` dataclass containing:
            - `selection_index`: The indices of the selected banks. (..., selected)
            - `selection_probabilities`: The probabilities for weighting
                                         the selected banks. (..., selected)
        :param superposition: Whether or not to superimpose the buffer. Defaults to True.
            - If True, returns a weighted combination of buffer states based on probabilities.
            - If False, returns the selected individual buffer states.
        :return:
            - If `superposition=True`: Returns the combined buffer of shape (...).
            - If `superposition=False`: Returns the selected buffers of shape (..., selected).
        """
        # Basic sanity checking. The buffer shape and selection shape
        buffer = self.buffer  # (...batch, ...data, options)
        return virtual_state_select(buffer, selection, dim=-1, superposition=superposition)

    def update_buffer(self,
                      expression: torch.Tensor,
                      selection: SelectionSpec,
                      superposition: bool = True):
        """
        Updates the currently contained buffer based on the given
        buffer expression, and the selection. Superposition allows
        handling certain scenarios.

        :param expression: The expression to update the buffer with.
            - Shape (buffers_selected, ...) if expressed using superposition off.
            - Shape (...) if expressed using superposition on.
        :param selection: The selections for the virtual buffer.
        :param superposition: Whether or not it was expressed with the superposition on.
        """
        # Scatter the updated expression into the last dimension (bank dimension)
        self.buffer = virtual_state_scatter(self.buffer, expression, selection,
                                            dim=-1, superposition=superposition)


class VirtualState:
    """
    This is an instance of a "virtual state", which is a collapsable feature
    using the virtual selection mechanism. Similar to a virtual buffer, a state
    is internally maintained across a "banks" dimension. One can instance a
    version of the state or update the state based on the instance.

    Unlike `VirtualBuffer`, `VirtualState` does not register as an `nn.Module`,
    and stores the state directly as a `torch.Tensor` without being part of the
    model's module hierarchy. The bank dimension is maintained as the last
    dimension, which influences how selection operates.
    """

    @classmethod
    def create(cls,
               bank_size: int,
               shape: Tuple[int, ...],
               device: Optional[torch.device] = None,
               dtype: Optional[torch.dtype] = None,
               init: Optional[Callable[[torch.Tensor], None]] = None
               ) -> 'VirtualState':
        """
        Creates and initializes a `VirtualState` with a bank of states according
        to the given specification.

        :param bank_size: The number of different state configurations in the bank.
        :param shape: The shape of each individual state in the bank.
        :param device: The device to store the state on. Defaults to the current device.
        :param dtype: The data type of the state. Defaults to `None`, which means the default dtype is used.
        :param init: An optional initialization function that takes a `torch.Tensor` and
                     initializes the state in place.
        :return: An instance of the `VirtualState` class.
        """
        # The bank dimension is now the last dimension in the state tensor
        state = torch.zeros([*shape, bank_size], device=device, dtype=dtype)
        if init is not None:
            init(state)
        return cls(state)

    def __init__(self, state: torch.Tensor):
        """
        Initializes a `VirtualState`. The `state` passed in has its last dimension
        declared as the bank dimension.

        :param state: The tensor representing the virtual state, with the shape
                      (..., bank_size), where the last dimension is the bank dimension.
        """
        assert state.dim() > 0, "No bank dimension was specified."

        self.shape = state.shape[:-1]  # All dimensions except the last are the shape
        self.bank_size = state.shape[-1]  # Last dimension is the bank dimension
        self.device = state.device
        self.dtype = state.dtype
        self.state = state

    def express_state(self,
                      selection: SelectionSpec,
                      superposition: bool = True
                      ) -> torch.Tensor:
        """
        Expresses the state in a way that it becomes nonvirtual.

        If `superposition=True`, the output will be a weighted combination (superposition)
        of selected state values, collapsing the virtual banks into a single state based on
        the selection probabilities.

        If `superposition=False`, the state values will remain separated, with each state
        indexed by the selection indices.

        :param selection: The selection for the state to operate under.
        :param superposition: Whether or not to superimpose the state. Defaults to True.
            - If True, returns a weighted combination of state values based on probabilities.
            - If False, returns the selected individual state values.
        :return:
            - If `superposition=True`: Returns the combined state of shape (...).
            - If `superposition=False`: Returns the selected states of shape (..., selected).
        """
        # Express the state using the last dimension as the bank dimension
        return virtual_state_select(self.state, selection, dim=-1, superposition=superposition)

    def update_state(self,
                     expression: torch.Tensor,
                     selection: SelectionSpec,
                     superposition: bool = True):
        """
        Updates the current state based on the given expression and selection.
        The superposition flag allows handling different scenarios.

        :param expression: The expression to update the state with.
            - Shape (..., selected, ...) if expressed using superposition=False.
            - Shape (...) if expressed using superposition=True.
        :param selection: The selections for the virtual state.
        :param superposition: Whether or not it was expressed with the superposition on.
        """
        # Scatter the expression into the last dimension (bank dimension)
        self.state = virtual_state_scatter(self.state, expression, selection, dim=-1, superposition=superposition)


class _VirtualModule(nn.Module):
    """
    The virtual module implementation. This wraps
    the core code, maintains links to lower layers
    in the layer hierarchy, and promises to
    update the superposition when commanded. This
    is not meant to be used directly

    It is responsible for accepting a collection of
    core layers to build parameter banks out of,
    and recursively setting up a virtual module.

    WARNING: This modifies the layers provided.
    Only use it directly if you are okay with that.
    """

    def create_virtual_parameters(self,
                                  core_layers: List[nn.Module]
                                  ) -> Dict[str, VirtualParameter]:
        """
        Creates the virtual parameter banks for these classes
        :param core_layers: The list of core layers to extract parmeters from
        :return: A dict of VirtualParameter objects
        """
        parameter_banks = {}
        for name, _ in core_layers[0].named_parameters(recurse=False):
            parameters = [layer.get_parameter(name) for layer in core_layers]
            parameters = torch.stack(parameters, dim=0)
            parameter_banks[name] = VirtualParameter(parameters)
        return parameter_banks

    def create_virtual_submodules(self,
                                  core_layers: List[nn.Module]
                                  ) -> Dict[str, '_VirtualModule']:
        """
        Creates a dict of virtual submodules. These are both
        the virtual submodules, and what name they were initalized
        with

        :param core_layers: The layers to create this from
        :return: The dict of virtual submodules
        """
        # Create any required sub virtual modules recurrently
        virtual_modules = {}
        for name, _ in core_layers[0].named_children():
            submodules = [layer.get_submodule(name) for layer in core_layers]
            virtual_modules[name] = _VirtualModule(submodules)
        return virtual_modules

    def create_virtual_buffers(self,
                               core_layers: List[nn.Module]
                               ) -> Dict[str, VirtualBuffer]:
        """
        Creates a dict of all the virtual buffers that we
        need to handle. Returns this dictionary.
        :param core_layers: The core layer collection to build from
        :return: The dict of virtual buffers
        """
        virtual_buffers: Dict[str, VirtualBuffer] = {}
        for name, _ in core_layers[0].named_buffers(recurse=False):
            buffers = [layer.get_buffer(name) for layer in core_layers]
            buffers = torch.stack(buffers, dim=-1)
            virtual_buffers[name] = VirtualBuffer(buffers)
        return virtual_buffers

    def __init__(self, core_layers: List[nn.Module]):
        super().__init__()

        # Create the virtual features. This includes the virtual
        # submodules, the virtual parameters, and the virtual

        virtual_buffers = self.create_virtual_buffers(core_layers)
        virtual_parameters = self.create_virtual_parameters(core_layers)
        virtual_modules = self.create_virtual_submodules(core_layers)

        # Store some of them. These will need to be accessed again in
        # the future

        self.parameter_banks = nn.ModuleDict(virtual_parameters)
        self.buffer_banks = nn.ModuleDict(virtual_buffers)

        # We need to seriously mangle one of the core layers to use it
        # for my logic. First, we are going to go ahead and replace
        # all modules with the associated virtual modules instead. Then,
        # we delete any parameters and replace them with buffers.

        core_logic = core_layers[0]



        for name, module in core_logic.named_children():
            setattr(self, name, virtual_modules[name])
        for name, parameter in core_logic.named_parameters(recurse=False):
            setattr(self, name, virtual_parameters[name])
        for name, buffer in core_logic.named_buffers(recurse=False):
            setattr(self, name, virtual_buffers[name])

        # Store the modified instance.
        self.core_logic = core_logic

    def set_superposition(self, selection: SelectionSpec):
        """
        Sets the superposition of the virtual module
        and all attached submodules to the given setup
        :param selection: The selection feature indicating the superposition to set
        """
        # Set all submodules to the correct superposition.
        for module in self.core_logic.modules():
            module: _VirtualModule
            module.set_superposition(selection)

        # Now, create the parameter superposition and insert it into
        # the logic bank.

        for name, virtual_parameter in self.parameter_banks.items():
            virtual_parameter: VirtualParameter
            parameter_superposition = virtual_parameter(selection)
            self.core_logic.register_buffer(name, parameter_superposition)

        # Do the same thing for each buffer banks
        for name, virtual_buffer in self.buffer_banks.items():
            virtual_buffer: VirtualBuffer
            buffer_superposition = virtual_buffer.express_buffer(selection, superposition=True)
            self.core_logic.register_buffer(name, buffer_superposition)

        # The superposition is now completely setup

    def update_buffers(self, selection: SelectionSpec):
        """
        Once a forward pass has been run, we have to manually
        invoke this to update the virtual buffer. Recursively
        updates the buffers for all virtual modules
        :param selection: The selection to do the update under
        """

        # Update all children's buffers
        for module in self.core_logic.modules():
            module: _VirtualModule
            module.update_buffers(selection)

        # Update my personal buffers
        for name, virtual_buffer in self.buffer_banks.items():
            virtual_buffer: VirtualBuffer
            virtual_buffer.update_buffer(self.core_logic.get_buffer(name),
                                         selection,
                                         superposition=True)
        # Done.

    def forward(self, *args, **kwargs) -> Any:
        """
        Runs the virtual module. This basically just involves invoking
        the core logic.
        :param args: The args to invoke it with
        :param kwargs: The kwargs to invoke it with
        :return: The return. Very generic
        """
        return self.core_logic(*args, **kwargs)


class VirtualLayer(nn.Module):
    """
    Turns the given layer or stack of layers into a
    set of responsive virtual layers. Dynamically swaps
    out the parameter kernels on a fully initialize
    layer instance in order to act as a virtual layer too

    Importantly, this operates by inspection and some hard
    decisions had to be made. Your warnings:

    WARNINGS/NOTES:
    - Tensors which are buffers, but NOT floating will not be almagated across
      the various layers. Instead, the first layer is used.
    - Features that are not modules, parameters, or buffers will NOT be turned
      into virtual layers
    """

    @classmethod
    def create_from_factory(cls,
                            layer: Type[nn.Module],
                            bank_size: int,
                            *args,
                            **kwargs
                            ) -> 'VirtualLayer':
        """
        Creates a virtual layer by repeatedly instancing the
        same layer over and over again. This should be the preferred
        initialization method, as it results in parameter banks that
        are independently initialized across different bank layers.

        :param layer_factory: The layer to initialize over and over again
        :param bank_size: The size of the layer bank to make
        :param args: The args to invoke with
        :param kwargs: The kwargs to invoke with
        :return: The virtual layer wrapper setup
        """

        layers = [layer(*args, **kwargs) for _ in range(bank_size)]
        return cls(layers)

    @classmethod
    def create_from_layer(cls,
                          layer: nn.Module,
                          bank_size: int
                          ) -> 'VirtualLayer':
        """
        Create a virtual layer from the given layer
        of size bank size. This generally should NOT be
        preferred, as it will result in all parameters being
        synchronous between the banks. While it is still possible
        to break this during training, it is likely slower.

        :param layer: The layer to prototype with
        :param bank_size: The size of the bank to make
        :return: The virtual layer
        """
        layers = [layer for _ in range(bank_size)]
        return cls(layers)

    @classmethod
    def create_from_layers_stack(cls,
                                 layers: List[nn.Module]
                                 ) -> 'VirtualLayer':
        """
        Creates a working layer module from a layer
        stack. If you have, for instance, a pretrained
        transformer you want to use in virtual layer construction
        this would be a good way to do so, so long as all
        layers were constructed the same. Which is commonly the case

        :param layers: The layers to make a virtual layer out of
        :return: The initialized virtual layer
        :raises TypeError: If your provided list is not compatible
        """
        # We basically just perform validation here. The way we do that is
        # by getting the parameter based save dictionaries. Then we verify
        # that everything that is a tensor has the same shape, and everything
        # that is not a tensor is equal
        assert len(layers) > 0

        def throw_if_not_valid(*states: Union[torch.Tensor, Any]):
            target = states[0]
            target_type = type(target)
            for i, state in enumerate(states[1:]):

                # Handle simple type matching. At the same
                # place, we should be of the same type
                if not isinstance(state, target_type):
                    msg = f"""
                    Save states were not syncronized. Types 
                    differed between 0 and {i + 1}
                    """
                    msg = textwrap.dedent(msg)
                    raise TypeError(msg)

                # Handles tensor shape matching,
                # or value matching
                if torch.is_tensor(target):
                    if target.shape != state.shape:
                        msg = f"""
                        Shapes for stored information differ
                        between 0 and {i + 1}. 
                        
                        Expected: {target.shape}
                        Got: {state.shape}
                        """
                        msg = textwrap.dedent(msg)
                        raise ValueError(msg)
                else:
                    if target != state:
                        msg = f"""
                        Values of stored information differ
                        between 0 and {i + 1}. This is only
                        allowed for tensors.
                        
                        Expected: {target}
                        Got: {state}
                        """
                        msg = textwrap.dedent(msg)
                        raise ValueError(msg)

        # Get and apply the validation
        save_states = [layer.state_dict() for layer in layers]
        parallel_pytree_map(throw_if_not_valid, *save_states)

        # Set up

        return cls(layers)

    def __init__(self, layers: List[nn.Module]):
        """
        A Virtual layer wrapper should always be initialized
        with a stack of layers which are all of the same
        type and which were setup with the same arguments,
        ensuring compatibility. More complex functionality
        is provided by some of the create builders
        :param layers: The core layers to initialize using
        """

        super().__init__()
        self.virtual_layer = _VirtualModule(layers)

    def forward(self, *args, selection: SelectionSpec, **kwargs) -> Any:
        """
        Runs the virtual layer implementaton. The selection must
        go after all other arguments. Note that this is the interface
        level, and as such the user must pass the selection spec
        each invokation.

        :param args: The args to pass to the wrapped layer
        :param selection: The selection feature indicating the virtual layer to build
        :param kwargs: The kwargs to pass to the wrapped layer
        :return: The result of the run
        """
        self.virtual_layer.set_superposition(selection)
        output = self.virtual_layer(*args, **kwargs)
        self.virtual_layer.update_buffers(selection)
        return output


## Bank selector logic.
#
# There will be some helper functions, then
# actual bank selection mechanisms
##

def make_top_p_selection_mask(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """
    Selection mechanism for the top-p selection type, or nucleus
    selection. In this mode, we collect cases until the probablity
    mass exceeds a threshold, then no more. We form a mask that shows
    what elements we want to select.

    :param logits: The logits to select from. Shape (..., logits)
    :param top_p: The cumulative probability threshold for top-p selection.
    :return: The selected top k mask. Shape (..., logits). Bool of true means selected
    """
    # Basic sanity check
    if not 1.0 >= top_p >= 0.0:
        raise ValueError(f"Top p should have been between 0 and 1 inclusive. Given was {top_p}")

    # Create the default selection mask
    selection_mask = torch.zeros_like(logits, dtype=bool)

    # Skip further computation and return immediately
    # if top p is set to not catch anything. If not,
    # we activate our logits so we can do probability
    # mass sampling
    if top_p == 0.0:
        return selection_mask

    probabilities = torch.softmax(logits, dim=-1)

    # We must perform nucleus sampling. This is tricky
    # when vectorized. What we must do is sort the probabilities
    # in ascending order, then figure out when the cumulative
    # sum is under a threshold and mask out everything above it
    #
    # However, what we are ACTUALLY doing is figuring out what
    # the mask looks like in the sorted domain, then moving
    # that back into the unsorted mask.

    ordered_probabilities, sorting_index = probabilities.sort(dim=-1, descending=True)
    cumulative_probabilities = ordered_probabilities.cumsum(dim=-1)
    cumulative_mask = cumulative_probabilities > top_p
    cumulative_mask[..., 0] = True  # First element is always included, to avoid numeric nonsense

    # We now transfer the cumulative mask back into the selection
    # mask, in the designated order.
    selection_mask.scatter_(dim=-1, index=sorting_index, src=cumulative_mask)

    return selection_mask


def make_top_k_selection_mask(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """
    Selection mechanism to make a top k mask. Selects the
    top k elements, and returns a mask indicating what
    elements were selected.

    Returns a top k selection mask based on the provided logits
    :param logits: The logits to select a top k from. (..., logits)
    :param top_k: Integer, indicating how many to select.
    :return: The selected top k mask. Shape (..., logits). Bool of true means selected
    """
    if not top_k >= 0:
        raise ValueError(f"Top k should have been greater than or equal to 0. Given was {top_k}")
    if top_k > logits.size(-1):
        top_k = logits.size(-1)

    selection_mask = torch.zeros_like(logits, dtype=bool)
    if top_k > 0:
        # Only bother to actually compute the top k while the
        # mode is active. We get the indexes associated with
        # the top k, and mark those parts of the mask as active
        _, index = torch.topk(logits, k=top_k, dim=-1)
        selection_mask[index] = True
    return selection_mask


def make_random_selection_mask(logits: torch.Tensor, num: int) -> torch.Tensor:
    """
    Creates a selection mask with num elements selected randomly.

    :param logits: The logits to select a top k from. (..., logits)
    :param num: Integer, indicating how many to randomly select.
    :return: The selected top k mask. Shape (..., logits). Bool of true means selected
    """
    # Basic sanity checking
    if not num >= 0:
        raise ValueError(f"num of selected should have been greater than or equal to 0. Given was {num}")
    if num > logits.size(-1):
        num = logits.size(-1)

    # Create the default selection mask
    selection_mask = torch.zeros_like(logits, dtype=bool)

    # If you are not going to select ANY, just return the default mask
    if num == 0:
        return selection_mask

    # Select a certain number randomly from what remains.
    # We do this by creating a random matrix of the same
    # shape, sorting it, and slicing out the sections needed.
    # This ends up behaving like a vectorized torch.randperm

    random_weights = torch.rand_like(logits)
    randomized_indices = torch.argsort(random_weights, dim=-1)
    randomized_indices = randomized_indices[..., :num]
    selection_mask[randomized_indices] = True

    # Return the selection
    return selection_mask


class AbstractBankSelector(nn.Module, ABC):
    """
    The `AbstractBankSelector` provides a framework for selecting parameter banks (virtual layers)
    from logits and dynamically building a `SelectionSpec`. It supports sparse selection mechanisms
    such as top-k, top-p (nucleus), and random sampling. This class serves as a base for subclasses
    that implement custom logic for generating logits used for bank selection.

    ---- Key Concepts ----
    - `select_logits`: This method generates the final `SelectionSpec`, combining any sparse selections
      such as top-k, top-p, or random sampling. It processes logits into selected indices and their associated
      probabilities. **Subclasses must call this method with their generated logits** to create the selection.
    - `SelectionSpec`: Defines the indices and probabilities for selecting virtual parameters from a bank.
    - `sparse_mode`: Controls how sparse selection is performed, using `top_k`, `top_p`, or `rand`. If none are
      specified, dense mode is used, where **all logits are included, but weighted by their probabilities**.

    ---- Usage ----
    - Subclasses **must implement the `forward` method** to generate logits that represent potential selections.
    - The `select_logits` method should be called within `forward` to convert generated logits into a `SelectionSpec`.
    - `top_k`, `top_p`, and `rand` are mutable fields and can be modified dynamically to adjust selection behavior
      at any point, without reinitializing the selector.

    ---- Initialization ----
    - `top_k`, `top_p`, and `rand` control how many logits are included in the selection:
        * `top_k`: Selects the highest k logits.
        * `top_p`: Uses nucleus sampling to select logits until the cumulative probability reaches p.
        * `rand`: Selects a random subset of logits.
      If all are inactive (set to zero), dense mode is used, where all logits contribute based on their probabilities.
    - `dropout_rate` is applied to the logits before selection, allowing for stochastic behavior during training.

    :param top_k: The number of top logits to select (optional, defaults to 0).
    :param top_p: The cumulative probability threshold for top-p selection (optional, defaults to 0.0).
    :param rand: The number of random logits to select (optional, defaults to 0).
    :param dropout_rate: Dropout rate to apply to the logits during selection (optional, defaults to 0.0).
    """

    #TODO: Add bank selector metrics and, optionally, bank balancing.
    @property
    def is_dense(self) -> bool:
        """
        Tells us whether selection is operating in dense mode. It is dynamically
        computed in case someone changes the fields.
        """
        if not self.top_k == 0:
            return False
        if not self.top_p == 0.0:
            return False
        if not self.rand == 0:
            return False
        return True

    def select_logits(self, logits: torch.Tensor) -> SelectionSpec:
        """
        Creates a working selection spec out of the logits, based on the initialization
        conditions. A variety of configurations can be utilized, and this supports them
        all.

        :param logits: The logits to select from. Shape (..., logits)
        :return: The created selection spec
        """

        # Generate the logit index
        logit_index = torch.arange(
            logits.shape[-1],
            device=logits.device,
            dtype=logits.dtype,
        )

        # Perform normalization
        logits = self.dropout(logits)

        # Detect dense logits, and handle them
        # separately

        if not self.is_dense:

            # Handle and combine all sparse selection mechanisms
            selection_mask = torch.zeros_like(logits, dtype=torch.bool)
            selection_mask = selection_mask | make_top_k_selection_mask(logits, self.top_k)
            selection_mask = selection_mask | make_top_p_selection_mask(logits, self.top_p)
            selection_mask = selection_mask | make_random_selection_mask(logits, self.rand)

            # The number of required logits is determined by counting up how many
            # selections are active, then keeping the maximum. This is used to initialize
            # a space to hold the selected logits. The space is initialized with highly
            # negative defaults, to ensure anything not transferred ends up masked

            num_required_logits, _ = selection_mask.sum(dim=-1).max()
            final_logits = torch.full([*logits.shape[:-1], num_required_logits],
                                      fill_value=-1e+9,
                                      device=logits.device, dtype=logits.dtype)

            # In order to transfer the selections into the final logit feature, we need
            # both a source and a destination mask. The destination mask is made by sorting
            # the selection_mask, then retaining num_required_logits mask entries. This ensures
            # there is always the same number of active entries per batch, ensuring that transfer
            # will be successful. Note that logit order is not guaranteed.

            transfer_mask, transfer_index = torch.sort(selection_mask, dim=-1, descending=True)
            transfer_mask = transfer_mask[..., :num_required_logits]
            transfer_index = transfer_index[..., :num_required_logits]

            # Transfer into the final logit space. Also, use an arange
            # to get an associated index selection

            final_logits[transfer_mask] = logits[selection_mask]
            final_index = logit_index[transfer_index]

        else:
            # Handle dense mode operation. For now, this
            # consists of using the sparse logic.

            final_index = logit_index
            while final_index.dim() < logits.dim():
                final_index = final_index.unsqueeze(-1)

            final_index = final_index.expand_as(logits)
            final_logits = logits

        # Finish up. Active the logits, then return the result
        probabilities = torch.softmax(final_logits, dim=-1)
        return SelectionSpec(final_index, probabilities)

    def __init__(self,
                 top_k: Optional[int] = None,
                 top_p: Optional[float] = None,
                 rand: Optional[int] = None,
                 dropout_rate: Optional[float] = None
                 ):
        """
        The abstract bank selection process cna be defined here
        We can say how many sparsely to include in terms of top-p (nucleous),
        top-k, or randomly selected.

        Do note that not providing any specification for k, p, or rand puts
        us in dense mode, in which we do not reduce at all.

        :param top_k: The top k logits are included.
        :param top_p: Nucleus sampling is done to get the top p logits
        :param rand: This many logits are randomly selected
        :param dropout_rate: The dropout rate to apply to the logits
        """
        super().__init__()

        # Data standardization
        if top_k is None:
            top_k = 0
        if top_p is None:
            top_p = 0.0
        if rand is None:
            rand = 0
        if dropout_rate is None:
            dropout_rate = 0.0

        # Setup
        self.top_k = top_k
        self.top_p = top_p
        self.rand = rand
        self.dropout = DropoutLogits(dropout_rate)

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Tuple[SelectionSpec, Any]:
        """
        Abstract definition of the forward mechanism. This method accepts arbitrary content
        (args and kwargs) and returns a `SelectionSpec` along with any updated recurrent state.

        ---- Parameters ----

        :param args: Positional arguments to be used in the concrete implementation.
        :param kwargs: Keyword arguments to be used in the concrete implementation.
        :return: A tuple containing:
            - `SelectionSpec`: A structure that defines the selected bank indices and the associated probabilities.
            - `state`: The recurrent state, passed along for future calls. Will be invoked as kwargs
        """
        pass


class LinearBankSelector(AbstractBankSelector):
    """
    The `LinearBankSelector` is a simple bank selector that uses a single linear
    layer to generate logits for selecting parameter banks (virtual layers).

    This selector applies the given embedding to the linear layer, processes
    the logits using selection mechanisms (top-k, top-p, or random sampling),
    and returns a `SelectionSpec` for the selected virtual layers.

    ---- Key Features ----
    - Embedding-based selection: Uses the provided embedding to create logits in
      the bank space.
    - Sparse Selection: Applies top-k, top-p, or random sampling to select a subset
      of banks.
    - Simplest form of bank selection: One linear layer and no recurrent state.
    - Providing no sparse selection input places us in dense mode. This is computationally intense.
    """

    def __init__(self,
                 d_model: int,
                 bank_size: int,
                 top_k: Optional[int] = None,
                 top_p: Optional[float] = None,
                 rand: Optional[int] = None,
                 dropout_rate: Optional[float] = None
                 ):
        """
        Initializes the `LinearBankSelector` with the given embedding size, bank size,
        and selection configuration.

        :param d_model: The size of the embeddings that will be provided.
        :param bank_size: The size of the bank selection to create.
        :param top_k: The number of top logits selected (optional, defaults to 0).
        :param top_p: The probability mass to select by (optional, defaults to 0.0).
        :param rand: The number of random logits to include (optional, defaults to 0).
        :param dropout_rate: Logit dropout rate (optional, defaults to 0.0).
        """
        super().__init__(top_k, top_p, rand, dropout_rate)
        self.projector = nn.Linear(d_model, bank_size)

    def forward(self, embedding: torch.Tensor) -> Tuple[SelectionSpec, None]:
        """
        Generates logits for selecting parameter banks based on the provided embedding
        and processes them into a `SelectionSpec` using the configured sparse selection
        mechanisms (top-k, top-p, or random sampling).

        :param embedding: The embedding to process for selecting banks.
        :return:
            - The `SelectionSpec` containing the selected indices and probabilities.
            - `None` as there is no recurrent state in this implementation.
        """
        logits = self.projector(embedding)
        return self.select_logits(logits), None


class PseudoMarkovBankSelector(AbstractBankSelector):
    """
    The Pseudo Markov Bank Selector is a more sophisticated bank selector that
    utilizes persistent state and combines transition-based biases with immediate
    computational results to influence virtual layer selection.

    This class assumes that related virtual layers should be closely linked,
    and it aims to model this through a Markov-like process. The Markov biases,
    or logits, act as transition preferences between virtual layers. These biases
    are dynamically updated and applied to the selection process, promoting or
    demoting certain layers based on the last expressed selection - basically the last
    markov state.

    However, this is a "pseudo" Markov process because, while these biases exist,
    the immediate computation results (from embeddings or other inputs) also play
    a significant role. This means the model can adapt and override the Markov
    biases on-the-fly based on the current task, emphasizing or demoting specific
    virtual layers as needed. This combination of transition biases and current
    computations results in a more flexible and context-sensitive virtual layer
    selection process.

    ---- Fields ----
    - `markov_biases`: A set of trainable biases representing transition preferences
      between virtual layers. These are dynamically updated and applied in combination
      with the embedding-based logits to influence the selection of virtual layers.
    - `state`: The recurrent state passed between iterations, which tracks the current
      transition probabilities across virtual layers.
    """

    def __init__(self,
                 d_model: int,
                 bank_size: int,
                 top_k: Optional[int] = None,
                 top_p: Optional[float] = None,
                 rand: Optional[int] = None,
                 dropout_rate: Optional[float] = None
                 ):
        """
        Initializes the layer with the given embedding size, bank size,
        and selection configuration.

        :param d_model: The size of the embeddings that will be provided.
        :param bank_size: The size of the bank selection to create.
        :param top_k: The number of top logits selected (optional, defaults to 0).
        :param top_p: The probability mass to select by (optional, defaults to 0.0).
        :param rand: The number of random logits to include (optional, defaults to 0).
        :param dropout_rate: Logit dropout rate (optional, defaults to 0.0).
        """
        super().__init__(top_k, top_p, rand, dropout_rate)
        self.d_model = d_model
        self.bank_size = bank_size

        # This bears a little explanation. It turns out
        # running a probability distribution through a linear
        # layer is basically equivalent to weighting a bunch of
        # markov biases by how active that state was. So we
        # can just use a linear projection for that.
        #
        # The embedding projector just works like normal, though.
        self.projector = nn.Linear(d_model, bank_size)
        self.markov_biases = nn.Linear(bank_size, bank_size)

    def forward(self,
                tensor: torch.Tensor,
                state: Optional[torch.Tensor] = None
                ) -> Tuple[SelectionSpec, Dict[str, torch.Tensor]]:
        """
        Runs the forward method for the pseudo markov process
        :param tensor: The tensor to build local influences from. Shape (..., d_model)
        :param state: A tensor representing the probabilities associated with each virtual layer (bank).
              This state is initialized as a softmax distribution over the banks, with the first
              bank receiving full probability (set to 1) on the first pass. The state biases
              the current selection towards related virtual layers based on past transitions.
              Shape (..., bank_size).
        :return:
            - The SelectionSpec
            - The dict with state in it
        """

        # Standardize data. If state was never setup,
        # it becomes a probability distribution set to
        # 1 at the first element
        if state is None:
            state = torch.zeros([*tensor.shape[:-1], self.bank_size], device=tensor.device, dtype=tensor.dtype)
            state[..., 0] = 1.0

        # Compute the logits. Then combine them with the transition biases.
        # Some options will likely become impossible due to the transition
        # biases

        logits = self.projector(tensor)
        logits = logits + self.markov_biases(state)

        # Activate and store the state. Get the selection. Return results
        state = {"state": torch.softmax(logits, dim=-1)}
        return self.select_logits(logits), state
