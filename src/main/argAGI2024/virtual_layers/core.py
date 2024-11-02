import textwrap
from abc import ABC
from typing import Optional, Tuple, Callable
import torch
from torch import nn


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


class SelectionSpec:
    """
    A specification for selecting and weighting different virtual layers
    (previously referred to as "banks") in the context of a virtualized
    argAGI2024 architecture.

    This structure encapsulates both the indices of the virtual layers involved
    in the computation and the associated probabilities or weights that define
    their contributions. It is particularly useful when multiple virtual layers
    are selected and their contributions need to be combined, either through
    superposition or discrete selection.

    ---- Selection Context ----
    The `SelectionSpec` provides context for interpreting data that has banks (virtual layers),
    by specifying which banks are chosen and how they are weighted. The class itself does not
    perform data selection but offers instructions for downstream components (such as
    `VirtualBuffer`, `VirtualParameter`, etc.) to interpret this selection.

    The `SelectionSpec` is separate from the data it describes. Virtual layers (or banks)
    store multiple configurations of data, and the `SelectionSpec` tells the system which
    configurations (banks) to pick from and how to combine them.

    ---- Example: Interpreting Selections ----
    When a selection is applied to virtualized data, the spec provides:
    - Indices of the selected banks.
    - Probabilities or weights for each selected bank, used to combine the results.

    The selected layers can apply over different combinations of batch and data dimensions.
    The actual interpretation—how selections affect the underlying data—depends on the
    implementation of the layer that uses the spec.

    ---- Dense selection ----

    An alternative mode of operation is to operate in a dense selection mode.

    In this mode, selection probabilities across the entire selection domain
    are provided, selection indices can be none, and downstream code does
    a dense, matmul based process to handle superposition.

    ---- Fields ----
    selection_index (torch.Tensor):
        An integer tensor containing the indices of the virtual layers selected
        for the current operation. Each index points to a specific layer in a
        collection of virtual layers.
        - Shape: (..., num_selected_virtual_layers).
        The shape can vary depending on how many dimensions the selection applies to.
        Batch dimensions are explicitly handled, while data dimensions are broadcast as needed.
        - May be none in dense mode, or may be provided.

    selection_probabilities (torch.Tensor):
        A tensor containing the probabilities or weights that determine how
        strongly each selected virtual layer contributes to the final result.
        This is used to calculate a weighted superposition of the selected layers.
        - Shape: (..., num_selected_virtual_layers).
        Like `selection_index`, this tensor's shape can accommodate different dimensions,
        and will broadcast over remaining data dimensions as required.
    """

    @staticmethod
    def is_floating_dtype(dtype: torch.dtype) -> bool:
        """
        Checks whether the given dtype is a floating-point type.
        Floating dtypes are required for selection probabilities to
        ensure valid interpolation or superposition behavior.

        :param dtype: The dtype to check.
        :return: True if the dtype is floating-point, False otherwise.
        """
        return dtype in {torch.float16, torch.float32, torch.float64}

    @property
    def dtype(self) -> torch.dtype:
        """
        Returns the dtype of the `selection_probabilities` tensor.
        """
        return self.selection_probabilities.dtype

    @property
    def device(self) -> torch.device:
        """
        Returns the device of the `selection_probabilities` tensor.
        """
        return self.selection_probabilities.device

    def __init__(self,
                 selection_index: Optional[torch.Tensor],
                 selection_probabilities: torch.Tensor,
                 dense_mode: bool = False,
                 ):
        """
        Initializes a `SelectionSpec` object.

        ---- Parameters ----
        :param selection_index: A tensor containing the indices of the selected virtual layers.
                                Must be of dtype `torch.long`.
        :param selection_probabilities: A floating-point tensor with probabilities or weights
                                        corresponding to the selected indices. Or across the
                                        entire selection space if in dense mode
        :param dense_mode: Whether the selection space is dense or sparse. If sparse,
                           selection index will be required, while if dense it will not.

        ---- Raises ----
        - ValueError: If the tensors are on different devices or their shapes do not match.
        - TypeError: If the selection indices are not of dtype `torch.long` or the probabilities
                     are not floating-point tensors.
        """
        # Validation
        if not dense_mode:
            if selection_index is None:
                msg = f"""
                'selection_index' may only be none when 'dense_mode' is true
                """
                msg = textwrap.dedent(msg)
                raise ValueError(msg)

            if selection_index.device != selection_probabilities.device:
                msg = f"""
                'selection_index' was on device {selection_index.device} while
                'selection_probabilities' was on device {selection_probabilities.device}.
                Must be same device
                """
                msg = textwrap.dedent(msg)
                raise ValueError(msg)
            if selection_index.dtype != torch.long:
                msg = f"""
                'selection_index' must be of dtype long, got {selection_index.dtype}
                """
                msg = textwrap.dedent(msg)
                raise TypeError(msg)

            if selection_index.shape != selection_probabilities.shape:
                msg = f"""
                'selection_probabilities" had shape {selection_probabilities.shape},
                while 'selection_index' had shape {selection_index.shape}. Must
                be the same
                """
                msg = textwrap.dedent(msg)
                raise ValueError(msg)

        if not torch.is_floating_point(selection_probabilities):
            msg = f"""
            'selection_probabilities' must be a floating point tensor. Instead,
            got tensor of type {selection_probabilities.dtype}
            """
            msg = textwrap.dedent(msg)
            raise TypeError(msg)


        # Storage
        self.selection_index = selection_index
        self.selection_probabilities = selection_probabilities
        self.dense_mode = dense_mode

    def to(self,
           dtype: Optional[torch.dtype] = None,
           device: Optional[torch.device] = None) -> 'SelectionSpec':
        """
        Converts the selection spec to exist with the given dtype,
        and on the given device, ensuring compatibility with other
        tensors in the system.

        This method moves both tensors between devices but only
        changes the dtype of the `selection_probabilities` tensor.

        ---- Parameters ----
        :param dtype: The dtype to move to. Must be a floating-point datatype if provided.
                      If left as None, it defaults to the current `dtype` of `selection_probabilities`.
        :param device: The device to move the selection spec to. Defaults to the current device.

        ---- Returns ----
        :return: A new `SelectionSpec` on the specified device and/or with the specified dtype.

        ---- Raises ----
        - TypeError: If a non-floating-point dtype is provided for probabilities.
        """

        # Parameter standardization
        dtype = dtype if dtype is not None else self.dtype
        device = device if device is not None else self.device

        # Validate dtype
        if not self.is_floating_dtype(dtype):
            msg = f"""
            Provided dtype must be of a floating variety. Instead,
            found dtype: {dtype}
            """
            msg = textwrap.dedent(msg)
            raise TypeError(msg)

        # Move tensors: `selection_index` stays as long, `selection_probabilities` migrates
        if self.selection_index is not None:
            index = self.selection_index.to(dtype=torch.long, device=device)
        else:
            index = None

        probabilities = self.selection_probabilities.to(dtype=dtype, device=device)

        return SelectionSpec(index, probabilities)


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

    ---- Dense mode ----

    When responding to dense mode, the behavior changes a bit. First, when
    superposition is True, we collapse everything using a single matmul
    or similar process.

    Meanwhile, superposition is False will, on account of being dense,
    just return the same state tensor that is passed in, mirroring how
    sparse mode works.

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
        - dense_mode: Whether operating in sparse or dense mode.

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
    dense_mode = selection.dense_mode

    # If dense is true AND superposition is false, we just return what is given
    # so that the user can run a parallel ensemble. This saves a bit of compute time.
    if not superposition and dense_mode:
        return state



    # Move the selection dim to the end of the state tensor
    state = state.movedim(dim, -1)  # (...batch, ..., options)

    # dense/sparse pathway branching occurs. In dense branch,
    # we ignore the indexes, expand the probabilites to match,
    # then perform the superposition.
    #
    # The main difference between the pathways is one will
    # perform an index select, while another will not. But
    # both will end with something that needs a matmul.
    if dense_mode:
        # Ensure probabilities has the right number
        # of dimensions for broadcast
        assert probabilities.shape[-1] == state.shape[-1]
        while probabilities.dim() < state.dim():
            probabilities = probabilities.unsqueeze(-2) # (..., ...., options)
        selected_state = state
    else:
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
        # Unsqueeze the last dimension of probabilities to make it a column vector
        # This prepares it for matrix multiplication with selected_state
        probabilities = probabilities.unsqueeze(-1)
        selected_state = selected_state.unsqueeze(-2)
        selected_state = torch.matmul(selected_state, probabilities).squeeze(-1).squeeze(-1)
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
        - Dense mode operates a little bit differently. selection_index is not used.

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

    - **Dense vs sparse**: Depending on how the selection spec is configured,
                           this can operate either in a sparse or a dense
                           mode.

                           in sparse mode, we expect the selection spec, and
                           the substate, to be based on advanced index extractions
                           of state.

                           In dense mode, we expect them to be exactly the same
                           state.

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
    dense_mode = selection.dense_mode

    # Move the selection dim to the end of the state tensor for easier processing
    state = state.movedim(dim, -1)  # (...batch, ..., options)
    substate = substate.movedim(dim, -1)  # (...batch, ..., selections)

    # Branch case for sparse vs dense behavior. The branches are
    # VERY different here.
    #
    # In the sparse case, we gather, interpolate,
    # then scatter the results.
    #
    # The dense case, meanwhile, can just proceed directly to interpolation
    # without having to scatter back into it.
    if dense_mode:
        # Unsqueeze the probabilities to have sufficient dimensions for interpolation
        assert probabilities.shape[-1] == state.shape[-1]
        while probabilities.dim() < state.dim():
            probabilities = probabilities.unsqueeze(-2)

        # Perform interpolation between initial and new state based on selection
        # probabilities

        state = state*(1-probabilities) + substate*probabilities
    else:
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
    making it suitable for models with virtual layers.

    It can either be initialized directly with a parameter — in which case
    the last dimension becomes the bank dimension — or one can use the
    `.create` method to make a parameter according to a given specification.

    The class supports automatic dtype conversion of the selection probabilities
    in the `SelectionSpec` if `allow_dynamic_type_conversion` is enabled. This allows
    the `SelectionSpec` to adapt its dtype to match the parameter's dtype.

    ---- dense mode  vs sparse mode ----

    Additionally, this automatically supports dense and sparse modes in the mode
    select mechanism. In dense mode, we use a matrix multiplication to reduce
    the entire thing, while in sparse mode we use vector indexing

    """

    @classmethod
    def create(cls,
               bank_size: int,
               shape: Tuple[int, ...],
               device: Optional[torch.device] = None,
               dtype: Optional[torch.dtype] = None,
               init: Optional[Callable[[torch.Tensor], None]] = None,
               allow_dynamic_type_conversion: bool = False
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
        :param allow_dynamic_type_conversion: Flag to enable automatic dtype conversion
                                              for `SelectionSpec` if the dtypes do not match.
                                              Defaults to `False`.
        :return: An instance of the `VirtualParameter` class.
        """

        parameter = torch.zeros(tuple([*shape, bank_size]), device=device,
                                dtype=dtype)
        if init is not None:
            init(parameter)
        return cls(parameter, allow_dynamic_type_conversion=allow_dynamic_type_conversion)

    def __init__(self,
                 parameter: nn.Parameter,
                 allow_dynamic_type_conversion: bool = False):
        """
        Initializes a `VirtualParameter` with a bank of parameters.

        :param parameter: The parameter to initialize. The last dimension
                          will always be the bank dimension.
        :param allow_dynamic_type_conversion: Flag to enable automatic dtype conversion
                                              for `SelectionSpec` if the dtypes do not match.
                                              Defaults to `False`.
        """
        assert parameter.dim() > 0, "No bank dimension was specified."

        super().__init__()
        self.bank_size = parameter.shape[-1]
        self.shape = parameter.shape[:-1]
        self.device = parameter.device
        self.dtype = parameter.dtype
        self.allow_dynamic_type_conversion = allow_dynamic_type_conversion
        self.parameter = nn.Parameter(parameter)

    def forward(self,
                bank_spec: SelectionSpec,
                ) -> torch.Tensor:
        """
        Returns a superposition of parameters based on the provided
        `bank_spec`.

        The `bank_spec` provides both the selected bank indices and their
        corresponding probabilities. The selected parameters are interpolated
        according to these probabilities to form a final output.

        If `allow_dynamic_type_conversion` is set to True, the `SelectionSpec`'s
        dtype will be automatically converted to match the parameter's dtype.
        If set to False, a mismatch will raise a `TypeError`.

        :param bank_spec: A `SelectionSpec` dataclass containing:
            - `selection_index`: The indices of the selected banks. (..., selected)
            - `selection_probabilities`: The probabilities for weighting
                                         the selected banks. (..., selected)
        :return: A tensor containing the weighted combination of selected
                 parameters. Shape: (..., *shape).
        """
        with torch.profiler.record_function("MakingVirtualParameter"):
            # Automatically convert the dtype of the selection spec if allowed
            if self.allow_dynamic_type_conversion and bank_spec.dtype != self.dtype:
                bank_spec = bank_spec.to(dtype=self.dtype)
            elif bank_spec.dtype != self.dtype:
                msg = f"""
                SelectionSpec dtype {bank_spec.dtype} does not match 
                parameter dtype {self.dtype}. Enable dynamic conversion
                by setting allow_dynamic_type_conversion=True.
                """
                msg = textwrap.dedent(msg)
                raise TypeError(msg)

            # Dense vs sparse branching. In dense mode, we combine everything
            # using matrix multiplication, while in sparse mode we instead
            # extract and combine using vector indexing.
            if bank_spec.dense_mode:
                assert bank_spec.selection_probabilities.shape[-1] == self.parameter.shape[-1]
                probabilities = bank_spec.selection_probabilities
                parameter = self.parameter  # (*shape, bank)

                # Add sufficient probabilities dimensions we can matmul over
                # parameter pieces
                for _ in range(len(self.shape)):
                    probabilities = probabilities.unsqueeze(-2)

                # Setup, run, squeeze matmul
                probabilities = probabilities.unsqueeze(-2)
                parameter = parameter.unsqueeze(-1)
                parameter = torch.matmul(probabilities, parameter).squeeze(-1).squeeze(-1)
            else:
                # Extract the selected banks using advanced indexing
                parameter = self.parameter.movedim(-1, 0)  # (bank, *shape)
                parameter = parameter[bank_spec.selection_index]  # (..., bank_selected, *shape)
                parameter = parameter.movedim(-self.parameter.dim(), -1)


                # Add sufficient probabilities dimensions we can matmul over
                # parameter pieces
                probabilities = bank_spec.selection_probabilities
                for _ in range(len(self.shape)):
                    probabilities = probabilities.unsqueeze(-2)
                parameter = torch.sum(probabilities*parameter, dim=-1)
            return parameter
class VirtualLayer(nn.Module, ABC):
    """
    A layer that promises to implement a virtual
    layer mechanism. This means some internal features
    are stored in virtual parameters, and it also
    possesses a contract on how to pass selection in.

    Kernels in a virtual layer should automatically
    expand to account for the fact that different batches
    may be placed in different superpositions. Fortunately
    VirtualParameter can already do this, and you can code
    everything else pretty much like normal.
    """

    def __init__(self, bank_size):
        super().__init__()
        self.bank_size = bank_size
