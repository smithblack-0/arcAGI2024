import torch
from typing import Sequence, Tuple

from .channel_bound_spec import CBTensorSpec
from .channel_bound_tensors import CBTensor


def rationalize_dim(dim: int, max_dim: int)->int:
    """
    Rationalizes dims to be within the given
    length, allowing sane indexing even with
    hidden channel behavior.

    :param dim: The dimension to rationalize
    :param max_dim: The maximum dimension allowed
    :return: The rationalized dimension
    """

    # If dim is greater than the number of dimensions (excluding the channel dimension), raise an error
    if dim > max_dim:
        raise ValueError(f"Attempt to unsqueeze non-existent dim {dim}. Tensor has {max_dim} dimensions.")

    # Handle negative dimensions by wrapping them to positive dimensions
    if dim < 0:
        dim -= 1

    return dim
@CBTensor.register_operator(torch.cat)
def cat(tensors: Sequence['CBTensor'], dim: int = 0) -> 'CBTensor':
    """
    Concatenates collections of CBTensors together. Is not capable
    of modifying the channel dimension. Pretends the channel dimension
    does not exist.

    ---- Parameters ----
    :param tensors:
        - A sequence of CBTensors to concatenate together.
        - All tensors must have the same channel specification and the same number of dimensions.

    :param dim:
        - The dimension along which to concatenate.
        - If `dim` is negative, adjusts to ignore the channel dimension.

    ---- Returns ----
    :return:
        - A new `CBTensor` resulting from concatenating the tensors along the specified dimension.

    ---- Raises ----
    - `ValueError`: If tensors are empty, have different specs, different dimensions, or an invalid dimension is targeted.
    """

    if len(tensors) == 0:
        raise ValueError("Cannot concatenate an empty list of tensors.")

    dim_num = tensors[0].get_tensor().dim()  # Number of dimensions excluding the channel dimension
    if any(tensor.get_tensor().dim() != dim_num for tensor in tensors[1:]):
        raise ValueError(f"Cannot concatenate tensors with different numbers of dimensions.")

    spec = tensors[0].spec
    if any(tensor.spec != spec for tensor in tensors[1:]):
        raise ValueError("Cannot concatenate tensors with different specs.")

    if dim >= dim_num:
        raise ValueError(f"Target dimension does not exist. Number of dimensions: {dim_num}, target: {dim}.")

    if dim < 0:
        # Adjust to not point at the channel dimension
        dim -= 1

    # Concatenate the tensors along the modified dim
    output_tensor = torch.cat([cb_tensor.get_tensor() for cb_tensor in tensors], dim=dim)
    return CBTensor(spec, output_tensor)

@CBTensor.register_operator(torch.eq)
def eq(tensor_a: CBTensor, tensor_b: CBTensor) -> torch.Tensor:
    """
    Checks if the channels along two CBTensors are element-wise equal.
    The result is a boolean tensor representing element-wise comparison over the
    common dimensions (excluding the channel dimension).

    :param tensor_a: The first CBTensor to compare.
    :param tensor_b: The second CBTensor to compare.
    :return: A boolean tensor indicating element-wise equality.
    """

    if tensor_a.spec != tensor_b.spec:
        raise ValueError(f"Tensors must have the same specs to perform element-wise comparison.")

    # Perform the equality check on the underlying tensors, then apply all-reduction
    tensor_output = torch.eq(tensor_a.tensor, tensor_b.tensor)

    # Perform all-reduction along the channel dimension (last dimension) to hide it
    return torch.all(tensor_output, dim=-1)
@CBTensor.register_operator(torch.unsqueeze)
def unsqueeze(tensor: CBTensor, dim: int = 0) -> CBTensor:
    """
    Unsqueezes a particular dimension by adding a new singleton dimension at the specified index.

    ---- Parameters ----
    :param tensor:
        - A `CBTensor` object representing the original tensor.
        - The tensor must have a valid shape excluding the final channel dimension.

    :param dim:
        - The dimension where the new singleton (size 1) dimension will be inserted.
        - Can be a positive or negative integer.
        - If negative, it counts dimensions from the end.
        - The dimension must be within the valid range of the tensor's dimensions.

    ---- Returns ----
    :return:
        - A new `CBTensor` with the same channel structure, but with a new dimension of size 1 inserted
          at the specified index.

    ---- Raises ----
    - `ValueError`: If the specified dimension is out of range for the tensor.

    """

    # Rationalize/convert dim.
    dim = rationalize_dim(dim, tensor.dim())

    # Unsqueeze the tensor at the specified dimension (ignoring the channel dimension)
    core = tensor.tensor.unsqueeze(dim)

    # Return a new CBTensor with the same spec and the unsqueezed tensor
    return CBTensor(tensor.spec, core)

@CBTensor.register_operator(torch.reshape)
def reshape(tensor: CBTensor, shape: Tuple[int, ...]) -> CBTensor:
    """
    Applies the reshape operation to the nonchannel dimensions.
    :param tensor: The CBTensor to reshape
    :param shape: The shape to change it into
    :return: The reshaped CBTensor
    """

    core = tensor.get_tensor()
    shape = [*shape] + [tensor.total_channel_width]
    core = torch.reshape(core, shape)
    return CBTensor(tensor.spec, core)

@CBTensor.register_operator(torch.flatten)
def flatten(tensor: CBTensor, start_dim: int=0, end_dim: int=-1) -> CBTensor:
    """
    A CBTensor variant of torch's flatten.

    :param tensor: The CBTensor to flatten
    :param start_dim: The dimension to start flattening at
    :param end_dim: The dimension to end flattening at
    :return: The flattened CB Tensor
    """

    start_dim = rationalize_dim(start_dim, tensor.dim())
    end_dim = rationalize_dim(end_dim, tensor.dim())
    core = torch.flatten(tensor.get_tensor(), start_dim, end_dim)
    return CBTensor(tensor.spec, core)
