import torch


def expand_last_dim(tensor, new_shape):
    """
    Expands the last dimension of the tensor into the specified new_shape.

    Args:
    tensor (torch.Tensor): The input tensor.
    new_shape (list or tuple): The target shape for the last dimension.

    Returns:
    torch.Tensor: The reshaped tensor.
    """
    # Get the original shape of the tensor
    original_shape = tensor.shape

    # Ensure that the last dimension size is compatible with the new_shape
    last_dim_size = original_shape[-1]
    if last_dim_size != torch.prod(torch.tensor(new_shape)):
        raise ValueError(
            f"The last dimension size ({last_dim_size}) is not compatible with the target shape {new_shape}.")

    # Calculate the new shape
    new_tensor_shape = original_shape[:-1] + tuple(new_shape)

    # Reshape the tensor
    reshaped_tensor = tensor.view(*new_tensor_shape)

    return reshaped_tensor


# Example usage
tensor = torch.randn(10, 20)
new_shape_1 = [4, 5]
new_shape_2 = [2, 2, 5]

# Expand the last dimension into the new shape
expanded_tensor_1 = expand_last_dim(tensor, new_shape_1)
expanded_tensor_2 = expand_last_dim(tensor, new_shape_2)

print(expanded_tensor_1.shape)  # Output: torch.Size([10, 4, 5])
print(expanded_tensor_2.shape)  # Output: torch.Size([10, 2, 2, 5])
