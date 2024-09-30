import torch


# Define the function without using as_tuple=True
def masked_select_gather_scatter(x: torch.Tensor) -> torch.Tensor:
    # Find indices where x is less than zero
    negative_indices = torch.nonzero(x < 0)

    # Gather values at those indices
    gathered = x[negative_indices[:, 0], negative_indices[:, 1]]

    # Set all negative values to zero
    gathered = torch.zeros_like(gathered)

    # Scatter back the zeroed values to their original positions
    x[negative_indices[:, 0], negative_indices[:, 1]] = gathered

    return x


# Create a test tensor
test_tensor = torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])

# Try scripting the function
scripted_fn = torch.jit.script(masked_select_gather_scatter)

# Try tracing the function
traced_fn = torch.jit.trace(masked_select_gather_scatter, (test_tensor,))

# Apply both functions to the original test tensor
scripted_result = scripted_fn(test_tensor.clone())
traced_result = traced_fn(test_tensor.clone())

print("Original Scripted result:", scripted_result)
print("Original Traced result:", traced_result)

# Now, apply both functions to a different test tensor
new_test_tensor = torch.tensor([[10.0, -5.0, 3.0], [-1.0, 7.0, -3.0]])

# Test with the scripted function
new_scripted_result = scripted_fn(new_test_tensor.clone())

# Test with the traced function
new_traced_result = traced_fn(new_test_tensor.clone())

print("New Scripted result:", new_scripted_result)
print("New Traced result:", new_traced_result)
