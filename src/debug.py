import torch

# Create a leaf tensor without gradients
x = torch.tensor([1.0, 2.0, 3.0])

# Set it to retain gradients
x.requires_grad_(True)

# Perform some operations
y = x * 2
z = y.mean()

# Backward pass
z.backward()

# Access the gradients
print(x.grad)  # Output: tensor([0.6667, 0.6667, 0.6667])
