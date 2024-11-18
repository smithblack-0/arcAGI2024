import torch

# Example tensor
tensor = torch.tensor([1.0, 2.0, 3.0], device="cuda:0", dtype=torch.float32)

# Get the device as a string
device_str = str(tensor.device)

print(f"Device string: {device_str}")

# Example: Use the string to initialize a new tensor
new_tensor = torch.tensor([4.0, 5.0, 6.0], device=device_str)
print(f"New tensor is on: {new_tensor.device}")