import torch
import torch.nn.functional as F

# Define input tensor (batch size = 2, input features = 3)
input_tensor = torch.randn(2, 3)

# Define batched weight tensor (batch size = 2, output features = 4, input features = 3)
# Each batch has its own weight matrix
batched_weights = torch.randn(2, 4, 3)

# Define batched bias tensor (batch size = 2, output features = 4)
# Each batch has its own bias
batched_biases = torch.randn(2, 4)

# Try applying F.linear directly with batched weights and biases
output = F.linear(input_tensor, batched_weights, batched_biases)

#
