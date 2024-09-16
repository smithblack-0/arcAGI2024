import torch

# Initialize input tensor (destination)
input = torch.zeros(3, 5)

# Define index tensor for one dimension
index = torch.tensor([0, 1, 2, 0, 1])

# Define source tensor
src = torch.tensor([[10, 20, 30, 40, 50],
                    [60, 70, 80, 90, 100],
                    [110, 120, 130, 140, 150]]).float()

# Broadcast the index to match the shape of src
index_broadcasted = index.expand(src.shape)

# Scatter using the broadcasted index
output = input.scatter(1, index_broadcasted, src)

print("Broadcasted Index:\n", index_broadcasted)
print("Output:\n", output)
