import torch


data = torch.randn([10, 4])
index = torch.randint(0, 9, [20, 3, 4])
print(data[index, :].shape) # Yields 20, 3, 4, 4