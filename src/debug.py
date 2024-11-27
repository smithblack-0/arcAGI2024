import torch

cross_entropy = torch.nn.CrossEntropyLoss()

inputs = torch.randn([3, 5])
distribution_target = torch.softmax(torch.randn([3, 5]), dim=-1)

print(cross_entropy(inputs, targets))