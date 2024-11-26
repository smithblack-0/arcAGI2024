import torch
x = torch.arange(1., 20)
x = x.unfold(0,4,2)
print(x)

