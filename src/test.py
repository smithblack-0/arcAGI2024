import torch

test = torch.randn([10], requires_grad=True)
test2 = test + 0
test2.retain_grad()

intermediary_with_mask = torch.where(torch.rand([10]) > 0.5, test+0, test)
final = intermediary_with_mask + 0.0
final.retain_grad()
test.retain_grad()

final.sum().backward(retain_graph=True)
print(test.grad)
final.sum().backward(retain_graph=True)
print(final.grad)