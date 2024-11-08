import torch



# Example usage in a training loop
tensor_to_process = torch.randn(5, requires_grad=True)
desired_grads = torch.ones_like(tensor_to_process)  # Replace with actual desired gradients
tensor_to_process.register_hook(lambda x : print(desired_grads.equal(x)))

# Use the function in your graph
loss_endpoint = GradientSubstitutionEndpoint.apply(tensor_to_process, desired_grads)
loss = 1.0 + loss_endpoint  # You can add this to any loss function

loss.backward()


