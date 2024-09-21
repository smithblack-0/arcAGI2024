import torch

shape1 = (4, 1)
shape2 = (3, 4, 2)

try:
    result_shape = torch.broadcast_shapes(shape1, shape2)
    print(f"Shapes can be broadcast together. Resulting shape: {result_shape}")
except RuntimeError as e:
    print(f"Cannot broadcast shapes: {e}")
