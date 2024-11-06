import torch
import time
import timeit


def benchmark_operation(tensor, operation, *args, **kwargs):
    # Run a warm-up to avoid any initialization effects
    operation(tensor, *args, **kwargs)

    # Measure the time for the operation
    start_time = time.time()
    operation(tensor, *args, **kwargs)
    end_time = time.time()

    return end_time - start_time


# Set the tensor size and quantile to test
tensor_size = (1000, 1000)
quantile_value = 0.5  # 50th percentile (median)
def middle_quantiles_mask(tensor: torch.Tensor,
                     dim: int,
                     )->torch.Tensor:
    """
    Gets a mask that is inclusive only of the middle two quantiles,
    that matches tensor.
    :param tensor: The tensor to get the quantile mask on.
    :param dim: The dimension to perform quantile sorting on
    :return: The mask. Top and bottme quantiles are false. Middle two are true
    """
    # Get the starting point. Then figure out the top and bottom halfs
    mean = tensor.mean(dim=-1, keepdim=True)
    top_half = tensor >= mean
    bottom_half = tensor < mean

    # Take the mean of the top half, and the bottom half
    first_quartile = (tensor*bottom_half).sum(dim=dim, keepdim=True) / (1e-9 + bottom_half.sum(dim=dim, keepdim=True))
    third_quartile = (tensor*top_half).sum(dim=dim, keepdim=True) / (1e-9 + top_half.sum(dim=dim, keepdim=True))

    # Get everything that is between the first and third quantiles
    output = (tensor >= first_quartile) & (tensor < third_quartile)
    return output

def middle_quantiles_mean(tensor: torch.Tensor, dim: int, keepdims: bool=False)->torch.Tensor:
    """
    Performs a mean with only the middle two quantiles.
    Quite fast. Only about 5x slower than mean itself
    :param tensor: the tensor to perform a middle quantiles mean on
    :param dim: The dimension to perform it on
    :param keepdims: Whether to keep the dimensions
    :return: The mean, using only the middle two quantiles
    """
    selection_mask = middle_quantiles_mask(tensor, dim=dim)
    sum = torch.sum(selection_mask*tensor, dim=dim, keepdim=keepdims)
    normalizer = torch.sum(selection_mask, dim=dim, keepdim=keepdims)
    return sum / normalizer
# Create a random tensor
tensor = torch.rand(tensor_size)


# Benchmark
mean_time = timeit.timeit(lambda : tensor.mean(), number=10)
quantile_time = timeit.timeit(lambda : tensor.quantile(quantile_value), number=10)
top_k = timeit.timeit(lambda : tensor.topk(1000), number=10)
sort = timeit.timeit(lambda : tensor.sort(), number=10)
std = timeit.timeit(lambda : tensor.std(), number=10)
fast_quantiles = timeit.timeit(lambda : middle_quantiles_mean(tensor, dim=-1), number=10)
# Display

print(f"torch.mean execution time: {mean_time:.6f} seconds")
print(f"torch.quantile execution time: {quantile_time:.6f} seconds")
print(f"torch.topk execution time: {top_k:.6f} seconds")
print(f"torch.sort execution time: {sort:.6f} seconds")
print(f"torch.std execution time: {std:.6f} seconds")
print(f"middle_quantiles_mean execution time: {fast_quantiles:.6f} seconds")
