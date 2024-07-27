import torch
from typing import Any, List, Callable

# Type for the linear kernel encoding.
#
# This is hardcoded by the pytorch-fast-transformers library.

LKE = List[torch.Tensor]

def transform_kernel(kernel: LKE,
                     transform: Callable[[torch.Tensor], torch.Tensor]
                     )->LKE:
    """
    Applies a transform to all portions of a given kernel

    :param kernel: The kernel to operate on.
    :param transform: A callable accepting and returning a torch tensor
    :return: The transformed LKE
    """
    output = []
    for tensor in kernel:
        output.append(transform(tensor))
    return output

def add_kernels(*kernels: List[LKE])->LKE:
    """
    Adds together a list of compatible LKE kernels

    :param kernels: A bunch of kernels to add together
    :return: A single kernel resulting from adding up all the tensors
    """
    output = []
    for tensors in zip(*kernels):
        output.append(torch.stack(tensors).sum(0))
    return output

def stack_kernels(*kernels: List[LKE], dim = 0)->LKE:
    """
    stacks together a group of kernels at the indicated axis.

    This increases the dimension of the resulting kernel.

    :param kernels: A bunch of kernels to concat together
    :return: The resulting merged kernel.
    """
    output = []
    for tensors in list(zip(*kernels)):
        output.append(torch.stack(tensors, dim=dim))
    return output

def concat_kernels(*kernels: List[LKE], dim = 0)->LKE:
    """
    Concats kernels together if possible.

    :param kernels: The kernels to concat together
    :param dim: What dimension to concat together on
    :return: The concated kernel
    """
    output = []
    for tensors in zip(*kernels):
        output.append(torch.concat(tensors, dim=dim))
    return output

def split_kernel(kernel: LKE, dim = 0)->List[LKE]:
    """
    Splits an LKE back into lists along a particular dimension.

    :param kernel: The kernel to reference
    :param dim: The dimension to split on
    :return: A list of kernels with reduced dimensions.
    """
    output_data = []
    for tensor in kernel:
        output_data.append(tensor.unbind(dim))
    return list(zip(*output_data))


