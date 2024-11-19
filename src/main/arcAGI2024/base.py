import os
import shutil
import json
from typing import Union, List, Tuple, Dict, Any, Optional, Callable
from abc import ABC, abstractmethod
from tokenizers import Tokenizer
from dataclasses import dataclass, asdict
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
try:
    import torch_xla
except:
    pass

# Types
TensorTree = Union[
    torch.Tensor,  # Base case: Tensor
    List['TensorTree'],  # Recursion for lists
    Tuple['TensorTree', ...],  # Recursion for tuples
    Dict[str, 'TensorTree']  # Recursion for dictionaries
]

def get_rng_state(device: torch.device):
    if device.type == "cpu":
        return torch.get_rng_state()
    elif device.type == "cuda":
        return torch.cuda.get_rng_state(device)
    elif device.type == "xla":
        return torch_xla.core.xla_model.get_rng_state(device)

    else:
        raise ValueError("Unsupported device type. Must be 'cpu' or 'cuda'.")


def set_rng_state(state, device: torch.device):
    """
    Sets the RNG state associated with a given device
    :param state:
    :param device:
    :return:
    """
    if device.type == 'cpu':
        torch.set_rng_state(state)
    elif device.type == 'cuda':
        torch.cuda.set_rng_state(state, device)
    elif device.type == "xla":
        torch_xla.core.xla_model.set_rng_state(state, device)
    else:
        raise ValueError("Unsupported device type. Must be 'cpu' or 'cuda'.")

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

class DropoutLogits(nn.Module):
    """
    A dropout layer for logits, which applies dropout by masking logits to
    a large negative value (epsilon) to simulate "dropping out" certain logits.
    """

    def __init__(self, p: float = 0.5, epsilon: float = -1e9):
        """
        Initialize the DropoutLogits layer.
        :param p: The probability of dropping out a logit (default is 0.5).
        :param epsilon: The large negative value used for masking (default is -1e9).
        """
        super(DropoutLogits, self).__init__()
        self.p = p
        self.epsilon = epsilon

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply dropout to the logits by masking them to a large negative value (epsilon).
        :param logits: The input logit tensor.
        :return: The logits with dropout applied.
        """
        if self.training:
            # Create a dropout mask (0 = drop, 1 = keep)
            dropout_mask = torch.bernoulli(torch.full_like(logits, 1 - self.p))

            # Apply the dropout mask, setting dropped logits to epsilon
            logits = torch.where(dropout_mask == 1, logits, torch.full_like(logits, self.epsilon))

        return logits

class DeviceDtypeWatch(nn.Module):
    """
    Initialized with, and subsequently watches,
    the device and dtype. Responds properly
    when .to is invoked

    Should be used to store and lookup
    device and dtype information.
    """
    @property
    def device(self)->torch.device:
        return self.watch.device

    @property
    def dtype(self)->torch.dtype:
        return self.watch.dtype

    @device.setter
    def device(self, value: torch.device):
        self.watch.to(device=value)

    @dtype.setter
    def dtype(self, value: torch.dtype):
        self.watch.to(dtype=value)

    def __init__(self,
                 device: Optional[torch.device] = None,
                 dtype: Optional[torch.dtype] = None,
                 ):


        # Standardize
        if device is None:
            device = torch.device('cpu')
        if dtype is None:
            dtype = torch.float32
        super().__init__()

        # Setup watch buffer. This is completely
        # empty, but will automatically respond
        # when .to is used
        watch = torch.empty([0], dtype=dtype, device=device)
        self.register_buffer("watch", watch)




class PytreeState(ABC):
    """
    A abstract state that implemented two methods,
    designed to turn the state into something
    that can be saved and loaded as a pytree.

    This makes it compatible with some otherwise
    impossible to use functions, such as parallel
    pytree map.

    Note that the first instance will be kept
    around and used to initialize the subsequent
    state. Nontensor features, such as a common threshold,
    can be initializes using the class.

    It also promises to have statistics and metrics
    """

    @abstractmethod
    def save_state(self) -> Tuple[TensorTree, Optional[Any]]:
        """
        Saves the state feature as a tensortree, making it
        accessable to many classes and features
        :return:
            - The tensortree containing the state to be processed
            - The bypass features.
        """

    @abstractmethod
    def load_state(self, pytree: TensorTree, bypass: Any) -> 'PytreeState':
        """
        The reverse of the above. Loads the state from
        a given pytree.
        :param pytree: The pytree to load from
        :param bypass: The bypass to load, if any
        :return: A setup instance.
        """

@dataclass
class SavableConfig(ABC):
    """
    A savable config is a dataclass feature which
    is so named because it can be saved to a folder
    """

    # class name. Usually a shared class feature.
    file_name: str

    # Concrete details.
    #
    # These are some good defaults, but can
    # be overwritten later if needed.

    def _save_to_folder(self, directory: Union[str, os.PathLike]):
        file = os.path.join(directory, self.file_name)
        with open(file, 'w') as f:
            config = asdict(self)
            json.dump(config, f)

    def _load_from_folder(cls, directory: Union[str, os.PathLike]) -> 'AbstractMemoryConfig':
        file = os.path.join(directory, cls.file_name)
        with open(file, 'r') as f:
            config = json.load(f)
        return cls(**config)

    # Actual save/load interface
    def save_to_folder(self, directory: Union[str, os.PathLike]):
        os.makedirs(directory, exist_ok=True)
        self._save_to_folder(directory)

    def load_from_folder(self, directory: Union[str, os.PathLike])->'SavableConfig':
        return self._load_from_folder(directory)


class SavableLayer(nn.Module, ABC):
    """
    A savable layer is a layer that implements
    two methods designed to allow saving and loading
    from folders.
    """


def parallel_pytree_map(func: Callable[..., Any], *pytrees: Any) -> Any:
    """
    Recursively applies a function to corresponding leaves of multiple pytrees with the same structure.

    ---- support ----

    Support is present for the dict, list, tuple, and SavableState classes. See SaveableState
    for more details on that, but it can be mixed in and implemented to let us pytree map things
    that otherwise would not work.

    ---- spec ---

    Args:
        func (Callable[..., Any]): A function to apply to corresponding leaves of the pytrees.
        *pytrees (NestedTensor): Multiple pytrees with the same structure.

    Returns:
        NestedTensor: A new pytree with the function applied to corresponding leaves,
                      excluding those where the function returns None.
    """
    # Check if all pytrees are lists, tuples, or dicts
    if all(isinstance(pytree, list) for pytree in pytrees):
        result = [parallel_pytree_map(func, *elems) for elems in zip(*pytrees)]
        return result
    elif all(isinstance(pytree, tuple) for pytree in pytrees):
        result = tuple(parallel_pytree_map(func, *elems) for elems in zip(*pytrees))
        return result
    elif all(isinstance(pytree, dict) for pytree in pytrees):
        result = {key: parallel_pytree_map(func, *(pytree[key] for pytree in pytrees))
                  for key in pytrees[0]}
        return result
    elif all(isinstance(pytree, PytreeState) for pytree in pytrees):
        # Convert to pytrees, and update state
        template = pytrees[0]
        _, bypass = template.save_state()
        flat_states = [state.save_state()[0] for state in pytrees]
        updated_state = parallel_pytree_map(func, *flat_states)
        return template.load_state(updated_state, bypass)
    else:
        # These are leaves, apply the function to them
        return func(*pytrees)


class GradientSubstitutionEndpoint(torch.autograd.Function):
    """
    Acts as a location in which gradients can be manually
    substituted into the tensor stream.
    """

    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx,
                tensor: torch.Tensor,
                desired_gradients: torch.Tensor) -> torch.Tensor:
        assert tensor.shape == desired_gradients.shape

        # Save the desired gradients for the backward pass
        ctx.save_for_backward(desired_gradients)

        # Return a zero-filled scalar that can connect to the loss
        return torch.tensor(0.0,
                            requires_grad=True,
                            dtype=tensor.dtype,
                            device=tensor.device)

    @staticmethod
    def backward(ctx,
                 grad_outputs: torch.Tensor
                 ) -> torch.Tensor:
        # Load the desired gradients
        desired_gradients, = ctx.saved_tensors

        # Substitute the gradient with the desired gradients.
        # Also, clearly indicate that desired grads is not
        # going to be differentiated further.
        return desired_gradients, None