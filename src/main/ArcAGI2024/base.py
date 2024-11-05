from typing import Union, List, Tuple, Dict, Any, Optional, Callable
from abc import ABC, abstractmethod
from tokenizers import Tokenizer

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

# Types
TensorTree = Union[
    torch.Tensor,  # Base case: Tensor
    List['TensorTree'],  # Recursion for lists
    Tuple['TensorTree', ...],  # Recursion for tuples
    Dict[str, 'TensorTree']  # Recursion for dictionaries
]



# Core layers
class StatefulCore(nn.Module, ABC):
    """
    Any class which is going to involve managing
    state in this project is basically going to
    need to implement this.
    """
    @abstractmethod
    def setup_state(self, tensor: torch.Tensor)->TensorTree:
        """
        Sets up state based on the provided tensor of embeddings. Note that
        if you do not use state, just return an empty dict.

        :param tensor: The tensor of embeddings
        :return: Whatever state we need. Can be none.
        """
    @abstractmethod
    def forward(self,
                embedding: torch.Tensor,
                states: TensorTree,
                *parameters: Any)->Tuple[torch.Tensor, TensorTree]:
        """
        Performs the forward pass. Tensor is a tensor of embeddings, while states is any
        state information that needs to be tracked.
        :param tensor: The embedding we are processing
        :param states: The states, if any, associated with the embedding
        :param parameters: Any additional parameters we might need
        :return:
        """
        pass

def get_rng_state(device: torch.device):
    if device.type == "cpu":
        return torch.get_rng_state()
    elif device.type == "cuda":
        return torch.cuda.get_rng_state(device)
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
    else:
        raise ValueError("Unsupported device type. Must be 'cpu' or 'cuda'.")


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


class SavableState(ABC):
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
    def load_state(self, pytree: TensorTree, bypass: Any) -> 'SavableState':
        """
        The reverse of the above. Loads the state from
        a given pytree.
        :param pytree: The pytree to load from
        :param bypass: The bypass to load, if any
        :return: A setup instance.
        """

def parallel_pytree_map(func: Callable[..., Any], *pytrees: Any) -> Any:
    """
    Recursively applies a function to corresponding leaves of multiple pytrees with the same structure.
    Nodes where the function returns None are dropped.

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
        return [elem for elem in result if elem is not None]  # Remove none results
    elif all(isinstance(pytree, tuple) for pytree in pytrees):
        result = tuple(parallel_pytree_map(func, *elems) for elems in zip(*pytrees))
        return tuple(elem for elem in result if elem is not None)  # Remove none results
    elif all(isinstance(pytree, dict) for pytree in pytrees):
        result = {key: parallel_pytree_map(func, *(pytree[key] for pytree in pytrees))
                  for key in pytrees[0]}
        return {key: value for key, value in result.items() if value is not None}  # Remove none results.
    elif all(isinstance(pytree, SavableState) for pytree in pytrees):
        # Convert to pytrees, and update state
        template = pytrees[0]
        _, bypass = template.save_state()
        updated_state = parallel_pytree_map(func, *(state.save_state()[0] for state in pytrees))
        return template.load_state(updated_state, bypass)
    else:
        # These are leaves, apply the function to them
        return func(*pytrees)


