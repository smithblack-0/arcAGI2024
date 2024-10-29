from typing import Union, List, Tuple, Dict, Any, Optional, Callable
from abc import ABC, abstractmethod

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

class CheckpointWithSeed(nn.Module):
    """
    A specialized checkpoint unit designed to keep
    seeds consistent through forward and backwards
    passes so that dropout and such remains sane.
    """

    Seed = Tuple[int, str, Optional[int]]
    def get_seed(self, device: torch.device) -> Seed:
        """
        Get the seed associated with the device a tensor is on.
        :param device: The device to get the seed from.
        :return: A tuple containing the seed, the device type ('cpu' or 'cuda'),
                 and the device number if applicable (for 'cuda', otherwise None).
        """
        # If the tensor is on CPU
        if device.type == 'cpu':
            # Return the CPU seed
            return torch.seed(), 'cpu', None

        # If the tensor is on CUDA
        elif device.type == 'cuda':
            # Get the current CUDA device index
            device_num = device.index
            # Return the CUDA seed and the device number
            return torch.cuda.seed(), 'cuda', device_num

        # Fallback in case a device type is unsupported (shouldn't happen)
        else:
            raise RuntimeError(f"Unsupported device type: {device.type}")


    def set_seed(self, seed: Seed):
        """
        Set the RNG seed for the given device context (CPU or CUDA).
        :param seed: The seed to set for the device.
        :param device_type: The type of the device ('cpu' or 'cuda').
        :param device_num: The specific CUDA device number (required for CUDA devices, otherwise None).
        """
        seed, device_type, device_num = seed
        if device_type == 'cuda':
            if device_num is not None:
                # Ensure the correct CUDA device is set
                torch.cuda.set_device(device_num)
            # Set the CUDA seed for the given device
            torch.cuda.manual_seed(seed)
        elif device_type == 'cpu':
            # Set the seed for the CPU
            torch.manual_seed(seed)
        else:
            raise RuntimeError(f"Unsupported device type: {device_type}")

    def __init__(self, p=0.5):
        super(CheckpointDropout, self).__init__()
        self.p = p


    def run_checkpoint(self,
                       func: Callable,
                       seed: Seed,
                       device: torch.device,
                       *args,
                       **kwargs
                       )->Any:
        """
        Runs the checkpoint, while setting or restoring seeds
        :param func: Function to run
        :param seed: The seed we cached
        :param args: Any function args
        :param kwargs: Any function kwargs
        :return: Whatever was returned
        """
        original_seed = self.get_seed(device)
        self.set_seed(seed)
        output = func(*args, **kwargs)
        self.set_seed(original_seed)
        return output
    def forward(self,
                func: Callable,
                device: torch.device,
                *args,
                **kwargs
                )->Any:
        """
        Performs the forward checkpoint. Caches the
        seed, and restores it on the backwards pass
        pattern that was used.
        :param tensor: The tensor to dropout
        :return: The tensor with dropout applied.
        """

        seed = self.get_seed(device)
        return checkpoint(self.run_checkpoint, func, seed, device, *args, **kwargs)


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


