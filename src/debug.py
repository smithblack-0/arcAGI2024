from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union, Callable, Tuple
import os
import json
import torch
import torch._jit_internal as _jit_internal
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
class AbstractMemoryConfig(SavableConfig):
    """
    The abstract specification for the memory
    config contains a few decent defaults.
    It can be implemented to produce configs more
    specific to a particular memory flavor.

    ** Interpolation factor config ***

    The write factor used to commit updates into memories has a lot
    of math associated with it. They do the the following.

    max_interpolation_factor: The maximum probabilty that can be written in a single step.
                              This is needed in order to prevent division by zero. Set to
                              0.999 as default, but lower might help with numeric stability

    The following two control how the write interpolation rates are initialized. Those factors
    are initialized uniformly between these, and can then be trained. These are basically
    decay factors between 0 and 1, that control how fast the running interpolation decays away
    when the model chooses to write to the memory in every step.

    min_write_half_life_init: Controls the minimum half life that the write interpolation rates
                              can be set on initialization. Must be >= 0.
    max_write_half_life_init: Controls the maximum half life that the write interpolation rates
                              can be set on initialization. Must be > min_write_half_life_init.

    """

    @property
    @abstractmethod
    def interpolation_factor_shapes(self) -> torch.Size:
        # Must return something that broadcasts with the write
        # probabilities.
        raise NotImplementedError("Need to implement interpolation shapes.")


@dataclass
class BankMemoryConfig(AbstractMemoryConfig):
    """
    Specifies the memory configuration for
    a bank memory state. This will include
    things like num memories

    Detail on what each config entry does follows

    **Attn memories**

    The attention memories are a vector of num_memories x d_memory
    blocks where:

    d_memory: The width of each memory unit
    num_memories: The number of memories
    num_read_heads: Number of latent vectors to create out of the input to transfer out of the memory.
                    We then bind the memories onto it
    num_write_heads: Number of latent vectors to create out of the input to transfer into the memory.
                     We then bind the memories onto it.
    write_dropout_factor: A specialized dropout piece. It drops out attempts to write to some
                          memory location. Default is 0.1
    linear_kernel_activation: The kernel activation function for the linear attention mechanism
                              Attention uses linear kernel attention, which requires an activation.


    ** Abstract config ***
    These generally have pretty good defaults, but can be modified if needed.
    The write factor used to commit updates into memories has a lot
    of math associated with it. They do the following.

    max_interpolation_factor: The maximum probabilty that can be written in a single step.
                              This is needed in order to prevent division by zero. Set to
                              0.999 as default, but lower might help with numeric stability

    The following two control how the write interpolation rates are initialized. Those factors
    are initialized uniformly between these, and can then be trained. These are basically
    decay factors between 0 and 1, that control how fast the running interpolation decays away
    when the model chooses to write to the memory in every step.

    min_write_half_life_init: Controls the minimum half life that the write interpolation rates
                              can be set on initialization. Must be >= 0.
    max_write_half_life_init: Controls the maximum half life that the write interpolation rates
                              can be set on initialization. Must be > min_write_half_life_init.
    """

    @property
    def interpolation_factor_shapes(self) -> torch.Size:
        return torch.Size([self.num_memories, self.d_memory])

    d_memory: int
    num_memories: int
    num_read_heads: int
    num_write_heads: int
    write_dropout_factor: float = 0.01
    linear_kernel_activation: Callable[[torch.Tensor], torch.Tensor] = torch.relu

_rcb = _jit_internal.createResolutionCallbackFromFrame()
script = torch.jit.script(BankMemoryConfig, _rcb=_rcb)

