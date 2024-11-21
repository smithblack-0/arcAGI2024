"""
This specifies the abstract contract our recurrent memory
system follows, and also implements the interpolation
mixing process.
"""
import functools
import os

import torch
import json
from torch import nn
from torch.autograd import profiler
from typing import Tuple, Dict, Union, Protocol, Type, Any, Optional, List
from dataclasses import dataclass, asdict
from ..base import PytreeState, SavableConfig, DeviceDtypeWatch, TensorTree, parallel_pytree_map
from abc import ABC, abstractmethod


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

    max_interpolation_factor: float
    min_write_half_life_init: float
    max_write_half_life_init: float


MemoryData = Dict[str, torch.Tensor]


class MemoryState(PytreeState):
    """
    A common memory state.

    Memory state is internally divided into
    interpolatable and non interpolatable pytrees.
    Interpolatable pytrees will automatically be
    updated using the write factor, but
    noninterpolatable pytrees have to be manually
    changed once set.
    """

    @property
    def cum_write_mass(self) -> torch.Tensor:
        """
        The cumulative write mass contains the sum of all
        the write factors.
        """
        return self.persistent_state["cum_write_mass"]

    @cum_write_mass.setter
    def cum_write_mass(self, value: torch.Tensor):
        assert value.shape == self.cum_write_mass.shape
        self.persistent_state["cum_write_mass"] = value

    @property
    def timestep(self) -> torch.Tensor:
        """
        The timestep we are currently on, when accounting
        for batch padding masking.
        """
        return self.persistent_state["timestep"]

    @timestep.setter
    def timestep(self, value: torch.Tensor):
        assert value.shape == self.timestep.shape
        self.persistent_state["timestep"] = value

    @property
    def running_distance(self) -> torch.Tensor:
        """
        The unnormalized distance, indicating from the start of the sequence
        where each memory unit will, on average, be reading from.
        """
        return self.persistent_state["running_distance"]

    @property
    def normalized_timestep_distance(self) -> torch.Tensor:
        """
        The normalized timestep distance. Measure the current timestep as
        0, and all the way back at the beginning as one. Indicates how
        far into the past each memory slot is looking
        """
        timestep = self.timestep
        running_distance = self.running_distance
        while timestep.dim() < running_distance.dim():
            running_distance = running_distance.unsqueeze(-1)
        return (timestep - running_distance) / (timestep + 1e-6)

    def __init__(self,
                 persistent_state: MemoryData,
                 interpolation_state: MemoryData,
                 ):
        self.persistent_state = persistent_state
        self.interpolation_state = interpolation_state

    def get_interpolation_states(self) -> MemoryData:
        """
        Gets a relevant memory implementation in the appropriate order
        involving the features that can be interpolated. These will
        later be processed to update in parallel pytree map
        """
        return self.interpolation_state

    def get_persistent_state(self) -> MemoryData:
        """
        Gets relevant parameter state in the appropriate order,
        and involving the indicated names. These features must
        be manually updated if they are updated at all.
        """
        return self.persistent_state

    def replace_interpolation(self,
                              interpolation_state: MemoryData
                              ) -> 'MemoryState':
        """
        Replaces the relevant features of the interpolation state in a single go
        :param interpolation_state: The relevant interpolation state
        :return: The revised memory state
        """
        interpolation_state = {key: interpolation_state[key]
                              if key in interpolation_state
        else self.interpolation_state[key]
                               for key in self.interpolation_state.keys()
                               }
        return MemoryState(self.persistent_state, interpolation_state)

    def update_persistent(self,
                          feature_name: str,
                          update: torch.Tensor,
                          ) -> 'MemoryState':
        """
       Integrates a persistent update into the
       memory state
       :param feature_name: Name of the state to update
       :param update: The update to integrate
       :return: The new memory state
       """
        assert feature_name in self.persistent_state
        new_persistent = self.persistent_state.copy()
        new_persistent[feature_name] = update
        return MemoryState(new_persistent, self.interpolation_state)

    def save_state(self) -> Tuple[MemoryData, MemoryData]:
        return self.get_interpolation_states(), self.get_persistent_state()

    def setup_for_gradients_(self):
        """
        Turns the stored tensors into leafs which accumulate gradients
        """
        for tensor in self.get_interpolation_states().values():
            tensor.detach_()
            tensor.requires_grad_(True)

    @classmethod
    def load_state(cls,
                   pytree: MemoryData,
                   bypass: MemoryData) -> 'MemoryState':
        constructor_kwargs = pytree.copy()
        constructor_kwargs.update(bypass)
        return cls(pytree, bypass)


class AbstractCreateState(nn.Module, ABC):
    """
    Creates a blank memory state based on the
    batch shape. The forward method needs to
    be defined.

    One special note is the user must remember
    to include a "cum_write_mass" term in the
    persistent memory or they will face an error.

    This should be a zeros array shaped like your
    write factors.
    """

    @property
    @torch.jit.export
    def device(self) -> torch.device:
        return self._metainfo.device

    @property
    @torch.jit.export
    def dtype(self) -> torch.dtype:
        return self._metainfo.dtype

    def __init__(self, dtype: torch.dtype, device: torch.device):
        super().__init__()
        self._metainfo = DeviceDtypeWatch(device=device, dtype=dtype)

    @abstractmethod
    def setup_state(self, batch_shape: List[int]) -> MemoryState:
        """
        When implemented, returns a memory object when seeing a batch shape
        :param batch_shape: The batch shape to consider
        :return: The memory state. Remember to include
                 cumulative_write_factors in persistant.
        """

    def forward(self, batch_shape: List[int]) -> MemoryState:
        state = self.setup_state(batch_shape)
        if "cum_write_mass" not in state.persistent_state:
            raise ValueError("Did not provide a cum_write_mass in the persistant state")
        if "timestep" not in state.persistent_state:
            raise ValueError("Did not provide a timestep feature in the persistent state")
        if "running_distance" not in state.interpolation_state:
            raise ValueError("Did not provide running_distance in the interpolation state")
        return state


class AbstractReadMemory(nn.Module, ABC):
    """
    Abstract read memory step. We abstractly
    specify how we will read from a memory unit.
    """

    @property
    def device(self) -> torch.device:
        return self._metainfo.device

    @property
    def dtype(self) -> torch.dtype:
        return self._metainfo.dtype

    def __init__(self,
                 dtype: torch.dtype,
                 device: torch.device,
                 ):
        """
        :param dtype: The dtype
        :param device: The device
        """
        super().__init__()
        self._metainfo = DeviceDtypeWatch(device=device, dtype=dtype)

    @abstractmethod
    def forward(self,
                query: torch.Tensor,
                memory: MemoryState,
                ) -> torch.Tensor:
        """
        Abstract specification of a memory read action
        :param query: The query to read with. Shape (..., d_model). Recurrent, so no items dim
        :param memory: The memory state. Currently abstract
        :return: The memory read. Shape (..., d_model)
        """


@torch.jit.script
def _advance_memory_case(memory_tensor: torch.Tensor,
                         update_tensor: torch.Tensor,
                         write_factor: torch.Tensor,
                         batch_mask: torch.Tensor,
                         ) -> torch.Tensor:
    """
    Performs the advance memory step. As it is highly
    performant code, it has been scripted. Part of
    AbstractWriteMemory. Must be scripted
    separately from its class due to the way torchscript works

    :param memory_tensor: The tensor currently existing in the memory
    :param update_tensor: The proposed update in the memory
    :param write_factor: The write factor to proceed under.
    :param batch_mask: Whether or not the batch is masked. True means do not update
    :return: The next memory tensor
    """
    assert update_tensor.shape == memory_tensor.shape, f"update had shape {update_tensor.shape}, memory {memory_tensor.shape}"
    assert write_factor.dim() <= memory_tensor.dim()
    assert batch_mask.dim() <= memory_tensor.dim()
    while write_factor.dim() < memory_tensor.dim():
        write_factor = write_factor.unsqueeze(-1)
    while batch_mask.dim() < memory_tensor.dim():
        batch_mask = batch_mask.unsqueeze(-1)

    memory_update = memory_tensor * (1 - write_factor) + update_tensor * write_factor
    memory_tensor = torch.where(batch_mask, memory_tensor, memory_update)
    return memory_tensor


@torch.jit.script
def _retard_memory_case(memory_tensor: torch.Tensor,
                        update_tensor: torch.Tensor,
                        write_factor: torch.Tensor,
                        batch_mask: torch.Tensor,
                        ) -> torch.Tensor:
    """
    Retards the memory tensor. This means we go backwards in
    time.

    as this is highly
    performant code, it has been scripted. Part of
    AbstractWriteMemory. Must be scripted
    separately from its class due to the way torchscript works

    For reasons of numeric stability, we use a logarithm when
    doing the division.
    :param memory_tensor: The tensor currently existing in the memory
    :param update_tensor: The proposed update in the memory
    :param write_factor: The write factor to proceed under
    :param batch_mask: Whether or not the batch is masked. True means do not update
    :return: The previous memory tensor
    """
    assert update_tensor.shape == memory_tensor.shape
    assert write_factor.dim() <= memory_tensor.dim()
    assert batch_mask.dim() <= memory_tensor.dim()
    while write_factor.dim() < memory_tensor.dim():
        write_factor = write_factor.unsqueeze(-1)
    while batch_mask.dim() < memory_tensor.dim():
        batch_mask = batch_mask.unsqueeze(-1)

    log_memory = torch.log(update_tensor * write_factor - memory_tensor + 1e-9)
    log_memory -= torch.log(1 - write_factor + 1e-9)  # divides
    memory_update = torch.exp(log_memory)
    memory_tensor = torch.where(batch_mask, memory_tensor, memory_update)
    return memory_tensor


class AbstractWriteMemory(nn.Module, ABC):
    """
    An abstract implementation of the write
    memory process.


    ---- abstract over ----

    The abstract class is responsible for advancing or retarding
    the memory state, and for integrating decay rate information
    into the write probabilities.

    A reversable running interpolation is ued with a certain
    write factor, such that in forward mode, and reverse mode,
    respectively the logic behaves as

    $$ s_{i+1} = s_i*(1-p_{write}) + s_{ut}*p_{write} $$
    $$s_{i} = \frac{s_{ut}*p_{write} - s_{i+1}}{(1-p_{write})}$$

    ---- Step process ----

    A implementation-specific update along with a write probability is provided by the
    user. The write probability is a gate intended to answer whether we want to write
    or not, without accounting too much for strength.

    From this write probability, the following occurs to advance or retard the memory.

    - Multiply the write probability by the interpolation rates. These are based on a sigmoid
      activation of rate logits, and are between 0 and 1. This governs the strength of the
      update. Low interpolation rates are thus good at looking at long term averages.
    - Multiply by the maximum write factor. This is set to something really high,
      like a half life of 100,000, and makes sure we can never actually divide by zero.
    - Interpolate using one of the two processes above to step forward in the memory,
      or alternatively step backwards

    It should be mentioned that for numeric reasons, the division is performed in
    logarithm form during the reverse step.
    """

    @property
    def device(self) -> torch.device:
        return self._metainfo.device

    @property
    def dtype(self) -> torch.dtype:
        return self._metainfo.dtype

    ##
    # Performance zone
    #
    # A large portion of the abstract code will
    # be spending time in these sections.
    #
    # Since torchscript does not understand
    # inheritance or recursion, we restrict
    # scripting to only tensor based functions.
    #
    ##

    def _initialize_rate_logits(self,
                                shape: torch.Size,
                                low_half_life: float,
                                high_half_life: float
                                ) -> torch.Tensor:
        """
        Initialize interpolation logits based on specified half-life ranges.

        The process involves:
        1. Sampling half-life values uniformly between low and high thresholds.
        2. Computing decay factors corresponding to each half-life.
        3. Transforming decay factors into logits using inverse sigmoid.
        4. Applying a squared transformation to logits to ensure stability across a wide range of half-lives.

        This method ensures that the interpolation rates remain numerically stable, especially when handling very
        long half-lives.

        :param shape: The shape to initialize.
        :param low_half_life: The minimum half-life for initialization.
        :param high_half_life: The maximum half-life for initialization.
        :return: Initialized logits for interpolation factors.
        """

        # Setup a tensor uniformly filled with these half lives.
        half_lives = torch.zeros(shape, device=self.device, dtype=self.dtype)
        half_lives = half_lives.uniform_(low_half_life, high_half_life)

        # Compute the associated decay factors
        decay_factors = (0.5) ** (1 / half_lives)

        # Run the inverse sigmoid. This gets the smoothed logits
        smoothed_logits = -torch.log(1 / decay_factors - 1)

        # Run the inverse smoothing, getting the actual logits
        return torch.sign(smoothed_logits) * smoothed_logits.pow(2)

    @staticmethod
    @torch.jit.script
    def _compute_interpolation_factors(interpolation_logits: torch.Tensor) -> torch.Tensor:
        """
        Computes interpolation factors from logits with stability considerations.

        The transformation involves:
        1. Taking the square root of the absolute logits to reduce the growth rate at extreme values.
        2. Preserving the sign of the original logits.
        3. Applying the sigmoid function to map transformed logits to the (0, 1) range.

        This approach mitigates the risk of dead gradients by ensuring that logits do not grow too rapidly in magnitude.

        :param interpolation_logits: The raw interpolation parameters.
        :return: The activated interpolation factors, constrained between 0 and 1.
        """

        # These logits grow slower at more extreme values.
        #
        # This helps prevent dead gradients going through the sigmoid.
        smoothed_logits = interpolation_logits.abs().sqrt()
        smoothed_logits = torch.sign(interpolation_logits) * smoothed_logits

        # Activate and return
        return torch.sigmoid(smoothed_logits)

    @staticmethod
    @torch.jit.script
    def _advance_memory(
            memory_tensors: Dict[str, torch.Tensor],
            update_tensors: Dict[str, torch.Tensor],
            write_factor: torch.Tensor,
            batch_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Advances the memory by interpolating between the memory
        and update tensors.
        :param memory_tensors: A dictionary of the interpolatable memory tensors
        :param update_tensors: A dictionary of the interpolatable update tensors
        :param write_factor: The write factor to use
        :param batch_mask: The batch mask to use
        :return: The final tensors.
        """
        assert memory_tensors.keys() == update_tensors.keys()
        output_tensors: Dict[str, torch.Tensor] = {}
        for name in memory_tensors:
            memory_tensor = memory_tensors[name]
            update_tensor = update_tensors[name]
            final_update = _advance_memory_case(memory_tensor, update_tensor, write_factor, batch_mask)
            output_tensors[name] = final_update
        return output_tensors

    @staticmethod
    @torch.jit.script
    def _retard_memory(
            memory_tensors: Dict[str, torch.Tensor],
            update_tensors: Dict[str, torch.Tensor],
            write_factor: torch.Tensor,
            batch_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Retards the interpolatable portions of the
        memory by reversing the original operation
        :param memory_tensors: The memory tensors in their next state
        :param update_tensors: The update tensors used in the forward step
        :param write_factor: The write factor
        :param batch_mask: The batch mask
        :return:
        """
        assert memory_tensors.keys() == update_tensors.keys()
        output_tensors: Dict[str, torch.Tensor] = {}
        for name in memory_tensors:
            memory_tensor = memory_tensors[name]
            update_tensor = update_tensors[name]
            final_update = _retard_memory_case(memory_tensor, update_tensor, write_factor, batch_mask)
            output_tensors[name] = final_update
        return output_tensors

    def __init__(self,
                 dtype: torch.dtype,
                 device: torch.device,
                 config: AbstractMemoryConfig
                 ):
        """
        :param dtype: The dtype
        :param device: The device.
        :param config: The memory config, or what we need of it anyhow
        """

        super().__init__()
        self._max_interpolation_rate = config.max_interpolation_factor
        self._metainfo = DeviceDtypeWatch(device=device, dtype=dtype)

        interpolation_logits = self._initialize_rate_logits(config.interpolation_factor_shapes,
                                                            config.min_write_half_life_init,
                                                            config.max_write_half_life_init)
        self._interpolation_logits = nn.Parameter(interpolation_logits)

    @abstractmethod
    def _compute_common(self,
                        query: torch.Tensor,
                        persistent_state: Dict[str, torch.Tensor],
                        ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        The main user specified component.

        The user must specify how to process a query into
        a state update and a write probability. Note you
        only have access to the persistent state when making
        these computations, and should only depend on things
        that are initialized ONCE.

        :param query: The query tensor. Shape (..., d_model), presumably
        :param persistent_state: The persistent state of the memory.
        :return:
        - update: An implementation-specific update, consisting of a dict of
                  tensortrees corrolating with the interpolation memory content.
        - write_probability: A write probability that tells us how strongly to write to the memory slots.
            - Must cleanly multiply the interpolation factor shape.
        """

    ## External interface
    #
    # The intention is to call compute common,
    # then your forward and backwards methods as needed.

    def compute_common(self,
                       query: torch.Tensor,
                       memory: MemoryState,
                       ) -> Tuple[MemoryData, torch.Tensor]:
        """
        Internal helper with the additional logic needed
        to get the write factor ready to go based on the write
        probabilities.

        Calls into implementation detail compute_common
        :param query: The query to make the update and factor on
        :param memory: The memory state. The parameter state will be retrieved off it.
        :return: The update state, and the write factor
        """
        with profiler.record_function("Computing write parameters"):
            persistent_state = memory.get_persistent_state()
            with profiler.record_function("Computing write parameters: Implementation"):
                update_state, write_probability = self._compute_common(query, persistent_state)

            # Compute write factor
            interpolation_rates = self._compute_interpolation_factors(self._interpolation_logits)
            interpolation_rates = self._max_interpolation_rate * interpolation_rates
            write_factor = interpolation_rates * write_probability

            # We need to integrate the timestep into the running distance so interpolation will be performed
            #
            # But for that to work we need it needs to be as long as write factor
            assert "running_distance" not in update_state
            timestep = memory.timestep
            while timestep.dim() < write_factor.dim():
                timestep = timestep.unsqueeze(-1)
            update_state["running_distance"] = timestep
        return update_state, write_factor

    def reverse_memory(self,
                       update: MemoryData,
                       write_factor: torch.Tensor,
                       batch_mask: torch.Tensor,
                       memory: MemoryState,
                       ) -> MemoryState:
        """
        Figures out the previous memory state from the
        current and the various update factors.
        :param update: The update state to integrate.
        :param write_factor: The write factor
        :param batch_mask: Shape (...). Indicates whether memory can be updated. True means No.
        :param memory: The current memory state
        :return: The last memory state
        """

        # Perform the interpolation update.
        interpolatable_state = memory.get_interpolation_states()
        final_interpolatable_state = self._retard_memory(interpolatable_state, update,
                                                         write_factor, batch_mask)
        memory = memory.replace_interpolation(final_interpolatable_state)

        # Update the persistant factors
        memory.cum_write_mass -= write_factor
        memory.timestep -= batch_mask.to(write_factor.dtype)

        # Return the new memory instance
        return memory

    def advance_memory(self,
                       update: MemoryData,
                       write_factor: torch.Tensor,
                       batch_mask: torch.Tensor,
                       memory: MemoryState,
                       ) -> MemoryState:
        """
        Commits the computed update into the long term memory.
        Commits using interpolation.
        :param update: The update to integrate
        :param write_factor: The write factor to go with it
        :param batch_mask: Shape (...). Indicates whether memory can be updated. True means No.
        :param memory: The current memory
        :return: The updated memory
        """

        # Perform the interpolation update
        interpolatable_state = memory.get_interpolation_states()
        final_interpolatable_state = self._advance_memory(interpolatable_state, update,
                                                          write_factor, batch_mask)
        memory = memory.replace_interpolation(final_interpolatable_state)

        # Update the persistent factors

        memory.cum_write_mass += write_factor
        memory.timestep += batch_mask.to(write_factor.dtype)

        # Return the new memory instance
        return memory


class AbstractMemoryUnit(nn.Module, ABC):
    """
    An abstract specification contracting out work
    on the memory unit. Promises to work properly
    so long as you can initialize it with the required features.
    """

    def __init__(self,
                 create_state_unit: AbstractCreateState,
                 read_unit: AbstractReadMemory,
                 write_unit: AbstractWriteMemory,
                 ):
        super().__init__()
        self.state_creator = create_state_unit
        self.memory_reader = read_unit
        self.memory_writer = write_unit

    @torch.jit.export
    def create_state(self,
                     batch_shape: List[int]
                     ) -> MemoryState:
        """
        Creates and returns the blank memory state
        :param batch_shape: the batch shape to match
        :return: The concrete memory state.
        """
        return self.state_creator(batch_shape)

    @torch.jit.export
    def reverse(self,
                tensor: torch.Tensor,
                batch_mask: torch.Tensor,
                next_memory: MemoryState,
                ) -> Tuple[Tuple[torch.Tensor, MemoryState], MemoryState]:
        """
        The reverse implementation. We go about running the reverse process, then
        setup gradients and run the forward process to get our graphs. We return
        both the intermediate parameters set to accumulate gradients, and the
        same results as in the forward pass with graphs attached.

        :param tensor: The original tensor input
        :param batch_mask: Shape (...). Indicates whether memory can be updated. True means No.
        :param next_memory: The next memory state
        :return:
        - The originally produced output, exactly as seen in forward.
        - The original memory state. Setup to accumulate gradients.
        """
        with profiler.record_function("reversing the memories"):
            # Compute the write components.
            update, write_factor = self.memory_writer.compute_common(tensor, next_memory)

            # Get the original memory state.
            #
            # Make it retain grads so we can get
            # our gradients off it later for the next
            # backwards pass.
            with torch.no_grad():
                original_memory = self.memory_writer.reverse_memory(update, write_factor,
                                                                    batch_mask, next_memory)
            original_memory.setup_for_gradients_()

        # Manually complete the read
        with profiler.record_function("forward pass"):
            next_memory = self.memory_writer.advance_memory(update, write_factor, batch_mask, original_memory)
            read = self.memory_reader(tensor, next_memory)
        return (read, next_memory), original_memory

    @torch.jit.export
    def forward(self,
                tensor: torch.Tensor,
                batch_mask: torch.Tensor,
                memory: MemoryState
                ) -> Tuple[torch.Tensor, MemoryState]:
        """
        Forward pass for the memory unit.
        :param tensor: The tensor to use to access and update the mem state. Shape (..., d_model)
        :param batch_mask: Shape (...). Indicates whether memory can be updated. True means No.
        :param memory: The memory state.
        :return:
        - The response tensor. Shape (..., d_model)
        - The new memory state
        """
        with profiler.record_function("forward pass"):
            update, write_factor = self.memory_writer.compute_common(tensor, memory)
            next_memory = self.memory_writer.advance_memory(update, write_factor, batch_mask, memory)
            read = self.memory_reader(tensor, next_memory)
        return read, next_memory


class ConcreteMemoryUnitProtocol(Protocol):
    """
    Concrete instances of the abstract memory unit
    must be initialized with certain additional, externally
    provided parameters.

    These include the d_model under consideration,
    the dropout rate, the dtype, and the device.
    Consideration is made for these behaviors here.
    """

    def __init__(self,
                 d_model: int,
                 dtype: torch.dtype,
                 device: torch.device,
                 config: AbstractMemoryConfig
                 ):
        """
        Config spec for the protocol
        :param d_model: The width of tensors going into the read and write mechanism
        :param dtype: The dtype to build the parameters as
        :param device: The device to build the parameters as.
        :param config: The memory config for the specific instance.
        """

    def create_state(self,
                     batch_shape: torch.Size
                     ) -> MemoryState:
        """
        Creates and returns the blank memory state
        :param batch_shape: the batch shape to match
        :return: The concrete memory state.
        """

    def reverse(self,
                tensor: torch.Tensor,
                batch_mask: torch.Tensor,
                next_memory: MemoryState,
                ) -> Tuple[Tuple[torch.Tensor, MemoryState], MemoryState]:
        """
        The reverse implementation. We go about running the reverse process, then
        setup gradients and run the forward process to get our graphs. We return
        both the intermediate parameters set to accumulate gradients, and the
        same results as in the forward pass with graphs attached.

        :param tensor: The original tensor input
        :param batch_mask: Shape (...). Indicates whether memory can be updated. True means No.
        :param next_memory: The next memory state
        :return:
        - The originally produced output, exactly as seen in forward.
        - The original memory state. Setup to accumulate gradients.
        """

    def forward(self,
                tensor: torch.Tensor,
                batch_mask: torch.Tensor,
                memory: MemoryState
                ) -> Tuple[torch.Tensor, MemoryState]:
        """
        Forward pass for the memory unit.
        :param tensor: The tensor to use to access and update the mem state. Shape (..., d_model)
        :param batch_mask: Shape (...). Indicates whether memory can be updated. True means No.
        :param memory: The memory state.
        :return:
        - The response tensor. Shape (..., d_model)
        - The new memory state
        """


concrete_classes_registry: Dict[Type[AbstractMemoryConfig], Any] = {}


def register_concrete_implementation(config: Type[AbstractMemoryConfig],
                                     cls: ConcreteMemoryUnitProtocol
                                     ):
    """
    Registers a particular concrete implementation with the classes registry.
    This will allow one to just pass in a config, and have it instantly
    setup the class.
    :param config: The type of config under consideration
    :param cls: The class to instance the config with.
    """
    concrete_classes_registry[config] = cls


def make_memory_unit(d_model: int,
                     dtype: torch.dtype,
                     device: torch.device,
                     config: AbstractMemoryConfig
                     ) -> AbstractMemoryUnit:
    """
    Creates a concrete memory unit out of
    the specified parameters, assuming the config
    is a valid memory config belonging to a
    concrete implementation.

    :param d_model: The width of tensors flowing into the model
    :param dtype: The dtype to create under
    :param device: The device to create under
    :param config: The concrete config
    :return: The setup memory layer
    """
    if config.__class__ not in concrete_classes_registry:
        raise TypeError(f"config of name {config.__class__.__name__} was never registered")
    cls = concrete_classes_registry[config.__class__]
    return cls(d_model, dtype, device, config)
