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

    max_write_factor: The maximum probabilty that can be written in a single step.
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

    max_write_factor: float
    min_write_half_life_init: float
    max_write_half_life_init: float


MemoryData = Dict[str, torch.Tensor]


@torch.jit.script
def _advance_memory(memory_tensor: torch.Tensor,
                    update_tensor: torch.Tensor,
                    write_probability: torch.Tensor,
                    erase_probability: torch.Tensor,
                    batch_mask: torch.Tensor,
                    ) -> torch.Tensor:
    """
    Performs the advance memory step. As it is highly
    performant code, it has been scripted. It is scripted
    separately from it's class due to performance reasons.

    Implements:

    s_{i+1} = (1 - p_w*p_e)*s_i + p_w*u_i


    :param memory_tensor: The tensor currently existing in the memory
    :param update_tensor: The proposed update in the memory
    :param write_probability: The write factor to proceed under.
    :param erase_probability: The erase factor to proceed under.
    :param batch_mask: Whether or not the batch is masked. True means do not update
    :return: The next memory tensor
    """
    if update_tensor.shape != memory_tensor.shape:
        raise ValueError(
            f"Update and memory tensor have different shapes: {update_tensor.shape}, {memory_tensor.shape}")
    if write_probability.dim() > memory_tensor.dim():
        raise ValueError(
            f"WriteFactor tensor has too many dimensions: {write_probability.dim()} > {memory_tensor.dim()}")
    if batch_mask.dim() > memory_tensor.dim():
        raise ValueError(f"batch mask has too many dimensions: {batch_mask.dim()} > {memory_tensor.dim()}")
    if erase_probability.dim() > memory_tensor.dim():
        raise ValueError(f"erase factor has too many dims: {erase_probability.dim()} > {memory_tensor.dim()}")

    while write_probability.dim() < memory_tensor.dim():
        write_probability = write_probability.unsqueeze(-1)
    while batch_mask.dim() < memory_tensor.dim():
        batch_mask = batch_mask.unsqueeze(-1)
    while erase_probability.dim() < memory_tensor.dim():
        erase_probability = erase_probability.unsqueeze(-1)

    memory_update = memory_tensor * (1 - write_probability * erase_probability) + update_tensor * write_probability
    memory_tensor = torch.where(batch_mask, memory_tensor, memory_update)
    return memory_tensor


@torch.jit.script
def _advance_metrics(metrics: Dict[str, torch.Tensor],
                     write_probability: torch.Tensor,
                     erase_probability: torch.Tensor,
                     batch_mask: torch.Tensor
                     ) -> Dict[str, torch.Tensor]:
    """
    Advances various important metrics that are needed. This includes
    keeping track of the sum of the write factors as used for the write
    and erase gates.

    We also track, on average, what timestep we can reach before gradients
    do not propogate any further due to erasure.

    :param metrics: The list of metrics
    :param write_probability: The write factor to proceed under.
    :param erase_probability: The erase factor to proceed under.
    :param batch_mask: Whether or not the batch is masked. True means do not update
    :return: The dict of new metrics.
    """

    # Fairly tame metrics involving timesteps and probability masses.

    final_metrics = {}
    final_metrics['cum_write_mass'] = metrics['cum_write_mass'] + write_probability

    while write_probability.dim() < erase_probability.dim():
        write_probability = write_probability.unsqueeze(-1)
    erase_factor = write_probability * erase_probability

    final_metrics['cum_erase_mass'] = metrics['cum_erase_mass'] + erase_factor
    final_metrics['timestep'] = metrics['timestep'] + batch_mask.to(write_probability.dtype)

    # This is a running interpolation based on the timestep, that
    # tells us basically when the last erase step was. Conceptually, if
    # we erased everything, we would set this value to the timestep
    # that happened in.

    final_metrics['average_timestep_distance'] = (metrics["average_timestep_distance"] * (1 - erase_factor) +
                                                  metrics['timestep'] * erase_factor)
    return final_metrics


@torch.jit.script
def _retard_memory(memory_tensor: torch.Tensor,
                   update_tensor: torch.Tensor,
                   write_factor: torch.Tensor,
                   erase_factor: torch.Tensor,
                   batch_mask: torch.Tensor,
                   ) -> torch.Tensor:
    """
    Retards the memory tensor. This means we go backwards in
    time. As this is highly performant code, it has been scripted.
    Additionally, logarithms are used for reasons of numeric stability

    Implements:

    s_{i} = (s_{i+1} - p_w*u_i)/(1 - p_w*p_e)

    :param memory_tensor: The tensor currently existing in the memory
    :param update_tensor: The proposed update in the memory
    :param write_factor: The write factor to proceed under
    :param batch_mask: Whether or not the batch is masked. True means do not update
    :return: The previous memory tensor
    """
    if update_tensor.shape != memory_tensor.shape:
        raise ValueError(
            f"Update and memory tensor have different shapes: {update_tensor.shape}, {memory_tensor.shape}")
    if write_factor.dim() > memory_tensor.dim():
        raise ValueError(f"WriteFactor tensor has too many dimensions: {write_factor.dim()} > {memory_tensor.dim()}")
    if batch_mask.dim() > memory_tensor.dim():
        raise ValueError(f"batch mask has too many dimensions: {batch_mask.dim()} > {memory_tensor.dim()}")
    if erase_factor.dim() > memory_tensor.dim():
        raise ValueError(f"erase factor has too many dims: {erase_factor.dim()} > {memory_tensor.dim()}")

    while write_factor.dim() < memory_tensor.dim():
        write_factor = write_factor.unsqueeze(-1)
    while batch_mask.dim() < memory_tensor.dim():
        batch_mask = batch_mask.unsqueeze(-1)
    while erase_factor.dim() < memory_tensor.dim():
        erase_factor = erase_factor.unsqueeze(-1)

    log_memory = torch.log(update_tensor * write_factor - memory_tensor + 1e-9)
    log_memory -= torch.log(1 - write_factor * erase_factor + 1e-9)  # divides
    memory_update = torch.exp(log_memory)
    memory_tensor = torch.where(batch_mask, memory_tensor, memory_update)
    return memory_tensor


@torch.jit.script
def _retard_metrics(metrics: Dict[str, torch.Tensor],
                    write_probability: torch.Tensor,
                    erase_probability: torch.Tensor,
                    batch_mask: torch.Tensor
                    ) -> Dict[str, torch.Tensor]:
    """
    Retards the various metrics we are tracking, figuring out what they
    are by walking the memory backwards. =

    :param metrics: The list of metrics
    :param write_probability: The write factor to proceed under.
    :param erase_probability: The erase factor to proceed under.
    :param batch_mask: Whether or not the batch is masked. True means do not update
    :return: The dict of new metrics.
    """

    # Fairly tame metrics involving timesteps and probability masses.

    final_metrics = {}
    final_metrics['cum_write_mass'] = metrics['cum_write_mass'] - write_probability

    while write_probability.dim() < erase_probability.dim():
        write_probability = write_probability.unsqueeze(-1)
    erase_factor = write_probability * erase_probability

    final_metrics['cum_erase_mass'] = metrics['cum_erase_mass'] - erase_factor
    final_metrics['timestep'] = metrics['timestep'] - batch_mask.to(write_probability.dtype)

    # This is a running interpolation based on the timestep, that
    # tells us basically when the last erase step was. Conceptually, if
    # we erased everything, we would set this value to the timestep
    # that happened in.

    log_metric = torch.log(metrics['average_timestep_distance'] - metrics['timestep'] * erase_factor + 1e-9)
    log_metric -= torch.log(1 - erase_factor + 1e-9)
    final_metrics['average_timestep_distance'] = torch.exp(log_metric)

    return final_metrics


class MemoryState(PytreeState):
    """
    Memory state is internally divided into
    mutable and persistent dictionaries of tensors.

    ---- memory write, reversable, forward ----

    Memory is maintained in terms of a write-erase
    action. In particular, given current state s_i, update u_i,
    erase probability p_e, and write probability p_w, the updated state
    is created out of the provided mutable tensor by a broadcasted
    version of

    s_{i+1} = (1 - p_w*p_e)*s_i + p_w*u_i

    This can be reversed during the reverse pass. Getting the last state from
    the current one proceeds as:

    s_{i} = (s_{i+1} - p_w*u_i)/(1 - p_w*p_e)

    The write probability and erase probability are
    usually further constrained to be within a certain
    rate on the actual gates.

    ---- metrics  ----

    The memory state is designed to support a particular,
    long duration memory metric or series of memory
    metrics as well.

    cum_write_mass: The cumulative write mass of all write factors.
    cum_erase_mass: The cumulative erase mass of all the erase factors.
    timestep: The number of timesteps, per batch. Stops updating when padding is detected
    average_timestep_distance: The "average" location the memory is peering into. Covers all write factor slots.
                              - It is maintained by keeping a running average around that is updated
                                based on the erase gate and the timestep.
                              - When a strong erase operation is detected, we move towards the current timestep
                                as an interpolation.
    average_gradient_distance: The "average" distance in timesteps that gradients can propogate back by.
    normalized_gradient_distance: Indicates what percentage into the past, measured over 0-1, gradients
                                  can propogate through
                                - 0 means that we are looking only into the immediate past
                                - 1 would mean they can travel all the way back to the beginning
    """

    @property
    def cum_write_mass(self) -> torch.Tensor:
        """
        The cumulative write mass contains the sum of all
        the write factors.
        """
        return self.metric_tensors["cum_write_mass"]

    @property
    def cum_erase_mass(self) -> torch.Tensor:
        """
        The cumulative erase mass contains the sum of all
        erase operations.
        """
        return self.metric_tensors["cum_erase_mass"]

    @cum_write_mass.setter
    def cum_write_mass(self, value: torch.Tensor):
        assert value.shape == self.cum_write_mass.shape
        self.metric_tensors["cum_write_mass"] = value

    @cum_erase_mass.setter
    def cum_erase_mass(self, value: torch.Tensor):
        assert value.shape == self.cum_erase_mass.shape
        self.metric_tensors["cum_erase_mass"] = value

    @property
    def timestep(self) -> torch.Tensor:
        """
        The timestep we are currently on, when accounting
        for batch padding masking.
        """
        return self.metric_tensors["timestep"]

    @timestep.setter
    def timestep(self, value: torch.Tensor):
        assert value.shape == self.timestep.shape
        self.metric_tensors["timestep"] = value

    @property
    def average_timestep_distance(self) -> torch.Tensor:
        """
        The unnormalized distance, indicating from the start of the sequence
        where each memory unit will, on average, be reading from.
        """
        return self.memory_tensors["average_timestep_distance"]

    @property
    def average_gradient_distance(self) -> torch.Tensor:
        """
        The unnormalized distance in terms of timesteps that we can expect,
        on average, gradients to propogate through.
        """
        timestep = self.timestep
        while timestep.dim() < self.average_timestep_distance.dim():
            timestep = timestep.unsqueeze(-1)
        return timestep - self.average_timestep_distance

    @property
    def normalized_gradient_distance(self) -> torch.Tensor:
        """
        The normalized timestep distance. Measure the current timestep as
        0, and all the way back at the beginning as one. Indicates how
        far into the past each memory slot is looking
        """
        timestep = self.timestep
        running_distance = self.average_timestep_distance
        while timestep.dim() < running_distance.dim():
            timestep = timestep.unsqueeze(-1)
        return (timestep - running_distance) / (timestep + 1e-6)

    def __init__(self,
                 metric_tensors: MemoryData,
                 memory_tensors: MemoryData,
                 persistent: MemoryData
                 ):
        if "cum_write_mass" not in metric_tensors:
            raise KeyError("cum_write_mass was not found in persistent state. Did you forget to init it?")
        if "cum_erase_mass" not in metric_tensors:
            raise KeyError("cum_erase_mass was not found in persistent state. Did you forget to init it?")
        if "timestep" not in metric_tensors:
            raise KeyError("timestep was not found in persistent state. Did you forget to init it?")
        if "average_timestep_distance" not in metric_tensors:
            raise KeyError("average_timestep_distance not found.")

        self.metric_tensors = metric_tensors
        self.memory_tensors = memory_tensors
        self.persistent_tensors = persistent

    def step_memory_forward(self,
                            update: MemoryData,
                            write_probability: torch.Tensor,
                            erase_probability: torch.Tensor,
                            batch_mask: torch.Tensor,
                            ):
        """
        Steps the memory forward by one unit. This consists
        of performing the write/erase action.

        :param batch_mask: The batch mask for the step
        :param write_probability: The write factor for the step
        :param erase_probability: The erase factor for the step
        :param update: The memory update for the step
        :return: The new memory state
        """

        # Update the substate memories.
        memories = {}
        for name in self.memory_tensors.keys():
            memory_tensor = self.memory_tensors[name]
            update_tensor = update[name]
            memories[name] = _advance_memory(memory_tensor,
                                             update_tensor,
                                             write_probability,
                                             erase_probability,
                                             batch_mask)
        metrics = _advance_metrics(self.metric_tensors,
                                   write_probability,
                                   erase_probability,
                                   batch_mask)
        return MemoryState(metrics, memories, self.persistent_tensors)

    def step_memory_reverse(self,
                            update: MemoryData,
                            write_probability: torch.Tensor,
                            erase_probability: torch.Tensor,
                            batch_mask: torch.Tensor,
                            ) -> 'MemoryState':
        """
        Steps the memory reverse by one unit. This consists
        of taking the inverse of the write/erase action.

        :param batch_mask: The batch mask for the step
        :param write_probability: The write factor for the step
        :param erase_probability: The erase factor for the step
        :param update: The memory update for the step
        :return: The new memory state
        """

        memories = {}
        for name in self.memory_tensors.keys():
            memory_tensor = self.memory_tensors[name]
            update_tensor = update[name]
            memories[name] = _retard_memory(memory_tensor,
                                            update_tensor,
                                            write_probability,
                                            erase_probability,
                                            batch_mask)

        metrics = _retard_metrics(self.metric_tensors,
                                  write_probability,
                                  erase_probability,
                                  batch_mask
                                  )
        return MemoryState(metrics, memories, self.persistent_tensors)

    def get_memories(self) -> MemoryData:
        return self.memory_tensors

    def get_persistent(self) -> MemoryData:
        return self.persistent_tensors

    def save_state(self) -> Tuple[Tuple[MemoryData, MemoryData, MemoryData], None]:
        return (self.metric_tensors, self.memory_tensors, self.persistent_tensors), None

    @classmethod
    def load_state(cls,
                   pytree: Tuple[MemoryData, MemoryData],
                   bypass: None) -> 'MemoryState':
        metric_tensors, memory_tensors, persistent_tensors = pytree
        return cls(metric_tensors, memory_tensors, persistent_tensors)

    def setup_for_gradients_(self):
        """
        Turns the stored tensors into leafs which accumulate gradients
        """
        for tensor in self.memory_tensors.values():
            tensor.detach_()
            tensor.requires_grad_(True)
        for tensor in self.metric_tensors.values():
            tensor.detach_()
            tensor.requires_grad_(True)


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
    def forward(self, batch_shape: List[int]) -> MemoryState:
        """
        When implemented, returns a memory object when seeing a batch shape
        :param batch_shape: The batch shape to consider
        :return: The memory state. Remember to include
                 cumulative_write_factors in persistant.
        """


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
    def read_memory(self,
                    query: torch.Tensor,
                    memories: MemoryData,
                    persistent: MemoryData
                    ) -> torch.Tensor:
        """
        Abstract specification of a memory
        read action. What the user needs to implement

        :param query: The query to read with. Shape (..., d_model). Recurrent, so no items dim
        :param memories: The memory tensors. Same as in create.
        :param persistent: The persistent tensors. These never change across timesteps. Generally parameters
        :return: The result of reading. Shape (..., d_model)
        """

    def forward(self,
                query: torch.Tensor,
                memories: MemoryState,
                ) -> torch.Tensor:
        """
        Performs the memory read process
        :param query: The query to read with. Shape (..., d_model). Recurrent, so no items dim
        :param memories: The memory state
        :return:
        """
        persistent = memories.get_persistent()
        memory = memories.get_memories()
        return self.read_memory(query, memory, persistent)


class AbstractWriteMemory(nn.Module, ABC):
    """
    An abstract implementation of the write
    memory process. Most of the actual
    update logic is contained in earlier mechanism
    such as the memory state.
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
                        persistent: Dict[str, torch.Tensor],
                        ) -> Tuple[Dict[str, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    ]:
        """
        The main user specified component.

        The user must specify how to process a query into
        a state update and a write probability. Note you
        only have access to the persistent state when making
        these computations, and should only depend on things
        that are initialized ONCE.

        :param query: The query tensor. Shape (..., d_model), presumably
        :param persistent: Anything that was declared from setup, but that does not change
                           between timesteps.
        :return:
        - update: An implementation-specific update, consisting of a dict of
                  tensortrees corrolating with the interpolation memory content.
        - write_probability: A write probability that tells us how strongly to write to the memory slots.
            - Must cleanly multiply the interpolation factor shape.
        - erase_probability: A erase probability that tells us how strongly to forget what we
                             have seen before.
        """

    ## External interface
    #
    # The intention is to call compute common,
    # then your forward and backwards methods as needed.

    def compute_common(self,
                       query: torch.Tensor,
                       memory: MemoryState,
                       ) -> Tuple[MemoryData, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Internal helper with the additional logic needed
        to get the write factor ready to go based on the write
        probabilities.

        Calls into implementation detail compute_common
        :param query: The query to make the update and factor on
        :param memory: The memory state. The parameter state will be retrieved off it.
        :return: The update state, the write prob, the erase prob.
        """
        with profiler.record_function("Computing write parameters"):
            persistent_tensors = memory.get_persistent()
            with profiler.record_function("Computing write parameters: Implementation"):
                update_state, write_probability, erase_probability = self._compute_common(query,
                                                                                          persistent_tensors)

            # Compute write probability
            interpolation_rates = self._compute_interpolation_factors(self._interpolation_logits)
            interpolation_rates = self._max_interpolation_rate * interpolation_rates
            write_probability = interpolation_rates * write_probability

        return update_state, (write_probability, erase_probability)

    def reverse_memory(self,
                       update: MemoryData,
                       control_factors: Tuple[torch.Tensor, torch.Tensor],
                       batch_mask: torch.Tensor,
                       memory: MemoryState,
                       ) -> MemoryState:
        """
        Figures out the previous memory state from the
        current and the various update factors.
        :param update: The update state to integrate.
        :param control_factors: The control probabilities
        :param batch_mask: Shape (...). Indicates whether memory can be updated. True means No.
        :param memory: The current memory state
        :return: The last memory state
        """

        # Perform the interpolation update.
        write_probability, erase_probability = control_factors
        return memory.step_memory_reverse(update, write_probability, erase_probability, batch_mask)

    def advance_memory(self,
                       update: MemoryData,
                       control_factors: Tuple[torch.Tensor, torch.Tensor],
                       batch_mask: torch.Tensor,
                       memory: MemoryState,
                       ) -> MemoryState:
        """
        Commits the computed update into the long term memory.
        Commits using interpolation.
        :param update: The update to integrate
        :param control_factors: The control probabilities
        :param batch_mask: Shape (...). Indicates whether memory can be updated. True means No.
        :param memory: The current memory
        :return: The updated memory
        """

        write_probability, erase_probability = control_factors
        return memory.step_memory_forward(update, write_probability, erase_probability, batch_mask)


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
