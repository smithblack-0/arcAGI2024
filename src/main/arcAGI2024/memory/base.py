"""
This specifies the abstract contract our recurrent memory
system follows, and also implements the interpolation
mixing process.
"""
import functools
import os
import math
import torch
import json
import math
from torch import nn
from torch.autograd import profiler
from torch.nn import functional as F
from torch.distributions import normal
from typing import Tuple, Dict, Union, Protocol, Type, Any, Optional, List
from dataclasses import dataclass, asdict
from ..base import PytreeState, SavableConfig, DeviceDtypeWatch, can_broadcast
from abc import ABC, abstractmethod


@dataclass
class GradientTimeLossConfig(SavableConfig):
    """
    The  specification for defining how the gradient
    time loss is computed and integrated.

    Basically, we analyze the timestep metrics which
    tell us how far back in time gradients can travel
    as a set of bins like a histogram, producing a discrete
    probability distribution.

    Then,
    ---- fields ----
.
    num_bins:
    - Number of histogram bins. They will be spaced
      evenly over 0..1.
    - Keeping in mind the idea of hard bins is useful for understanding the config.
      However, the actual implementation uses gaussian soft bins for losses
    deviation_factor:
    - A factor we multiply the default std by if we want to change how wide the soft bins are
    - By default, bins are setup so that the std deviation covers the width of a hard bin - that is,
      it is equal to the distance from the middle to the edge of a bin.
    - Adjust this away from 1.0 to, for example, 2.0 to spread over two of these deviations.
    target_distribution:
     - Expected probability distributions for the bins.
     - Earlier bins go back in time to earlier values, with a bin of 0 going all the way back to the
       first timestep.
     - Must add up to close to 1.0
    target_thresholds:
     - How much, in absolute probability, a bin can deviate from its target before loss kicks in
     - Loss kicks in aggressively as a constraint
     - Symmetric.
    loss_weight:
    - The strength of the loss when it kicks in.
    - Should usually be quite high, as this is a constraint that only activates when needed.
    loss_type:
    - The loss type that can kick in.
    - Known at the moment are "quadratic_threshold" and "linear_threshold"
    """

    num_bins: int
    z_score: float
    target_distribution: List[float]
    target_thresholds: List[float]
    loss_weight: float
    loss_type: str

    def __post_init__(self):
        if self.num_bins < 1:
            raise ValueError("num_bins must be greater than or equal to 1.")
        if self.z_score <= 0:
            raise ValueError("deviation factor must be greater than zero")
        distribution_sum = sum(self.target_distribution)
        if abs(distribution_sum - 1) > 1e-4:
            raise ValueError("target distribution did not sum to 1.0")
        for threshold, distribution_element in zip(self.target_thresholds, self.target_distribution):
            if distribution_element + threshold > 1.0:
                raise ValueError("Threshold takes probability over 1.0 for a bin")
            if distribution_element - threshold < 0.0:
                raise ValueError("Threshold takes probability below 0.0 for a bin")
        if self.loss_weight < 0.0:
            raise ValueError("loss_weight must be greater than or equal to 0.")


@dataclass
class MemRegularizationLossConfig(SavableConfig):
    """
    The specification for the regularization loss for
    the memory config. This generally does things like
    discourages the memory from getting too large.

    ---- fields ----

    loss_type: type of regularization loss. 'l1' and 'l2' are common cases
    loss_weight: How strong the loss should be. Generally fairly weak in comparison
                 to the constraint loss.
    """
    magnitude_loss_type: str
    magnitude_loss_weight: float

    def __post_init__(self):
        if self.magnitude_loss_type not in {'l1', 'l2'}:
            raise ValueError(
                f"Unsupported magnitude_loss_type '{self.magnitude_loss_type}'. Supported types are 'l1' and 'l2'.")
        if self.magnitude_loss_weight < 0.0:
            raise ValueError("magnitude_loss_weight must be non-negative.")


class AbstractMemoryConfig(SavableConfig):
    """
    The abstract specification for the memory
    config contains a few decent defaults.
    It can be implemented to produce configs more
    specific to a particular memory flavor.

    ** Interpolation factor config ***

    The following two control how the write interpolation rates are initialized. Those factors
    are initialized uniformly between these, and can then be trained. These are basically
    decay factors between 0 and 1, that control how fast the running interpolation decays away
    when the model chooses to write to the memory in every step.

    min_write_half_life_init: Controls the minimum half life that the write interpolation rates
                              can be set on initialization. Must be >= 0.
    max_write_half_life_init: Controls the maximum half life that the write interpolation rates
                              can be set on initialization. Must be > min_write_half_life_init.
    erase_epsilon_factor: A small factor between 0 and 1.  Usually much closer to zero than 1. Limits
                          how strongly the model can commit to the erase mode of operation, which
                          prevents division by zero and keeps the logic invertible.


    """

    @property
    @abstractmethod
    def interpolation_factor_shapes(self) -> torch.Size:
        # Must return something that broadcasts with the write
        # probabilities.
        raise NotImplementedError("Need to implement interpolation shapes.")

    @property
    def erase_cap_factor(self) -> float:
        return 1 - self.erase_epsilon_factor

    min_write_half_life_init: float
    max_write_half_life_init: float
    erase_epsilon_factor: float
    gradient_loss: GradientTimeLossConfig
    mem_regularization_loss: MemRegularizationLossConfig

    def __post_init__(self):
        if self.min_write_half_life_init < 0:
            raise ValueError("min_write_half_life_init must be non-negative.")
        if self.max_write_half_life_init <= self.min_write_half_life_init:
            raise ValueError("max_write_half_life_init must be greater than min_write_half_life_init.")
        if not (0.0 < self.erase_epsilon_factor < 1.0):
            raise ValueError("erase_epsilon_factor must be between 0 and 1.")


MemoryData = Dict[str, torch.Tensor]


def _compute_erase_factor(write_probability: torch.Tensor,
                          erase_probability: torch.Tensor,
                          epsilon_factor: float
                          ) -> torch.Tensor:
    """
    Compute the erase factor.

    This is one of two primary gates controlling the memory system.
    Due to reversibility concerns, an epsilon factor is included
    in the computation that limits how close the erase factor
    can get to zero.

    Since it is recomputed in several places, we centralize the logic
    in a helper function.

    :param write_probability: The write probability
    :param erase_probability: The erase probablity
    :param epsilon_factor: The epsilon constraint
    :return: The resulting control gate
    """
    return 1 - epsilon_factor * write_probability * erase_probability


def _compute_write_factor(write_probability: torch.Tensor) -> torch.Tensor:
    """
    Compute the write factor.

    This is one of two primary gates controlling the memory system.

    At the time of this comment, it is actually a no-op, but
    is included for code clarity reasons. Torchscript
    will compile that away.

    :param write_probability: The write probability
    :return: The resulting control gate
    """
    return write_probability


def _perform_batch_masking(batch_masking: torch.Tensor,
                           unmasked_case: torch.Tensor,
                           masked_case: torch.Tensor,
                           ) -> torch.Tensor:
    """
    Performs the batch masking. Keep in mind a value of
    true would mean mask. Unsqueezes the batch mask if needed
    :param batch_masking: The batch mask
    :param unmasked_case: What to provide in the unmasked case
    :param masked_case: What to provide in the masked case
    :return: The returned tensor
    """
    if unmasked_case.shape != masked_case.shape:
        raise ValueError(
            f"unmasked and masked case do not have same shape: {unmasked_case.shape} and {masked_case.shape}")
    while batch_masking.dim() < unmasked_case.dim():
        batch_masking = batch_masking.unsqueeze(-1)
    return torch.where(batch_masking, masked_case, unmasked_case)


def _standardize_step_parameters(state_tensor: torch.Tensor,
                                 update_tensor: torch.Tensor,
                                 erase_gate: torch.Tensor,
                                 write_gate: torch.Tensor) -> Tuple[torch.Tensor,
torch.Tensor,
torch.Tensor,
torch.Tensor]:
    """
    Checks if the step parameters are sane. Standardizes those
    parameters.
    :return:
    """
    if not can_broadcast(state_tensor.shape, update_tensor.shape):
        raise ValueError(
            f"Update and state tensor have nonbroadcastable shapes: {update_tensor.shape}, {state_tensor.shape}")
    if write_gate.dim() > state_tensor.dim():
        raise ValueError(
            f"write gate tensor has too many dimensions: {write_gate.dim()} > {state_tensor.dim()}")
    if erase_gate.dim() > state_tensor.dim():
        raise ValueError(f"erase gate has too many dims: {erase_gate.dim()} > {state_tensor.dim()}")

    num_needed_dims = update_tensor.dim()
    while num_needed_dims > write_gate.dim():
        write_gate = write_gate.unsqueeze(-1)
    while num_needed_dims > erase_gate.dim():
        erase_gate = erase_gate.unsqueeze(-1)

    return state_tensor, update_tensor, erase_gate, write_gate


def _step_state_forward(state_tensor: torch.Tensor,
                        update_tensor: torch.Tensor,
                        erase_gate: torch.Tensor,
                        write_gate: torch.Tensor) -> torch.Tensor:
    """
    Performs the forward core memory step. This is a weighted interpolate between
    various components.

    Operates according to:

    s_{i+1} = s_{i}*g_{erase} + u_{i}*g_{write}
    """
    state_tensor, update_tensor, erase_gate, write_gate = _standardize_step_parameters(state_tensor,
                                                                                       update_tensor,
                                                                                       erase_gate,
                                                                                       write_gate)
    return state_tensor * erase_gate + update_tensor * write_gate


def _step_state_reverse(state_tensor,
                        update_tensor,
                        erase_gate,
                        write_gate) -> torch.Tensor:
    """
    Performs the reverse core memory step action. This is an inverse
    of the forward step. Note that it is presumed some epsilon action
    has been taken against the erae gate to avoid dividing by zero.

    Operates according to:

    s_{i} = (s_{i+1} - u_{i}*g_{write})/(g_{erase})

    :return: The original state.
    """
    state_tensor, update_tensor, erase_gate, write_gate = _standardize_step_parameters(state_tensor,
                                                                                       update_tensor,
                                                                                       erase_gate,
                                                                                       write_gate)
    return (state_tensor - update_tensor * write_gate) / erase_gate


def _advance_memory(memory_tensor: torch.Tensor,
                    update_tensor: torch.Tensor,
                    write_gate: torch.Tensor,
                    erase_gate: torch.Tensor,
                    batch_mask: torch.Tensor,
                    ) -> torch.Tensor:
    """
    Performs the advance memory step. As it is highly
    performant code, it has been scripted. It is scripted
    separately from it's class due to performance reasons.

    Implements:

    s_{i+1} = (1 - p_w*p_e)*s_i + p_w*u_i

    With safety factors for numeric effects.

    :param memory_tensor: The tensor currently existing in the memory
    :param update_tensor: The proposed update in the memory
    :param write_gate: The write gate to proceed under.
    :param erase_gate: The erase gate to proceed under.
    :param batch_mask: Whether or not the batch is masked. True means do not update
    :return: The next memory tensor
    """
    memory_update = _step_state_forward(memory_tensor, update_tensor, erase_gate, write_gate)
    memory_tensor = _perform_batch_masking(batch_mask, memory_update, memory_tensor)
    return memory_tensor


def _advance_metrics(metrics: Dict[str, torch.Tensor],
                     write_gate: torch.Tensor,
                     erase_gate: torch.Tensor,
                     batch_mask: torch.Tensor
                     ) -> Dict[str, torch.Tensor]:
    """
    Advances various important metrics that are needed. This includes
    keeping track of the sum of the write factors as used for the write
    and erase gates.

    We also track, on average, what timestep we can reach before gradients
    do not propogate any further due to erasure.

    :param metrics: The list of metrics
    :param write_gate: The write factor to proceed under.
    :param erase_gate: The erase factor to proceed under.
    :param batch_mask: Whether or not the batch is masked. True means do not update
    :return: The dict of new metrics.
    """

    # Fairly tame metrics involving timesteps and probability masses.

    final_metrics = {}
    final_metrics['cum_write_mass'] = metrics['cum_write_mass'] + write_gate
    final_metrics['cum_erase_mass'] = metrics['cum_erase_mass'] + erase_gate
    final_metrics['timestep'] = metrics['timestep'] + (~batch_mask).to(write_gate.dtype)

    # This is a set of running interpolations. One
    # tells us, basically, how far into the past we could actually
    # propagate gradients. The other tells us since the last
    # erase how much write mass has been committed.

    one = torch.tensor(1.0, dtype=write_gate.dtype, device=write_gate.device)

    final_metrics["effective_write_mass"] = _step_state_forward(metrics["effective_write_mass"],
                                                                write_gate,
                                                                erase_gate,
                                                                one
                                                                )

    timestep = final_metrics["timestep"]
    while timestep.dim() < metrics["average_timestep_distance"].dim():
        timestep = timestep.unsqueeze(-1)

    final_metrics["average_timestep_distance"] = _step_state_forward(metrics["average_timestep_distance"],
                                                                     timestep,
                                                                     erase_gate,
                                                                     1 - erase_gate
                                                                     )

    # Account for batch masking. Metrics do not update where the batch was masked
    for name in final_metrics.keys():
        initial_metric = metrics[name]
        final_metric = final_metrics[name]
        final_metrics[name] = _perform_batch_masking(batch_mask, final_metric, initial_metric)
    return final_metrics


def _retard_memory(memory_tensor: torch.Tensor,
                   update_tensor: torch.Tensor,
                   write_gate: torch.Tensor,
                   erase_gate: torch.Tensor,
                   batch_mask: torch.Tensor,
                   ) -> torch.Tensor:
    """
    Retards the memory tensor. This means we go backwards in
    time. As this is highly performant code, it has been scripted.

    Implements:

    s_{i} = (s_{i+1} - p_w*u_i)/(1 - p_w*p_e)

    :param memory_tensor: The tensor currently existing in the memory
    :param update_tensor: The proposed update in the memory
    :param write_gate: The write gate  to proceed under
    :param erase_gate: The erase gate to proceed under.
    :param batch_mask: Whether or not the batch is masked. True means do not update
    :return: The previous memory tensor
    """
    memory_update = _step_state_reverse(memory_tensor, update_tensor, erase_gate, write_gate)
    memory_tensor = _perform_batch_masking(batch_mask, memory_update, memory_tensor)
    return memory_tensor


@torch.jit.script
def _retard_metrics(metrics: Dict[str, torch.Tensor],
                    write_gate: torch.Tensor,
                    erase_gate: torch.Tensor,
                    batch_mask: torch.Tensor
                    ) -> Dict[str, torch.Tensor]:
    """
    Retards the various metrics we are tracking, figuring out what they
    are by walking the memory backwards. =

    :param metrics: The list of metrics
    :param write_gate: The write gate  to proceed under
    :param erase_gate: The erase gate to proceed under.
    :param batch_mask: Whether or not the batch is masked. True means do not update
    :return: The dict of new metrics.
    """

    final_metrics = {}

    # Fairly tame metrics involving timesteps and probability masses.

    final_metrics['cum_write_mass'] = metrics['cum_write_mass'] - write_gate
    final_metrics['cum_erase_mass'] = metrics['cum_erase_mass'] - erase_gate
    final_metrics['timestep'] = metrics['timestep'] - (~batch_mask).to(write_gate.dtype)

    # This is a set of running interpolations. One
    # tells us, basically, how far into the past we could actually
    # propogate gradients. The other tells us since the last
    # erase how much write mass has been committed.

    one = torch.tensor(1.0, dtype=write_gate.dtype, device=write_gate.device)

    final_metrics["effective_write_mass"] = _step_state_reverse(metrics["effective_write_mass"],
                                                                write_gate,
                                                                erase_gate,
                                                                one
                                                                )

    timestep = metrics["timestep"]
    while timestep.dim() < metrics["average_timestep_distance"].dim():
        timestep = timestep.unsqueeze(-1)

    final_metrics["average_timestep_distance"] = _step_state_reverse(metrics["average_timestep_distance"],
                                                                     timestep,
                                                                     erase_gate,
                                                                     1 - erase_gate
                                                                     )

    # Account for batch masking. Metrics do not update where the batch was masked
    for name in final_metrics.keys():
        initial_metric = metrics[name]
        final_metric = final_metrics[name]
        final_metrics[name] = _perform_batch_masking(batch_mask, final_metric, initial_metric)
    return final_metrics


# Due to how torchscript works we must define top level
# aliases so the RCB callback can find them
torch_device = torch.device
torch_dtype = torch.dtype


class MemoryState(PytreeState):
    """
    Memory state is internally divided into
    mutable and persistent dictionaries of tensors.

    ---- memory write, reversible, forward ----

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
    timestep: The number of timesteps, per batch. Stops updating when padding is detected.
    average_timestep_distance: The "average" location the memory is peering into. Covers all write factor slots.
                              - It is maintained by keeping a running average that is updated
                                based on the erase gate and the timestep.
                              - When a strong erase operation is detected, we move towards the current timestep
                                as an interpolation.
    normalized_timestep_distance: The average timestep distance normalized by the current timestep.
                                    Indicates the relative position in time that the memory is referencing,
                                    scaled between 1 (current timestep) and 0 (beginning of the sequence).
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

    @property
    def effective_write_mass(self) -> torch.Tensor:
        """
        The write mass that has been committed since
        the last erase, basically.
        """
        return self.metric_tensors["effective_write_mass"]

    @property
    def normalized_effective_write_mass(self) -> torch.Tensor:
        """
        The effective write mass, but normalized over
        the number of timesteps.
        """
        timestep = self.timestep
        while timestep.dim() < self.effective_write_mass.dim():
            timestep = timestep.unsqueeze(-1)
        return self.effective_write_mass / (timestep + 1e-9)

    @property
    def timestep(self) -> torch.Tensor:
        """
        The timestep we are currently on, when accounting
        for batch padding masking.
        """
        return self.metric_tensors["timestep"]

    @property
    def average_timestep_distance(self) -> torch.Tensor:
        """
        The unnormalized distance, indicating from the start of the sequence
        where each memory unit will, on average, be reading from.
        """
        return self.metric_tensors["average_timestep_distance"]

    @property
    def normalized_timestep_distance(self) -> torch.Tensor:
        """
        The normalized average timestep distance.

        This metric represents the average relative position in time that the memory is referencing,
        scaled between 1 (current timestep) and 0 (beginning of the sequence).

        It is computed as:
            normalized_timestep_distance = average_timestep_distance / (timestep + epsilon)

        where:
        - `average_timestep_distance` is the exponentially weighted average timestep, updated based on the erase factor.
        - `timestep` is the current timestep, adjusted for batch masking.
        - `epsilon` is a small constant to prevent division by zero.
        """
        timestep = self.timestep
        while timestep.dim() < self.average_timestep_distance.dim():
            timestep = timestep.unsqueeze(-1)
        return self.average_timestep_distance / (timestep + 1e-9)

    @property
    def device(self) -> torch_device:
        return self.cum_write_mass.device

    @property
    def dtype(self) -> torch_dtype:
        return self.cum_write_mass.dtype

    def __init__(self,
                 metric_tensors: MemoryData,
                 memory_tensors: MemoryData,
                 persistent: MemoryData
                 ):
        if "cum_write_mass" not in metric_tensors:
            raise KeyError("cum_write_mass was not found in persistent state. Did you forget to init it?")
        if "cum_erase_mass" not in metric_tensors:
            raise KeyError("cum_erase_mass was not found in persistent state. Did you forget to init it?")
        if "effective_write_mass" not in metric_tensors:
            raise KeyError("effective_write_mass was not found in persistent state. Did you forget to init it?")
        if "timestep" not in metric_tensors:
            raise KeyError("timestep was not found in persistent state. Did you forget to init it?")
        if "average_timestep_distance" not in metric_tensors:
            raise KeyError("average_timestep_distance not found.")

        self.metric_tensors = metric_tensors
        self.memory_tensors = memory_tensors
        self.persistent_tensors = persistent

    def step_memory_forward(self,
                            update: MemoryData,
                            write_gate: torch.Tensor,
                            erase_gate: torch.Tensor,
                            batch_mask: torch.Tensor,
                            ):
        """
        Steps the memory forward by one unit. This consists
        of performing the write/erase action.

        :param batch_mask: The batch mask for the step
        :param write_gate: The write gate for the step
        :param erase_gate: The erase gate for the step
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
                                             write_gate,
                                             erase_gate,
                                             batch_mask)

        # Update the metrics
        metrics = _advance_metrics(self.metric_tensors,
                                   write_gate,
                                   erase_gate,
                                   batch_mask)

        # Return the new state
        return MemoryState(metrics, memories, self.persistent_tensors)

    def step_memory_reverse(self,
                            update: MemoryData,
                            write_gate: torch.Tensor,
                            erase_gate: torch.Tensor,
                            batch_mask: torch.Tensor,
                            ) -> 'MemoryState':
        """
        Steps the memory reverse by one unit. This consists
        of taking the inverse of the write/erase action.

        :param batch_mask: The batch mask for the step
        :param write_gate: The write gate for the step
        :param erase_gate: The erase gate for the step
        :param update: The memory update for the step
        :return: The new memory state
        """

        memories = {}
        # Update the memories
        for name in self.memory_tensors.keys():
            memory_tensor = self.memory_tensors[name]
            update_tensor = update[name]
            memories[name] = _retard_memory(memory_tensor,
                                            update_tensor,
                                            write_gate,
                                            erase_gate,
                                            batch_mask)

        # Update the metrics
        metrics = _retard_metrics(self.metric_tensors,
                                  write_gate,
                                  erase_gate,
                                  batch_mask
                                  )

        # Return the last memory state.
        return MemoryState(metrics, memories, self.persistent_tensors)

    def get_memories(self) -> MemoryData:
        return self.memory_tensors

    def get_persistent(self) -> MemoryData:
        return self.persistent_tensors

    def save_state(self) -> Tuple[Tuple[MemoryData, MemoryData, MemoryData], None]:
        return (self.metric_tensors, self.memory_tensors, self.persistent_tensors), None

    @classmethod
    def load_state(cls,
                   pytree: Tuple[MemoryData, MemoryData, MemoryData],
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
                 cumulative_write_factors in persistent.
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
        self._erase_cap_factor = config.erase_cap_factor
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

            # Compute interpolation rate adjustment to write probability
            interpolation_rates = self._compute_interpolation_factors(self._interpolation_logits)
            for _ in query.shape[:-1]:
                interpolation_rates = interpolation_rates.unsqueeze(0)
            while write_probability.dim() < interpolation_rates.dim():
                write_probability = write_probability.unsqueeze(-1)
            write_probability = interpolation_rates * write_probability

            # Compute gates

            erase_gate = _compute_erase_factor(write_probability, erase_probability, self._erase_cap_factor)
            write_gate = _compute_write_factor(write_probability)

        return update_state, (write_gate, erase_gate)

    def reverse_memory(self,
                       update: MemoryData,
                       control_gates: Tuple[torch.Tensor, torch.Tensor],
                       batch_mask: torch.Tensor,
                       memory: MemoryState,
                       ) -> MemoryState:
        """
        Figures out the previous memory state from the
        current and the various update factors.
        :param update: The update state to integrate.
        :param control_gates: The control gates. Write and erase
        :param batch_mask: Shape (...). Indicates whether memory can be updated. True means No.
        :param memory: The current memory state
        :return: The last memory state
        """

        # Perform the interpolation update.
        write_gate, erase_gate = control_gates
        return memory.step_memory_reverse(update, write_gate, erase_gate, batch_mask)

    def advance_memory(self,
                       update: MemoryData,
                       control_gates: Tuple[torch.Tensor, torch.Tensor],
                       batch_mask: torch.Tensor,
                       memory: MemoryState,
                       ) -> MemoryState:
        """
        Commits the computed update into the long term memory.
        Commits using interpolation.
        :param update: The update to integrate
        :param control_gates: The control gates. Write and erase
        :param batch_mask: Shape (...). Indicates whether memory can be updated. True means No.
        :param memory: The current memory
        :return: The updated memory
        """

        write_gate, erase_gate = control_gates
        return memory.step_memory_forward(update, write_gate, erase_gate, batch_mask)


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
            update, control_factors = self.memory_writer.compute_common(tensor, next_memory)

            # Get the original memory state.
            #
            # Make it retain grads so we can get
            # our gradients off it later for the next
            # backwards pass.
            with torch.no_grad():
                original_memory = self.memory_writer.reverse_memory(update, control_factors,
                                                                    batch_mask, next_memory)
            original_memory.setup_for_gradients_()

        # Manually complete the read
        with profiler.record_function("forward pass"):
            next_memory = self.memory_writer.advance_memory(update, control_factors, batch_mask, original_memory)
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
            update, control_factors = self.memory_writer.compute_common(tensor, memory)
            next_memory = self.memory_writer.advance_memory(update, control_factors, batch_mask, memory)
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
                                     cls: Type[ConcreteMemoryUnitProtocol]
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


class GradientTimestepLoss(nn.Module):
    """
    A loss mechanism based on the gradient time
    loss. This will assign the timestep metrics,
    which let us know how far back in time gradients
    could propagate, to soft bins that are then treated
    as histograms.

    We then specify how much mass we
    allow in each bin between thresholds.
    Going outside these thresholds will
    kick on a strong constraint loss.
    """

    def compute_probability_masses(self, timestep_locations: torch.Tensor) -> torch.Tensor:
        """
        Computes the probability masses associated with each bin based on
        a gaussian kernel at that location.
        :param timestep_locations: The timestep locations. Shape (batch_size, num_elements).
        :return: The bin probability masses. Soft. Shape (batch_size, num_bins).
        """
        # Basically, we compute the contribution of each location to each bin in terms of
        # mixture kernels of gaussians.

        bin_distance = self.bin_centers - timestep_locations.unsqueeze(-1)  # (batch_size, num_elements, num_bins)
        gaussian_magnitudes = torch.exp(
            -bin_distance ** 2 / (2 * self.bin_deviation ** 2))  # (batch_size, num_elements, num_bins)
        probability_mass = gaussian_magnitudes.sum(dim=-2)
        return probability_mass

    def compute_loss(self,
                     predicted_distribution: torch.Tensor,
                     ) -> torch.Tensor:
        """
        Computes the loss based on the histogram distributions.
        :param predicted_distribution: Shape (batch_size, num_bins). Predicted distribution. Direct
        :return: Some sort of loss. Details depend on config.
        """

        # Compute the hinge. This will only be active if a bin
        # goes outside it's thresholds

        difference = predicted_distribution - self.target_distribution  # (batch_size, num_bins)
        hinge = torch.clamp_min(difference.abs() - self.target_thresholds, 0.0)  # (batch_size, num_bins)

        # Convert the hinge into an actual loss
        if self.config.loss_type == "linear_threshold":
            loss = hinge
        elif self.config.loss_type == "quadratic_threshold":
            loss = hinge ** 2
        else:
            raise ValueError(f"loss_type {self.config.loss_type} was never recognized")
        loss = self.config.loss_weight * loss.sum()
        loss = loss / self.config.num_bins
        return loss

    def __init__(self, config: GradientTimeLossConfig):
        super().__init__()

        self.bin_edges = torch.linspace(0, 1.0, config.num_bins + 1)
        bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
        bin_deviation = bin_centers[0] - self.bin_edges[0]
        target_distribution = torch.tensor(config.target_distribution)
        target_thresholds = torch.tensor(config.target_thresholds)

        self.register_buffer('bin_centers', bin_centers)
        self.register_buffer('bin_deviation', bin_deviation)
        self.register_buffer('target_distribution', target_distribution)
        self.register_buffer('target_thresholds', target_thresholds)

        self.config = config

    def forward(self, memory: MemoryState) -> torch.Tensor:
        """
        We compute the gradient timestep loss.
        :param memory: The memory state to consider
        :return: A scalar of the loss
        """

        # Get the timestep distance, normalized, and
        # flattened into something in terms of batch, elements

        timestep_distance = memory.normalized_timestep_distance
        timestep_distance = timestep_distance.flatten(1, -1)

        # Generate the target and predicted distributions

        predicted_distribution = self.compute_probability_masses(timestep_distance)
        predicted_distribution /= predicted_distribution.sum(dim=-1, keepdim=True) + 1e-12

        return self.compute_loss(predicted_distribution)


class MemRegularizationLoss(nn.Module):
    """
    One fairly straightforward way the memory system could
    ultimately end up beating the gradient timestep loss
    without actually seeing significant benefit from the
    gradient effects is to simply not erase certain gates,
    or erase them at decreasing rates as time goes by.

    That is basically okay, as that extra capacity can then
    later be used when training with longer sequence.
    However, it can eventually result in a memory system in which
    the magnitudes of the memory grow out of control if the erase gate is
    never turned on again. Instead, the model should prioritize
    writing a tiny bit each time.

    This loss is a simple magnitude based penalty for when
    the memory units start to get very large. It should
    generally be fairly weak, but will eventually have
    something to say if the memory does not behave nicely.

    We do normalize over the number of memory slots, but NOT
    the batches, as with most losses for this project.
    """

    def __init__(self, config: MemRegularizationLossConfig):
        super().__init__()
        self.config = config

    def forward(self, memory: MemoryState) -> torch.Tensor:
        """
        Computes the regularization loss. We normalize by the
        number of elements, but not the number of batches.

        :param memory: The memory state to consider
        :return: The resulting loss.
        """
        all_memory = torch.cat([tensor.view(tensor.size(0), -1) for tensor in memory.memory_tensors.values()], dim=-1)
        num_elements = all_memory.numel() - all_memory.size(0)  # Exclude batch dimension if applicable
        if self.config.magnitude_loss_type == 'l1':
            loss = all_memory.abs().sum()
        elif self.config.magnitude_loss_type == 'l2':
            loss = (all_memory ** 2).sum()
        else:
            raise ValueError(
                f"Unsupported magnitude_loss_type '{self.config.magnitude_loss_type}'. Supported types are 'l1' and 'l2'.")
        loss = loss / (num_elements + 1e-9)
        return loss * self.config.magnitude_loss_weight
