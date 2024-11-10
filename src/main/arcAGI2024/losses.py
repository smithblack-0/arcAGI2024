"""
Defines some default loss mechanisms
"""
from typing import Optional
from abc import ABC, abstractmethod

import torch
from torch import nn




class MainLossInterface(nn.Module):
    """
    Any main loss mechanism, which
    provides loss based on the model,
    must implement this
    """
    @abstractmethod
    def forward(self, logits: torch.Tensor, targets: torch.Tensor,  schedule: Optional[torch.Tensor])->torch.Tensor:
        """
        Definition for the main loss input and output
        :param logits: The logits feature to take loss with. Shape (..., d_logits)
        :param target: The target. Integers. Shape (...).
        :param schedule: An optional scheduling control feature. We can adjust loss strength here
        :return: The loss. A scalar.
        """

class MemAccessLossInterface(nn.Module):
    """
    The memory access loss mechanism
    must implement this interface.
    """
    @abstractmethod
    def forward(self, write_probability_mass: torch.Tensor, schedule: Optional[float])->torch.Tensor:
        """
        Definition for the memory access loss input and output
        :param write_probability_mass:
        - A feature indicating, per memory slot, how much we wrote to that slot
        - Shape (..., num_memories)
        :param schedule: An optional scheduling float
        :return: A scalar loss.
        """
class CrossEntropyLoss(MainLossInterface):
    """
    A fairly normal cross entropy loss
    function. If the schedule parameter
    is defined, we multiply by it before
    returning.
    """
    def __init__(self,
                 padding_token_id: int,
                 weight: float = 1.0,
                 label_smoothing_rate: float = 0.1
                 ):
        """

        :param padding_token_id: Padding token to ignore
        :param weight: A static weight to scale by
        :param label_smoothing_rate: The label smoothing rate
        """
        super().__init__()
        self.weight = weight
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=padding_token_id,
                                                 label_smoothing=label_smoothing_rate,
                                                 reduction="sum")
    def forward(self, logits: torch.Tensor, targets: torch.Tensor, schedule: Optional[float]):
        loss = self.weight*self.cross_entropy(logits, targets)
        if schedule is not None:
            loss = loss*schedule
        return loss


def kl_divergence(p: torch.Tensor, q: torch.Tensor, epsilon=1e-10) -> torch.Tensor:
    """
    Computes the Kullback-Leibler divergence between two probability distributions P and Q.

    :param p: Tensor of the true distribution (should sum to 1 along the last dimension).
    :param q: Tensor of the approximate distribution (should sum to 1 along the last dimension).
    :param epsilon: Small constant to prevent division by zero and log of zero.
    :return: Tensor representing the KL divergence between P and Q.
    """
    # Ensure neither P nor Q has zero entries (prevents NaN in log and division)
    p = p.clamp(min=epsilon)
    q = q.clamp(min=epsilon)

    # Compute KL divergence
    kl_div = p * torch.log(p / q)

    # Sum over the last dimension
    return kl_div.sum(dim=-1)


class UniformMemLoss(MemAccessLossInterface):
    """
    Encourages the memory access pattern to occur uniformly,
    with memory write rates being even across all slots.
    """

    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def forward(self, write_probability_mass: torch.Tensor, schedule: Optional[float] = None) -> torch.Tensor:
        """
        Defines the memory access loss to promote uniform memory access.

        :param write_probability_mass: A tensor indicating how much was written to each memory slot.
                                       Shape (..., num_memories)
        :param schedule: An optional scheduling scalar.
        :return: A scalar loss.
        """
        # Normalize the write probability mass into a probability distribution
        write_probability_mass = write_probability_mass / (write_probability_mass.sum(dim=-1, keepdim=True) + 1e-4)

        # Compute KL divergence against a uniform target distribution
        loss = self.weight * self.kl_divergence(write_probability_mass)

        # Apply scheduling if provided
        if schedule is not None:
            loss = loss * schedule
        return loss

    def kl_divergence(self, write_prob: torch.Tensor) -> torch.Tensor:
        """
        Helper method to compute KL divergence between the write distribution
        and a uniform distribution.

        :param write_prob: The write probability distribution tensor.
        :return: KL divergence loss.
        """
        # Calculate log of the write probability distribution
        write_log_predictions = torch.log(write_prob + 1e-4)

        # Create the uniform target distribution
        num_memories = write_prob.shape[-1]
        target_distribution = torch.full_like(write_prob, 1 / num_memories)

        # Compute KL divergence
        kl_divergence = write_prob * (write_log_predictions - torch.log(target_distribution))
        return kl_divergence.sum(dim=-1).mean()  # Sum over the last dimension and average
