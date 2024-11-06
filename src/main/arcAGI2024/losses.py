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
                                                 label_smoothing=label_smoothing_rate)
    def forward(self, logits: torch.Tensor, targets: torch.Tensor, schedule: Optional[float]):
        loss = self.weight*self.cross_entropy(logits, targets)
        if schedule is not None:
            loss = loss*schedule
        return loss

class UniformMemLoss(MemAccessLossInterface):
    """
    Encourages the mem access pattern to occur
    uniformly, with memory write rates being
    pretty much even across the board

    It targets memory access pattern to be
    even across the board.
    """
    def __init__(self,
                 weight: float = 1.0,
                 ):
        super().__init__()
        self.divergence = nn.KLDivLoss()
        self.weight = weight

    def forward(self, write_probability_mass: torch.Tensor, schedule: Optional[float])->torch.Tensor:
        """
        Definition for the memory access loss input and output
        :param write_probability_mass:
        - A feature indicating, per memory slot, how much we wrote to that slot
        - Shape (..., num_memories)
        :param schedule: An optional scheduling float
        :return: A scalar loss.
        """
        # Normalize the write probability mass into a write
        # probability distribution
        write_probability_mass = write_probability_mass/write_probability_mass.sum(dim=-1, keepdim=True)
        write_log_predictions = torch.log(write_probability_mass)

        # Create the uniform target distribution
        num_memories = write_probability_mass.shape[-1]
        target_distribution = torch.full_like(write_log_predictions, 1/num_memories)

        # Create the loss. Scale
        loss = self.weight*self.divergence(write_log_predictions, target_distribution)
        if schedule is not None:
            loss = loss*schedule
        return loss