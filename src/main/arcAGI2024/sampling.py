
import torch
from torch import nn
from abc import abstractmethod
class SamplingInterface(nn.Module):
    """
    Any sampling mechanism compatible
    with the model will be compatible
    with this abstract spec.
    """
    @abstractmethod
    def forward(self, logits: torch.Tensor, temperature: float)->torch.Tensor:
        """
        Definition for the sampling input and output
        :param logits: Logits collection of shape (..., d_logit)
        :param temperature: The generative temperature
        :return: Tokens chosen. Of shape (...)
        """
class TopLogitSampling(SamplingInterface):
    """
    Selects the highest-probability token
    (greedy sampling).
    """

    def forward(self, logits: torch.Tensor, temperature: float) -> torch.Tensor:
        return torch.argmax(logits, dim=-1)


class DefaultSampling(SamplingInterface):
    """
    Samples from all logits using multinomial
    distribution (standard sampling).
    """
    def forward(self, logits: torch.Tensor, temperature: float) -> torch.Tensor:
        logits = logits / temperature
        probabilities = torch.softmax(logits, dim=-1)
        return torch.multinomial(probabilities, num_samples=1).squeeze(-1)
