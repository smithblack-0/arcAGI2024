
import torch
from torch import nn
from abc import abstractmethod


def make_top_p_selection_mask(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """
    Selection mechanism for the top-p selection type, or nucleus
    selection. In this mode, we collect cases until the probablity
    mass exceeds a threshold, then no more. We form a mask that shows
    what elements we want to select.

    :param logits: The logits to select from. Shape (..., logits)
    :param top_p: The cumulative probability threshold for top-p selection.
    :return: The selected top k mask. Shape (..., logits). Bool of true means selected
    """
    # Basic sanity check
    if not 1.0 >= top_p >= 0.0:
        raise ValueError(f"Top p should have been between 0 and 1 inclusive. Given was {top_p}")

    # Create the default selection mask
    selection_mask = torch.zeros_like(logits, dtype=bool)

    # Skip further computation and return immediately
    # if top p is set to not catch anything. If not,
    # we activate our logits so we can do probability
    # mass sampling
    if top_p == 0.0:
        return selection_mask

    probabilities = torch.softmax(logits, dim=-1)

    # We must perform nucleus sampling. This is tricky
    # when vectorized. What we must do is sort the probabilities
    # in ascending order, then figure out when the cumulative
    # sum is under a threshold and mask out everything above it
    #
    # However, what we are ACTUALLY doing is figuring out what
    # the mask looks like in the sorted domain, then moving
    # that back into the unsorted mask.
    #
    # We perform a roll here to make sure that we are only considering
    # the probabilities selected so far.

    ordered_probabilities, sorting_index = probabilities.sort(dim=-1, descending=True)
    cumulative_probabilities = ordered_probabilities.cumsum(dim=-1)
    cumulative_probabilities[..., -1] = 0.0
    cumulative_probabilities = cumulative_probabilities.roll(dims=-1, shifts=1)
    cumulative_mask = cumulative_probabilities <= top_p

    # We now transfer the cumulative mask back into the selection
    # mask, in the designated order.
    selection_mask.scatter_(dim=-1, index=sorting_index, src=cumulative_mask)

    return selection_mask


def make_top_k_selection_mask(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """
    Selection mechanism to make a top k mask. Selects the
    top k elements, and returns a mask indicating what
    elements were selected.

    Returns a top k selection mask based on the provided logits
    :param logits: The logits to select a top k from. (..., logits)
    :param top_k: Integer, indicating how many to select.
    :return: The selected top k mask. Shape (..., logits). Bool of true means selected
    """
    if not top_k >= 0:
        raise ValueError(f"Top k should have been greater than or equal to 0. Given was {top_k}")
    if top_k > logits.size(-1):
        top_k = logits.size(-1)

    selection_mask = torch.zeros_like(logits, dtype=bool)
    if top_k > 0:
        # Only bother to actually compute the top k while the
        # mode is active. We get the indexes associated with
        # the top k, and mark those parts of the mask as active
        _, index = torch.topk(logits, k=top_k, dim=-1)
        src = torch.full_like(index, True, dtype=torch.bool)
        selection_mask.scatter_(dim=-1, index=index, src=src)
    return selection_mask


def make_random_selection_mask(logits: torch.Tensor, num: int) -> torch.Tensor:
    """
    Creates a selection mask with num elements selected randomly.

    :param logits: The logits to select a top k from. (..., logits)
    :param num: Integer, indicating how many to randomly select.
    :return: The selected top k mask. Shape (..., logits). Bool of true means selected
    """
    # Basic sanity checking
    if not num >= 0:
        raise ValueError(f"num of selected should have been greater than or equal to 0. Given was {num}")
    if num > logits.size(-1):
        num = logits.size(-1)

    # Create the default selection mask
    selection_mask = torch.zeros_like(logits, dtype=bool)

    # If you are not going to select ANY, just return the default mask
    if num == 0:
        return selection_mask

    # Select a certain number randomly from what remains.
    # We do this by creating a random matrix of the same
    # shape, sorting it, and slicing out the sections needed.
    # This ends up behaving like a vectorized torch.randperm

    random_weights = torch.rand_like(logits)
    randomized_indices = torch.argsort(random_weights, dim=-1)
    randomized_indices = randomized_indices[..., :num]

    src = torch.full_like(randomized_indices, True, dtype=torch.bool)
    selection_mask.scatter_(dim=-1, index=randomized_indices, src=src)

    # Return the selection
    return selection_mask


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
        assert temperature >= 0
        if temperature == 0.0:
            return torch.argmax(logits, dim=-1)

        logits = logits / temperature
        probabilities = torch.softmax(logits, dim=-1)
        return torch.multinomial(probabilities, num_samples=1).squeeze(-1)

class TopKSampling(SamplingInterface):
    """
    Samples from logits by choosing only
    from among the top k elements.
    """
    def __init__(self, num_k: int):
        """
        :param num_k: Number of top k elements
        """
        assert num_k > 0
        assert isinstance(num_k, int)
        super().__init__()
        self.num_k = num_k

    def forward(self, logits: torch.Tensor, temperature: float) -> torch.Tensor:
        """
        Performs top k sampling
        :param logits: Logits collection of shape (..., d_logit)
        :param temperature: The generative temperature
        :return: Tokens chosen. Of shape (...)
        """
        assert temperature >= 0
        if temperature == 0.0:
            return torch.argmax(logits, dim=-1)

        logits = logits / temperature
        selected = make_top_k_selection_mask(logits, self.num_k)
        logits = logits.masked_fill(~selected, -1e9)
        probabilities = torch.softmax(logits, dim=-1)
        return torch.multinomial(probabilities, num_samples=1).squeeze(-1)

class NucleusSampling(SamplingInterface):
    """
    Performs nucleous, or top p,
    sampling against the logits
    """
    def __init__(self, top_p: float):
        """
        :param top_p: The p to collect and consider. Between 0 and 1
        """
        assert top_p >= 0.0
        assert top_p <= 1.0
        assert isinstance(top_p, float)

        super().__init__()
        self.top_p = top_p
    def forward(self, logits: torch.Tensor, temperature: float) -> torch.Tensor:
        """
        Performs nucleous sampling.
        :param logits: Logits collection of shape (..., d_logit)
        :param temperature: The generative temperature
        :return: Tokens chosen. Of shape (...)
        """
        assert temperature >= 0.0
        if temperature == 0.0:
            return torch.argmax(logits, dim=-1)

        logits = logits / temperature
        selection = make_top_p_selection_mask(logits, self.top_p)
        logits = logits.masked_fill(~selection, -1e9)
        probabilities = torch.softmax(logits, dim=-1)
        return torch.multinomial(probabilities, num_samples=1).squeeze(-1)

