import torch
import torch.nn.functional as F


def softmax_with_temperature(logits, temperature=1.0):
    """
    Apply softmax to logits with a temperature parameter.

    Args:
        logits (torch.Tensor): The logits to convert to probabilities.
        temperature (float): The temperature to use for scaling.

    Returns:
        torch.Tensor: The probabilities after applying softmax with temperature.
    """
    scaled_logits = logits / temperature
    probabilities = F.softmax(scaled_logits, dim=-1)
    return probabilities


def sample_from_distribution(probabilities, num_samples=1):
    """
    Sample indices from the given probability distributions.

    Args:
        probabilities (torch.Tensor): The probability distributions (batched).
        num_samples (int): Number of samples to draw from each distribution.

    Returns:
        torch.Tensor: The sampled indices.
    """
    return torch.multinomial(probabilities, num_samples, replacement=True)


# Example usage
logits = torch.tensor([[2.0, 1.0, 0.1], [0.5, 1.5, 1.0]])
temperature = 0.7

# Convert logits to probabilities with temperature
probabilities = softmax_with_temperature(logits, temperature)
print("Probabilities:", probabilities)

# Sample from the probability distributions
sampled_indices = sample_from_distribution(probabilities)
print("Sampled indices:", sampled_indices)
