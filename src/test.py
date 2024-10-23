import torch

def select_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """
    Selection mechanism for the top-p selection type, or nucleus
    selection. In this mode, we collect cases until the probablity
    mass exceeds a threshold, then no more.

    :param logits: The logits to select from. Shape (..., logits)
    :param top_p: The cumulative probability threshold for top-p selection.
    :return: The selected top k mask. Shape (..., logits). Bool of true means selected
    """

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

    ordered_probabilities, sorting_index = probabilities.sort(dim=-1, descending=True)
    cumulative_probabilities = ordered_probabilities.cumsum(dim=-1)
    cumulative_mask = cumulative_probabilities > top_p
    cumulative_mask[..., 0] = True  # First element is always included, to avoid numeric nonsense

    # We now transfer the cumulative mask back into the selection
    # mask, in the designated order.
    selection_mask.scatter_(dim=-1, index=sorting_index, src=cumulative_mask)

    return selection_mask

# Test Script
def test_select_top_p():
    # Example test cases with logits
    logits = torch.tensor([[0.1, 0.2, 3.0, 0.4, 0.5], [2.0, 1.0, 0.3, 0.2, 0.1]])

    # Test different top_p thresholds
    top_p_values = [0.5, 0.7, 0.9]

    for top_p in top_p_values:
        print(f"Testing with top_p = {top_p}")
        selection_mask = select_top_p(logits, top_p)
        print("Logits:\n", logits)
        print("Selection Mask:\n", selection_mask)

# Run the test
if __name__ == "__main__":
    test_select_top_p()
