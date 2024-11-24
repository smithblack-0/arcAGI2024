
import torch
import torch.jit
import timeit

def _numeric_stabilized_division(numerator: torch.Tensor,
                                 denominator: torch.Tensor,
                                 epsilon: float = 1e-9) -> torch.Tensor:
    """
    Attempts to perform a numerically stabilized division using logarithms.

    While this will not in and of itself protect from division by zero,
    it will ensure that near zero the error is kept as low as is feasible.

    :param numerator: The numerator to consider when performing the division
    :param denominator: The denominator
    :param epsilon: When needed, the epsilon for logarithm duty
    :return: The result of performing the division.
    """

    # Split the numerator and denominator up
    # into a signed component and a magnitude component.

    sign = torch.sign(numerator) * torch.sign(denominator)
    numerator_magnitude = torch.abs(numerator)
    denominator_magnitude = torch.abs(denominator)

    # Bias the numerator magnitude, and the denominator magnitude.
    #
    # By adding a small positive term to the numerator we can ensure that
    # we never feed in a zero to the logarithm, even if the logarithm is itself
    # zero.
    #
    # We also, by the relation (x+epsilon)/y = x/y + epsilon/y,

    numerator_magnitude = numerator_magnitude + epsilon
    bias_correction_term = epsilon / denominator_magnitude

    # Perform the division process.
    log_division = torch.log(numerator_magnitude)
    log_division -= torch.log(denominator_magnitude)
    output_magnitudes = torch.exp(log_division)

    # Compensate for the biasing process
    output_magnitudes = output_magnitudes - bias_correction_term

    # Return the output
    return sign * output_magnitudes

def direct_division(numerator: torch.Tensor,
                    denominator: torch.Tensor) -> torch.Tensor:
    """
    Performs direct element-wise division.

    :param numerator: The numerator tensor.
    :param denominator: The denominator tensor.
    :return: The result of numerator / denominator.
    """
    return numerator / denominator

def main():
    # TorchScript both functions for optimized performance
    scripted_stabilized_division = torch.jit.script(_numeric_stabilized_division)
    scripted_direct_division = torch.jit.script(direct_division)

    # Generate random test data
    torch.manual_seed(42)  # For reproducibility
    # Create large tensors to better observe performance differences
    numerator = torch.randn(10000)
    denominator = torch.randn(10000) + 1e-3  # Add epsilon to avoid exact zeros

    # Define wrapper functions for benchmarking
    def test_stabilized_division():
        return scripted_stabilized_division(numerator, denominator)

    def test_direct_division():
        return scripted_direct_division(numerator, denominator)

    # Number of iterations for benchmarking
    num_runs = 1000

    # Warm-up runs (optional but can help with more consistent timing)
    test_stabilized_division()
    test_direct_division()

    # Benchmark the stabilized division
    stabilized_time = timeit.timeit(test_stabilized_division, number=num_runs)
    # Benchmark the direct division
    direct_time = timeit.timeit(test_direct_division, number=num_runs)

    # Output the results
    print(f"Benchmark Results over {num_runs} runs:")
    print(f"Stabilized Division Time: {stabilized_time:.6f} seconds")
    print(f"Direct Division Time:      {direct_time:.6f} seconds")

    # Optionally, verify that both methods produce similar results
    stabilized_result = scripted_stabilized_division(numerator, denominator)
    direct_result = scripted_direct_division(numerator, denominator)
    difference = torch.abs(stabilized_result - direct_result).max().item()
    print(f"Maximum difference between methods: {difference:.6e}")

if __name__ == "__main__":
    main()
