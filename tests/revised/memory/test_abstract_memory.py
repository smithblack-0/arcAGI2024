# test_core_helpers.py

import unittest
import torch
from src.main.arcAGI2024.memory.base import (
    _advance_memory,
    _retard_memory,
    _advance_metrics,
    _retard_metrics,
    MemoryState
)

class TestCoreHelpers(unittest.TestCase):
    def setUp(self):
        # Common setup for all tests
        self.batch_size = 2
        self.num_elements = 4
        self.d_model = 3  # Example dimensionality

        # Initialize memory tensors
        self.memory_tensor = torch.randn(self.batch_size, self.num_elements, self.d_model)
        self.update_tensor = torch.randn(self.batch_size, self.num_elements, self.d_model)

        # Initialize probabilities
        self.write_probability = torch.rand(self.batch_size, self.num_elements)
        self.erase_probability = torch.rand(self.batch_size, self.num_elements)

        # Initialize batch_mask
        self.batch_mask = torch.zeros(self.batch_size, dtype=torch.bool)  # No masking

        # Initialize metrics
        self.metrics = {
            "cum_write_mass": torch.zeros(self.batch_size, self.num_elements),
            "cum_erase_mass": torch.zeros(self.batch_size, self.num_elements),
            "effective_write_mass": torch.zeros(self.batch_size, self.num_elements),
            "timestep": torch.zeros(self.batch_size),
            "average_timestep_distance": torch.zeros(self.batch_size, self.num_elements),
        }

    def test_advance_and_retard_memory_inverse(self):
        """
        Test that _advance_memory and _retard_memory are inverses of each other.
        """
        # Advance memory
        advanced_memory = _advance_memory(
            self.memory_tensor,
            self.update_tensor,
            self.write_probability,
            self.erase_probability,
            self.batch_mask
        )

        # Retard memory to get back the original memory tensor
        retarded_memory = _retard_memory(
            advanced_memory,
            self.update_tensor,
            self.write_probability,
            self.erase_probability,
            self.batch_mask
        )

        # Assert that retarded_memory is close to original memory_tensor
        # Increased atol to 1e-5 to accommodate minimal discrepancies from biasing
        self.assertTrue(torch.allclose(retarded_memory, self.memory_tensor, atol=1e-5),
                        "Retarded memory does not match original memory tensor.")

    def test_advance_metrics_and_retard_metrics_inverse(self):
        """
        Test that _advance_metrics and _retard_metrics are inverses of each other.
        """
        # Advance metrics
        advanced_metrics = _advance_metrics(
            self.metrics,
            self.write_probability,
            self.erase_probability,
            self.batch_mask
        )

        # Retard metrics to get back the original metrics
        retarded_metrics = _retard_metrics(
            advanced_metrics,
            self.write_probability,
            self.erase_probability,
            self.batch_mask
        )

        # Assert that retarded_metrics match original metrics
        for key in self.metrics:
            self.assertTrue(torch.allclose(retarded_metrics[key], self.metrics[key], atol=1e-6),
                            f"Retarded metric '{key}' does not match original.")

    def test_edge_cases_zero_probabilities(self):
        """
        Test behavior when write and erase probabilities are zero.
        """
        zero_write = torch.zeros_like(self.write_probability)
        zero_erase = torch.zeros_like(self.erase_probability)

        # Advance memory with zero probabilities
        advanced_memory = _advance_memory(
            self.memory_tensor,
            self.update_tensor,
            zero_write,
            zero_erase,
            self.batch_mask
        )

        # Memory should remain unchanged
        self.assertTrue(torch.allclose(advanced_memory, self.memory_tensor, atol=1e-6),
                        "Memory should remain unchanged when write and erase probabilities are zero.")

    def test_edge_cases_full_masking(self):
        """
        Test behavior when batch_mask is all True (no updates).
        """
        full_mask = torch.ones(self.batch_size, dtype=torch.bool)

        # Advance memory with full masking
        advanced_memory = _advance_memory(
            self.memory_tensor,
            self.update_tensor,
            self.write_probability,
            self.erase_probability,
            full_mask
        )

        # Memory should remain unchanged
        self.assertTrue(torch.allclose(advanced_memory, self.memory_tensor, atol=1e-6),
                        "Memory should remain unchanged when batch_mask is fully True.")

    def test_invalid_shapes_raise_errors(self):
        """
        Test that functions raise errors when input shapes are invalid.
        """
        # Mismatched memory and update tensor shapes
        with self.assertRaises(ValueError):
            _advance_memory(
                self.memory_tensor,
                self.update_tensor[:, :-1, :],  # Mismatched shape
                self.write_probability,
                self.erase_probability,
                self.batch_mask
            )

        # Write probability has too many dimensions
        with self.assertRaises(ValueError):
            _advance_memory(
                self.memory_tensor,
                self.update_tensor,
                self.write_probability.unsqueeze(1),  # Extra dimension
                self.erase_probability,
                self.batch_mask
            )

    def test_metrics_update_correctly(self):
        """
        Test that metrics are updated correctly.
        """
        # Advance metrics
        advanced_metrics = _advance_metrics(
            self.metrics,
            self.write_probability,
            self.erase_probability,
            self.batch_mask
        )

        # Expected updates
        expected_cum_write_mass = self.metrics["cum_write_mass"] + self.write_probability
        expected_cum_erase_mass = self.metrics["cum_erase_mass"] + (self.write_probability * self.erase_probability)
        expected_timestep = self.metrics["timestep"] + self.batch_mask.to(self.write_probability.dtype)
        expected_effective_write_mass = self.metrics["effective_write_mass"] * (1 - self.write_probability * self.erase_probability) + self.write_probability
        expected_average_timestep_distance = (
            self.metrics["average_timestep_distance"] * (1 - self.write_probability * self.erase_probability) +
            self.metrics['timestep'] * self.write_probability * self.erase_probability
        )

        # Assert updates
        self.assertTrue(torch.allclose(advanced_metrics['cum_write_mass'], expected_cum_write_mass, atol=1e-6),
                        "cum_write_mass did not update correctly.")
        self.assertTrue(torch.allclose(advanced_metrics['cum_erase_mass'], expected_cum_erase_mass, atol=1e-6),
                        "cum_erase_mass did not update correctly.")
        self.assertTrue(torch.allclose(advanced_metrics['timestep'], expected_timestep, atol=1e-6),
                        "timestep did not update correctly.")
        self.assertTrue(torch.allclose(advanced_metrics['effective_write_mass'], expected_effective_write_mass, atol=1e-6),
                        "effective_write_mass did not update correctly.")
        self.assertTrue(torch.allclose(advanced_metrics['average_timestep_distance'], expected_average_timestep_distance, atol=1e-6),
                        "average_timestep_distance did not update correctly.")

    def test_retard_memory_handles_negative_values(self):
        """
        Test that _retard_memory correctly handles negative values in memory_tensor and update_tensor.
        """
        # Create memory_tensor and update_tensor with negative values
        self.memory_tensor = -torch.abs(torch.randn(self.batch_size, self.num_elements, self.d_model))
        self.update_tensor = -torch.abs(torch.randn(self.batch_size, self.num_elements, self.d_model))
        self.write_probability = torch.rand(self.batch_size, self.num_elements)
        self.erase_probability = torch.rand(self.batch_size, self.num_elements)

        # Advance memory
        advanced_memory = _advance_memory(
            self.memory_tensor,
            self.update_tensor,
            self.write_probability,
            self.erase_probability,
            self.batch_mask
        )

        # Retard memory
        retarded_memory = _retard_memory(
            advanced_memory,
            self.update_tensor,
            self.write_probability,
            self.erase_probability,
            self.batch_mask
        )

        # Assert that retarded_memory is close to original memory_tensor
        # Increased atol to 1e-5 to account for biasing
        self.assertTrue(torch.allclose(retarded_memory, self.memory_tensor, atol=1e-5),
                        "Retarded memory does not match original memory tensor with negative values.")

    def test_retard_memory_no_nan_values(self):
        """
        Ensure that _retard_memory does not produce NaN values.
        """
        # Advance memory
        advanced_memory = _advance_memory(
            self.memory_tensor,
            self.update_tensor,
            self.write_probability,
            self.erase_probability,
            self.batch_mask
        )

        # Retard memory
        retarded_memory = _retard_memory(
            advanced_memory,
            self.update_tensor,
            self.write_probability,
            self.erase_probability,
            self.batch_mask
        )

        # Assert that retarded_memory does not contain NaNs
        self.assertFalse(torch.isnan(retarded_memory).any(),
                         "Retarded memory contains NaN values.")

    def test_retard_memory_with_high_probabilities(self):
        """
        Test _retard_memory with high write and erase probabilities to ensure numerical stability.
        """
        # Set high write and erase probabilities
        self.write_probability = torch.full((self.batch_size, self.num_elements), 0.99)
        self.erase_probability = torch.full((self.batch_size, self.num_elements), 0.99)

        # Advance memory
        advanced_memory = _advance_memory(
            self.memory_tensor,
            self.update_tensor,
            self.write_probability,
            self.erase_probability,
            self.batch_mask
        )

        # Retard memory
        retarded_memory = _retard_memory(
            advanced_memory,
            self.update_tensor,
            self.write_probability,
            self.erase_probability,
            self.batch_mask
        )

        # Assert that retarded_memory matches original memory_tensor within tolerance
        self.assertTrue(torch.allclose(retarded_memory, self.memory_tensor, atol=1e-4),
                        "Retarded memory does not match original memory tensor with high probabilities.")

    def test_retard_memory_with_batch_mask(self):
        """
        Test that _retard_memory respects the batch_mask, leaving masked batches unchanged.
        """
        # Create a batch_mask where the first batch is masked
        self.batch_mask = torch.tensor([True, False], dtype=torch.bool)

        # Advance memory
        advanced_memory = _advance_memory(
            self.memory_tensor,
            self.update_tensor,
            self.write_probability,
            self.erase_probability,
            self.batch_mask
        )

        # Retard memory
        retarded_memory = _retard_memory(
            advanced_memory,
            self.update_tensor,
            self.write_probability,
            self.erase_probability,
            self.batch_mask
        )

        # The first batch should remain unchanged
        self.assertTrue(torch.allclose(retarded_memory[0], self.memory_tensor[0], atol=1e-5),
                        "Retarded memory did not respect the batch_mask for the first batch.")

        # The second batch should be reverted correctly
        self.assertTrue(torch.allclose(retarded_memory[1], self.memory_tensor[1], atol=1e-5),
                        "Retarded memory did not revert correctly for the second batch.")

if __name__ == '__main__':
    unittest.main()
