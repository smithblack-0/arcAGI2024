"""
Generally, the logic in this section is centered around
testing private helper functions that are used to implement
the core logic.

If the logic going wrong will cause major issues
for the entire model, but it is implementation
dependent and brittle, the test goes here.
"""
import unittest
import torch
from src.main.arcAGI2024.memory.base import (
    _compute_erase_factor,
    _compute_write_factor,
    _step_state_forward,
    _step_state_reverse,
    _advance_memory,
    _retard_memory,
    _advance_metrics,
    _retard_metrics,
    MemoryState
)


class TestMemHelperFunctions(unittest.TestCase):
    """
    Test the memory helper functions for stability
    and usefulness
    """
    def setUp(self):
        # Common setup for all tests
        self.batch_size = 2
        self.num_elements = 4
        self.d_model = 3  # Example dimensionality

        # Initialize memory tensors
        self.memory_tensor = torch.randn(self.batch_size, self.num_elements, self.d_model)
        self.update_tensor = torch.randn(self.batch_size, self.num_elements, self.d_model)

        # Initialize probabilities
        self.write_gate = torch.rand(self.batch_size, self.num_elements)
        self.erase_gate = torch.rand(self.batch_size, self.num_elements)*0.99

    def test_forward_reverse_inverses(self):
        """
        Test that going forward, then reverse, actually results in inverses
        """
        batch_mask = torch.tensor([False, False])
        memory = _advance_memory(self.memory_tensor, self.update_tensor, self.write_gate, self.erase_gate, batch_mask)
        memory = _retard_memory(memory, self.update_tensor, self.write_gate, self.erase_gate, batch_mask)
        self.assertTrue(torch.allclose(memory, self.memory_tensor, atol=1e-5),
                        "Retarded memory does not match original memory tensor.")

    def test_forward_reverse_inverses_at_edge_cases(self):
        """
        Test that when the erase gate is nearly saturated, we still
        can make an inverse.
        """
        erase_gate = torch.ones([self.batch_size, self.num_elements])*0.99
        batch_mask = torch.tensor([False, False])
        memory = _advance_memory(self.memory_tensor, self.update_tensor, self.write_gate, erase_gate, batch_mask)
        memory = _retard_memory(memory, self.update_tensor, self.write_gate, erase_gate, batch_mask)
        self.assertTrue(torch.allclose(memory, self.memory_tensor, atol=1e-5),
                        "Retarded memory does not match original memory tensor.")

    def test_absolute_error(self):
        """
        Test that small pertubations at a nearly saturated erase gate only produce a large
        percent error, but not a large absolute error. So long as this is the case,
        the model is numerically very stable, as dense layers care about absolute not
        relative error.
        """
        erase_gate = torch.ones([self.batch_size, self.num_elements])*0.99
        batch_mask = torch.tensor([False, False])
        pertubation = 1e-5
        memory = _advance_memory(self.memory_tensor, self.update_tensor, self.write_gate, erase_gate, batch_mask)
        memory = _retard_memory(memory, self.update_tensor, self.write_gate, erase_gate-pertubation, batch_mask)
        self.assertTrue(torch.allclose(memory, self.memory_tensor, atol=1e-5),
                        "Retarded memory does not match original memory tensor.")

    def test_masking(self):
        """
        Test that the masking mechanism works.
        """
        batch_mask = torch.tensor([True, True])
        memory = _advance_memory(self.memory_tensor, self.update_tensor, self.write_gate, self.erase_gate, batch_mask)

        self.assertTrue(torch.allclose(memory, self.memory_tensor, atol=1e-5),
                        " memory does not match original memory tensor.")

        metrics = _retard_memory(self.memory_tensor, self.update_tensor, self.write_gate, self.erase_gate,
                                 batch_mask)
        self.assertTrue(torch.allclose(memory, self.memory_tensor, atol=1e-5),
                        " memory does not match original memory tensor.")


class TestMetricHelperFunctions(unittest.TestCase):
    """
    Test the metrics helper functions for functionality
    and stability. In particular, test they correctly invert
    the metrics.
    """
    def setUp(self):
        # Common setup for all tests
        self.batch_size = 2
        self.num_elements = 4
        self.d_model = 3  # Example dimensionality

        # Initialize gates
        self.write_gate = torch.rand(self.batch_size, self.num_elements)
        self.erase_gate = torch.rand(self.batch_size, self.num_elements)*0.99

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

    def test_advance_retard_metrics(self):
        """
        Test advancing then retarding the metrics, see if we get the same
        thing back.
        """

        metrics = _advance_metrics(self.metrics, self.write_gate, self.erase_gate, self.batch_mask)
        metrics = _retard_metrics(metrics, self.write_gate, self.erase_gate, self.batch_mask)
        for name in metrics.keys():
            expected = self.metrics[name]
            actual = metrics[name]
            self.assertTrue(torch.allclose(expected, actual, atol=1e-5), f"{name} were not the same")

    def test_advance_retard_extremes(self):
        """
        Test advancing then retarding the metrics with a pretty saturated
        erase gate.
        """
        erase_gate = torch.ones([self.batch_size, self.num_elements])*0.99
        batch_mask = torch.tensor([False, False])
        metrics = _advance_metrics(self.metrics, self.write_gate, erase_gate, batch_mask)
        metrics = _retard_metrics(metrics, self.write_gate, erase_gate, self.batch_mask)
        for name in metrics.keys():
            expected = self.metrics[name]
            actual = metrics[name]
            self.assertTrue(torch.allclose(expected, actual, atol=1e-5), f"{name} were not the same")

    def test_numeric_stability(self):
        """
        Test that under the duress of a very saturated erase gate, permutations
        due to errors have a minimal effect on the absolute error.
        """
        erase_gate = torch.ones([self.batch_size, self.num_elements])*0.99
        batch_mask = torch.tensor([False, False])
        pertubation = 1e-5
        metrics = _advance_metrics(self.metrics, self.write_gate, erase_gate, batch_mask)
        metrics = _retard_metrics(metrics, self.write_gate, erase_gate+pertubation, self.batch_mask)
        for name in metrics.keys():
            expected = self.metrics[name]
            actual = metrics[name]
            self.assertTrue(torch.allclose(expected, actual, atol=1e-3), f"{name} were not the same")

    def test_masking(self):
        """
        Test that we respond correctly to masking
        """
        batch_mask = torch.tensor([True, True])

        metrics = _advance_metrics(self.metrics, self.write_gate, self.erase_gate, batch_mask)
        for name in metrics.keys():
            expected = self.metrics[name]
            actual = metrics[name]
            self.assertTrue(torch.allclose(expected, actual, atol=1e-5), f"{name} were not the same")

        metrics = _retard_metrics(self.metrics, self.write_gate, self.erase_gate, batch_mask)
        for name in metrics.keys():
            expected = self.metrics[name]
            actual = metrics[name]
            self.assertTrue(torch.allclose(expected, actual, atol=1e-5), f"{name} were not the same")

    def test_cumulative_metrics_update(self):
        """
        Test that cum_write_mass and cum_erase_mass are correctly updated based on write_gate and erase_gate.
        """
        # Define specific write and erase gates
        write_gate = torch.tensor([[0.1, 0.2, 0.3, 0.4],
                                   [0.5, 0.6, 0.7, 0.8]])
        erase_gate = torch.tensor([[0.05, 0.05, 0.05, 0.05],
                                   [0.05, 0.05, 0.05, 0.05]])
        batch_mask = torch.tensor([False, False])

        # Initialize metrics
        initial_metrics = {
            "cum_write_mass": torch.zeros(self.batch_size, self.num_elements),
            "cum_erase_mass": torch.zeros(self.batch_size, self.num_elements),
            "effective_write_mass": torch.zeros(self.batch_size, self.num_elements),
            "timestep": torch.zeros(self.batch_size),
            "average_timestep_distance": torch.zeros(self.batch_size, self.num_elements),
        }

        # Advance metrics
        advanced_metrics = _advance_metrics(initial_metrics, write_gate, erase_gate, batch_mask)

        # Expected cum_write_mass and cum_erase_mass
        expected_cum_write_mass = write_gate
        expected_cum_erase_mass = erase_gate

        self.assertTrue(torch.allclose(advanced_metrics["cum_write_mass"], expected_cum_write_mass, atol=1e-5),
                        "cum_write_mass not updated correctly.")
        self.assertTrue(torch.allclose(advanced_metrics["cum_erase_mass"], expected_cum_erase_mass, atol=1e-5),
                        "cum_erase_mass not updated correctly.")


    def test_average_timestep_distance_update(self):
        """
        Test that average_timestep_distance is updated correctly based on erase_gate and timestep.
        """
        write_gate = torch.tensor([[0.5, 0.5, 0.5, 0.5],
                                   [0.5, 0.5, 0.5, 0.5]])
        erase_gate = torch.tensor([[0.5, 0.5, 0.5, 0.5],
                                   [0.5, 0.5, 0.5, 0.5]])
        batch_mask = torch.tensor([False, False])

        initial_metrics = {
            "cum_write_mass": torch.zeros(self.batch_size, self.num_elements),
            "cum_erase_mass": torch.zeros(self.batch_size, self.num_elements),
            "effective_write_mass": torch.zeros(self.batch_size, self.num_elements),
            "timestep": torch.tensor([1.0, 1.0]),
            "average_timestep_distance": torch.zeros(self.batch_size, self.num_elements),
        }

        # Advance metrics
        advanced_metrics = _advance_metrics(initial_metrics, write_gate, erase_gate, batch_mask)

        # Compute expected effective_write_mass
        # Assuming _step_state_forward computes: s_new = s_old * erase_gate + write_gate * 1
        expected_effective_write_mass = initial_metrics["effective_write_mass"] * erase_gate + write_gate * 1.0

        self.assertTrue(torch.allclose(advanced_metrics["effective_write_mass"], expected_effective_write_mass, atol=1e-5),
                        "effective_write_mass not updated correctly.")

        # Compute expected average_timestep_distance
        # Assuming it updates as effective_write_mass / (timestep + epsilon)
        expected_average_timestep_distance = expected_effective_write_mass / (
                    initial_metrics["timestep"].unsqueeze(-1) + 1e-9)

        self.assertTrue(
            torch.allclose(advanced_metrics["average_timestep_distance"], expected_average_timestep_distance, atol=1e-5),
            "average_timestep_distance not updated correctly.")
    def test_effective_write_mass_calculation(self):
        """
        Test that effective_write_mass is updated correctly.
        """
        write_gate = torch.tensor([[0.2, 0.3, 0.4, 0.1],
                                   [0.5, 0.6, 0.7, 0.8]])
        erase_gate = torch.tensor([[0.1, 0.2, 0.1, 0.3],
                                   [0.4, 0.1, 0.2, 0.1]])
        batch_mask = torch.tensor([False, False])

        initial_metrics = {
            "cum_write_mass": torch.zeros(self.batch_size, self.num_elements),
            "cum_erase_mass": torch.zeros(self.batch_size, self.num_elements),
            "effective_write_mass": torch.tensor([[0.5, 0.5, 0.5, 0.5],
                                                 [0.5, 0.5, 0.5, 0.5]]),
            "timestep": torch.tensor([1.0, 1.0]),
            "average_timestep_distance": torch.zeros(self.batch_size, self.num_elements),
        }

        metrics = _advance_metrics(initial_metrics, write_gate, erase_gate, batch_mask)

        expected_effective_write_mass = initial_metrics["effective_write_mass"] * erase_gate + write_gate
        self.assertTrue(torch.allclose(metrics["effective_write_mass"], expected_effective_write_mass, atol=1e-5),
                        "effective_write_mass not updated correctly based on write_gate and erase_gate.")

    def test_timestep_increment(self):
        """
        Test that timestep increments correctly when batch_mask is False and remains unchanged when True.
        """
        # Test Case 1: batch_mask is False
        initial_metrics = self.metrics.copy()
        write_gate = torch.rand(self.batch_size, self.num_elements)
        erase_gate = torch.rand(self.batch_size, self.num_elements)*0.99
        batch_mask = torch.tensor([False, False])

        metrics = _advance_metrics(initial_metrics, write_gate, erase_gate, batch_mask)

        expected_timestep = initial_metrics["timestep"] + 1
        self.assertTrue(torch.allclose(metrics["timestep"], expected_timestep, atol=1e-5),
                        "timestep did not increment correctly when batch_mask is False.")

        # Test Case 2: batch_mask is True
        initial_metrics = metrics.copy()
        batch_mask = torch.tensor([True, True])

        metrics = _advance_metrics(initial_metrics, write_gate, erase_gate, batch_mask)

        expected_timestep = initial_metrics["timestep"]  # Should remain unchanged
        self.assertTrue(torch.allclose(metrics["timestep"], expected_timestep, atol=1e-5),
                        "timestep incorrectly incremented when batch_mask is True.")

