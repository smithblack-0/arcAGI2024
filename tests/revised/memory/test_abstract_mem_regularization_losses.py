import unittest
import torch
from typing import List
from src.main.arcAGI2024.memory.base import (
    MemoryState,
    MemoryData,
    GradientTimestepLoss,
    GradientTimeLossConfig,
    MemRegularizationLoss,
    MemRegularizationLossConfig,
)

# Mock MemoryState for testing purposes
class MockMemoryState:
    def __init__(self, normalized_timestep_distance: torch.Tensor):
        self.normalized_timestep_distance = normalized_timestep_distance

class TestGradientTimestepLoss(unittest.TestCase):
    def setUp(self):
        # Create a valid GradientTimeLossConfig
        self.config = GradientTimeLossConfig(
            num_bins=4,
            deviation_factor=1.0,
            target_distribution=[0.25, 0.25, 0.25, 0.25],
            target_thresholds=[0.1, 0.1, 0.1, 0.1],
            loss_weight=10.0,
            loss_type='quadratic_threshold'
        )
        self.loss_module = GradientTimestepLoss(self.config)
        self.batch_size = 2
        self.num_elements = 5

    def test_initialization(self):
        # Test that bin_centers and bin_deviation are computed correctly
        expected_bin_centers = torch.tensor([0.125, 0.375, 0.625, 0.875])
        expected_bin_deviation = 0.125

        self.assertTrue(torch.allclose(self.loss_module.bin_centers, expected_bin_centers))
        self.assertAlmostEqual(self.loss_module.bin_deviation.item(), expected_bin_deviation)

    def test_loss_zero_within_thresholds(self):
        # Mock normalized_timestep_distance that matches target distribution
        timestep_locations = torch.tensor([
            [0.125, 0.375, 0.625, 0.875],
            [0.375, 0.625, 0.875, 0.125]
        ])  # Shape: (batch_size, num_elements)
        memory_state = MockMemoryState(timestep_locations)
        loss = self.loss_module(memory_state)
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_loss_activation_beyond_thresholds(self):
        # Mock normalized_timestep_distance that deviates beyond thresholds
        timestep_locations = torch.tensor([
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0]
        ])  # Extreme values to trigger loss

        memory_state = MockMemoryState(timestep_locations)
        loss = self.loss_module(memory_state)
        self.assertGreater(loss.item(), 0.0)

    def test_invalid_loss_type(self):
        # Attempt to initialize with an invalid loss type
        with self.assertRaises(ValueError):
            invalid_config = GradientTimeLossConfig(
                num_bins=4,
                deviation_factor=1.0,
                target_distribution=[0.25, 0.25, 0.25, 0.25],
                target_thresholds=[0.1, 0.1, 0.1, 0.1],
                loss_weight=10.0,
                loss_type='unsupported_loss_type'
            )
            GradientTimestepLoss(invalid_config)


class TestMemRegularizationLoss(unittest.TestCase):
    def setUp(self):
        # Create a valid MemRegularizationLossConfig
        self.config = MemRegularizationLossConfig(
            magnitude_loss_type='l2',
            magnitude_loss_weight=0.01
        )
        self.loss_module = MemRegularizationLoss(self.config)
        self.batch_size = 3
        self.memory_size = 10

    def test_zero_memory_tensors(self):
        # Provide memory tensors with zeros
        memory_tensors = {
            'memory': torch.zeros(self.batch_size, self.memory_size)
        }
        memory_state = MemoryState({}, memory_tensors, {})
        loss = self.loss_module(memory_state)
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_small_magnitude_tensors(self):
        # Provide memory tensors with small values
        small_value = 0.1
        memory_tensors = {
            'memory': torch.full((self.batch_size, self.memory_size), small_value)
        }
        memory_state = MemoryState({}, memory_tensors, {})
        expected_loss = (small_value ** 2) * self.memory_size / self.memory_size
        loss = self.loss_module(memory_state)
        self.assertAlmostEqual(loss.item(), expected_loss * self.config.magnitude_loss_weight, places=6)

    def test_large_magnitude_tensors(self):
        # Provide memory tensors with large values
        large_value = 10.0
        memory_tensors = {
            'memory': torch.full((self.batch_size, self.memory_size), large_value)
        }
        memory_state = MemoryState({}, memory_tensors, {})
        expected_loss = (large_value ** 2) * self.memory_size / self.memory_size
        loss = self.loss_module(memory_state)
        self.assertAlmostEqual(loss.item(), expected_loss * self.config.magnitude_loss_weight, places=5)

    def test_l1_loss_type(self):
        # Change loss type to 'l1' and test
        self.config.magnitude_loss_type = 'l1'
        self.loss_module = MemRegularizationLoss(self.config)
        value = 5.0
        memory_tensors = {
            'memory': torch.full((self.batch_size, self.memory_size), value)
        }
        memory_state = MemoryState({}, memory_tensors, {})
        expected_loss = value * self.memory_size / self.memory_size
        loss = self.loss_module(memory_state)
        self.assertAlmostEqual(loss.item(), expected_loss * self.config.magnitude_loss_weight, places=6)

    def test_loss_weight_scaling(self):
        # Test that loss scales with loss_weight
        self.config.magnitude_loss_weight = 0.1
        self.loss_module = MemRegularizationLoss(self.config)
        value = 2.0
        memory_tensors = {
            'memory': torch.full((self.batch_size, self.memory_size), value)
        }
        memory_state = MemoryState({}, memory_tensors, {})
        expected_loss = (value ** 2)
        loss = self.loss_module(memory_state)
        self.assertAlmostEqual(loss.item(), expected_loss * self.config.magnitude_loss_weight, places=6)

    def test_invalid_loss_type(self):
        # Attempt to initialize with an invalid loss type
        with self.assertRaises(ValueError):
            invalid_config = MemRegularizationLossConfig(
                magnitude_loss_type='unsupported_loss_type',
                magnitude_loss_weight=0.01
            )
            MemRegularizationLoss(invalid_config)

    def test_empty_memory_tensors(self):
        # Provide empty memory tensors
        memory_tensors = {}
        memory_state = MemoryState({}, memory_tensors, {})
        loss = self.loss_module(memory_state)
        # Loss should be zero since there are no tensors to regularize
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_non_standard_tensor_shapes(self):
        # Provide memory tensors with additional dimensions
        value = 1.0
        memory_tensors = {
            'memory': torch.full((self.batch_size, 2, 5), value)
        }
        memory_state = MemoryState({}, memory_tensors, {})
        num_elements = memory_tensors['memory'].numel() - self.batch_size * 2 * 5
        expected_loss = (value ** 2) * num_elements / num_elements
        loss = self.loss_module(memory_state)
        self.assertAlmostEqual(loss.item(), value ** 2 * self.config.magnitude_loss_weight, places=6)

# Run the tests
if __name__ == '__main__':
    unittest.main()
