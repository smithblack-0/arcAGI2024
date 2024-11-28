import unittest
import torch
from typing import List, Tuple, Dict
from dataclasses import dataclass
from src.main.arcAGI2024.memory import (
    MemoryState,
    GradientTimeLossConfig,
    MemRegularizationLossConfig,
    LinearMemoryConfig,
    make_memory_unit
)

class TestLinearMemoryUnitContract(unittest.TestCase):
    def setUp(self):
        """
        Set up the configuration and initialize the LinearMemoryUnit using make_memory_unit.
        """
        # Configuration for GradientTimestepLoss
        self.gradient_loss_config = GradientTimeLossConfig(
            num_bins=4,
            z_score=1.0,
            target_distribution=[0.25, 0.25, 0.25, 0.25],
            target_thresholds=[0.1, 0.1, 0.1, 0.1],
            loss_weight=100.0,
            loss_type='quadratic_threshold'
        )

        # Configuration for MemRegularizationLoss
        self.mem_reg_config = MemRegularizationLossConfig(
            magnitude_loss_type='l2',
            magnitude_loss_weight=0.01
        )

        # Configuration for LinearMemory
        self.linear_memory_config = LinearMemoryConfig(
            num_heads=2,
            d_address=5,
            d_memory=8,
            gradient_loss=self.gradient_loss_config,
            mem_regularization_loss=self.mem_reg_config,
            min_write_half_life_init=1.0,
            max_write_half_life_init=100.0,
            erase_epsilon_factor=0.0001,
            linear_activation_kernel=torch.relu
        )

        # Device and dtype
        self.device = torch.device('cpu')
        self.dtype = torch.float32

        # Initialize the LinearMemoryUnit using make_memory_unit
        self.memory_unit = make_memory_unit(
            d_model=16,
            dtype=self.dtype,
            device=self.device,
            config=self.linear_memory_config
        )

        # Define batch size
        self.batch_size = 4

    def test_create_state_with_make_memory_unit(self):
        """
        Verify that make_memory_unit initializes a working memory unit and that create_state can be called correctly.
        """
        # Create state
        batch_shape = [self.batch_size]
        memory_state = self.memory_unit.create_state(batch_shape)

        # Assert that memory_state is an instance of MemoryState
        self.assertIsInstance(memory_state, MemoryState, "create_state did not return a MemoryState instance.")

    def test_forward_with_create_and_reverse(self):
        """
        Test the full cycle: create_state -> forward -> reverse and verify memory restoration.
        """
        # Create initial memory state
        batch_shape = [self.batch_size]
        initial_memory_state = self.memory_unit.create_state(batch_shape)

        # Create a query tensor and batch mask
        d_model = 16
        query = torch.randn(self.batch_size, d_model, dtype=self.dtype, device=self.device)
        batch_mask = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)  # No masking

        # Perform forward pass
        read_output, new_memory_state = self.memory_unit.forward(query, batch_mask, initial_memory_state)

        # Perform reverse pass
        (read_reverse, restored_memory_state), original_memory = self.memory_unit.reverse(
            query, batch_mask, new_memory_state
        )

        # Verify that restored_memory_state matches initial_memory_state
        # Compare metric tensors
        for key in initial_memory_state.metric_tensors:
            initial_tensor = initial_memory_state.metric_tensors[key]
            restored_tensor = original_memory.metric_tensors[key]
            self.assertTrue(
                torch.allclose(initial_tensor, restored_tensor, atol=1e-5),
                f"Metric tensor '{key}' mismatch after reverse pass."
            )

        # Compare memory tensors
        for key in new_memory_state.memory_tensors:
            initial_tensor = new_memory_state.memory_tensors[key]
            restored_tensor = restored_memory_state.memory_tensors[key]
            self.assertTrue(
                torch.allclose(initial_tensor, restored_tensor, atol=1e-5),
                f"Memory tensor '{key}' mismatch after reverse pass."
            )

        # Additionally, verify that read_reverse matches read_output if applicable
        # Depending on implementation, this may or may not hold true
        # Here, we assume that reversing should ideally allow us to retrieve the original read
        # This assertion can be adjusted based on actual behavior
        self.assertTrue(
            torch.allclose(read_output, read_reverse, atol=1e-5),
            "Read output and read reverse do not match."
        )

    def test_gradients_propagate(self):
        """
        Test that gradients propogate along the restored memory
        state into the original memory state.
        :return:
        """

        # Create initial memory state
        batch_shape = [self.batch_size]
        initial_memory_state = self.memory_unit.create_state(batch_shape)

        # Create a query tensor and batch mask
        d_model = 16
        query = torch.randn(self.batch_size, d_model, dtype=self.dtype, device=self.device)
        batch_mask = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)  # No masking

        # Perform forward pass
        read_output, new_memory_state = self.memory_unit.forward(query, batch_mask, initial_memory_state)

        # Perform reverse pass
        (read_reverse, restored_memory_state), original_memory = self.memory_unit.reverse(
            query, batch_mask, new_memory_state
        )

        # propogate gradients
        loss = read_reverse.sum()
        for metric in restored_memory_state.metric_tensors:
            loss += restored_memory_state.metric_tensors[metric].sum()
        loss.backward()

        # check for gradients.
        for key in original_memory.memory_tensors:
            self.assertIsNotNone(original_memory.memory_tensors[key].grad, key)
        for key in original_memory.metric_tensors:
            self.assertIsNotNone(original_memory.metric_tensors[key].grad, key)

    def test_torchscript(self):
        """
        Test that the memory units we are producing can be compiled
        using torch jit.
        """
        # Script memory unit
        memory_unit = torch.jit.script(self.memory_unit)

        # Create initial memory state
        batch_shape = [self.batch_size]
        initial_memory_state = memory_unit.create_state(batch_shape)

        # Create a query tensor and batch mask
        d_model = 16
        query = torch.randn(self.batch_size, d_model, dtype=self.dtype, device=self.device)
        batch_mask = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)  # No masking

        # Perform forward pass
        read_output, new_memory_state = memory_unit.forward(query, batch_mask, initial_memory_state)

        # Perform reverse pass
        (read_reverse, restored_memory_state), original_memory = memory_unit.reverse(
            query, batch_mask, new_memory_state
        )

        # propagate gradients
        loss = read_reverse.sum()
        for metric in restored_memory_state.metric_tensors:
            loss += restored_memory_state.metric_tensors[metric].sum()
        loss.backward()
