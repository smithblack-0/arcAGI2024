# Unit test suite for the memory framework classes.
import unittest
import torch
from torch import nn
from typing import Tuple, Dict, Any, Type, List
from dataclasses import dataclass

# Import the abstract classes from your module.
# Adjust the import statements based on your project structure.
# For example, if your module is accessible via PYTHONPATH, you can import directly.
from src.main.arcAGI2024.memory.base import (
    AbstractMemoryConfig,
    MemoryState,
    AbstractCreateState,
    AbstractReadMemory,
    AbstractWriteMemory,
    AbstractMemoryUnit,
    register_concrete_implementation,
    make_memory_unit,
    concrete_classes_registry,
)

# Define concrete implementations for testing purposes.

@dataclass
class ConcreteMemoryConfig(AbstractMemoryConfig):
    """
    A concrete implementation of AbstractMemoryConfig for testing.
    """
    max_interpolation_factor: float = 0.99
    min_write_half_life_init: float = 1.0
    max_write_half_life_init: float = 10.0

    @property
    def interpolation_factor_shapes(self) -> torch.Size:
        # For testing purposes, we'll use a simple shape
        return torch.Size([1])

class ConcreteCreateState(AbstractCreateState):
    def setup_state(self, batch_shape: List[int]) -> MemoryState:
        # Initialize persistent_state with 'cum_write_mass' and 'timestep'
        persistent_state = {
            'cum_write_mass': torch.zeros(batch_shape, device=self.device, dtype=self.dtype),
            'timestep': torch.zeros(batch_shape, device=self.device, dtype=self.dtype),
        }
        # Initialize interpolation_state with 'memory_tensor' and 'running_distance'
        interpolation_state = {
            'memory_tensor': torch.zeros(batch_shape + torch.Size([10]), device=self.device, dtype=self.dtype),
            'running_distance': torch.zeros(batch_shape, device=self.device, dtype=self.dtype)
        }
        return MemoryState(persistent_state, interpolation_state)

class ConcreteReadMemory(AbstractReadMemory):
    def forward(self, query: torch.Tensor, memory: MemoryState) -> torch.Tensor:
        # For testing, simply return the memory tensor
        interpolation_state = memory.get_interpolation_states()
        return interpolation_state['memory_tensor']

class ConcreteWriteMemory(AbstractWriteMemory):
    def _compute_common(self, query: torch.Tensor, persistent_state: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        # For testing, create a simple update and write probability
        update = {'memory_tensor': query}
        write_probability = torch.sigmoid(query.mean(dim=-1, keepdim=False))
        return update, write_probability

class ConcreteMemoryUnit(AbstractMemoryUnit):
    def __init__(self, d_model: int, dtype: torch.dtype, device: torch.device, config: AbstractMemoryConfig):
        create_state_unit = ConcreteCreateState(dtype, device)
        read_unit = ConcreteReadMemory(dtype, device)
        write_unit = ConcreteWriteMemory(dtype, device, config)
        super().__init__(create_state_unit, read_unit, write_unit)

# Register the concrete implementation
register_concrete_implementation(ConcreteMemoryConfig, ConcreteMemoryUnit)

# Now, define the unit test suite.

class TestMemoryFramework(unittest.TestCase):
    def setUp(self):
        # Common setup for tests
        self.d_model = 10
        self.dtype = torch.float32
        self.device = torch.device('cpu')
        self.config = ConcreteMemoryConfig()
        self.memory_unit = make_memory_unit(self.d_model, self.dtype, self.device, self.config)
        self.batch_shape = torch.Size([2, 3])  # Example batch shape

    def test_memory_state_initialization(self):
        memory_state = self.memory_unit.create_state(self.batch_shape)
        # Check that the memory state is properly initialized
        self.assertIn('cum_write_mass', memory_state.get_persistent_state())
        self.assertIn('timestep', memory_state.get_persistent_state())
        self.assertIn('running_distance', memory_state.get_interpolation_states())
        self.assertIn('memory_tensor', memory_state.get_interpolation_states())
        self.assertEqual(memory_state.get_persistent_state()['cum_write_mass'].shape, self.batch_shape)
        self.assertEqual(memory_state.get_persistent_state()['timestep'].shape, self.batch_shape)
        self.assertEqual(memory_state.get_interpolation_states()['running_distance'].shape, self.batch_shape)
        self.assertEqual(memory_state.get_interpolation_states()['memory_tensor'].shape, self.batch_shape + torch.Size([10]))

    def test_forward_pass(self):
        memory_state = self.memory_unit.create_state(self.batch_shape)
        input_tensor = torch.randn(self.batch_shape + torch.Size([self.d_model]), dtype=self.dtype, device=self.device)
        batch_mask = torch.zeros(self.batch_shape, dtype=torch.bool, device=self.device)
        output_tensor, next_memory_state = self.memory_unit.forward(input_tensor, batch_mask, memory_state)
        # Check output shapes
        self.assertEqual(output_tensor.shape, self.batch_shape + torch.Size([10]))
        # Ensure memory state has been updated
        self.assertFalse(torch.equal(memory_state.get_interpolation_states()['memory_tensor'],
                                     next_memory_state.get_interpolation_states()['memory_tensor']))
        # Check that 'timestep' and 'cum_write_mass' have been updated
        self.assertTrue(torch.equal(next_memory_state.timestep, memory_state.timestep + batch_mask))

    def test_reverse_pass(self):
        memory_state = self.memory_unit.create_state(self.batch_shape)
        input_tensor = torch.randn(self.batch_shape + torch.Size([self.d_model]), dtype=self.dtype, device=self.device)
        batch_mask = torch.zeros(self.batch_shape, dtype=torch.bool, device=self.device)

        # Perform forward pass to get next memory state
        _, next_memory_state = self.memory_unit.forward(input_tensor, batch_mask, memory_state)

        # Perform reverse pass
        (output_tensor, restored_memory_state), original_memory = self.memory_unit.reverse(input_tensor, batch_mask, next_memory_state)

        # Check that the restored memory matches the original
        for key in memory_state.get_interpolation_states():
            original = memory_state.get_interpolation_states()[key]
            derived = original_memory.get_interpolation_states()[key]
            self.assertTrue(torch.allclose(original, derived, atol=1e-6))
        for key in memory_state.get_persistent_state():
            original = next_memory_state.get_persistent_state()[key]
            derived = original_memory.get_persistent_state()[key]
            self.assertTrue(torch.allclose(original, derived, atol=1e-6))

        # Check that the next memory and derived next memory match
        #
        # In other words, check output matches that from forward pass
        for key in next_memory_state.get_interpolation_states():
            original = next_memory_state.get_interpolation_states()[key]
            derived = restored_memory_state.get_interpolation_states()[key]
            self.assertTrue(torch.allclose(original, derived, atol=1e-6))
        for key in next_memory_state.get_persistent_state():
            original = next_memory_state.get_persistent_state()[key]
            derived = restored_memory_state.get_persistent_state()[key]
            self.assertTrue(torch.allclose(original, derived, atol=1e-6))


    def test_memory_read(self):
        memory_state = self.memory_unit.create_state(self.batch_shape)
        input_tensor = torch.randn(self.batch_shape + torch.Size([self.d_model]), dtype=self.dtype, device=self.device)
        # Read from memory
        read_output = self.memory_unit.memory_reader(input_tensor, memory_state)
        # Check output shape
        self.assertEqual(read_output.shape, self.batch_shape + torch.Size([10]))
        # Since memory_tensor is initialized to zeros, read_output should be zeros
        self.assertTrue(torch.allclose(read_output, torch.zeros_like(read_output), atol=1e-6))

    def test_memory_write(self):
        memory_state = self.memory_unit.create_state(self.batch_shape)
        input_tensor = torch.randn(self.batch_shape + torch.Size([self.d_model]), dtype=self.dtype, device=self.device)
        # Compute update and write factor
        update, write_factor = self.memory_unit.memory_writer.compute_common(input_tensor, memory_state)
        # Advance memory
        batch_mask = torch.zeros(self.batch_shape, dtype=torch.bool, device=self.device)
        next_memory_state = self.memory_unit.memory_writer.advance_memory(update, write_factor, batch_mask, memory_state)
        # Check that memory has been updated
        self.assertFalse(torch.equal(memory_state.get_interpolation_states()['memory_tensor'],
                                     next_memory_state.get_interpolation_states()['memory_tensor']))
        # Verify the updated memory tensor
        expected_memory = memory_state.get_interpolation_states()['memory_tensor'] * (1 - write_factor) + input_tensor * write_factor
        self.assertTrue(torch.allclose(next_memory_state.get_interpolation_states()['memory_tensor'], expected_memory, atol=1e-6))
        # Check that 'cum_write_mass' and 'timestep' have been updated
        expected_cum_write_mass = memory_state.cum_write_mass + write_factor
        expected_timestep = memory_state.timestep + batch_mask
        self.assertTrue(torch.equal(next_memory_state.cum_write_mass, expected_cum_write_mass))
        self.assertTrue(torch.equal(next_memory_state.timestep, expected_timestep))

    def test_registry_mechanism(self):
        # Ensure that the concrete class is registered
        self.assertIn(ConcreteMemoryConfig, concrete_classes_registry)
        # Create a memory unit using the registry
        memory_unit = make_memory_unit(self.d_model, self.dtype, self.device, self.config)
        self.assertIsInstance(memory_unit, ConcreteMemoryUnit)

    def test_interpolation_factor_initialization(self):
        write_unit = self.memory_unit.memory_writer
        # Check that interpolation logits are initialized
        self.assertIsNotNone(write_unit._interpolation_logits)
        # Check the shape of the interpolation logits
        expected_shape = self.config.interpolation_factor_shapes
        self.assertEqual(write_unit._interpolation_logits.shape, expected_shape)

    def test_compute_interpolation_factors(self):
        write_unit = self.memory_unit.memory_writer
        interpolation_factors = write_unit._compute_interpolation_factors(write_unit._interpolation_logits)
        # Check that interpolation factors are between 0 and 1
        self.assertTrue(torch.all(interpolation_factors >= 0))
        self.assertTrue(torch.all(interpolation_factors <= 1))

    def test_device_and_dtype(self):
        # Check that all tensors are on the correct device and dtype
        memory_state = self.memory_unit.create_state(self.batch_shape)
        for tensor in memory_state.get_persistent_state().values():
            self.assertEqual(tensor.device, self.device)
            self.assertEqual(tensor.dtype, self.dtype)
        for tensor in memory_state.get_interpolation_states().values():
            self.assertEqual(tensor.device, self.device)
            self.assertEqual(tensor.dtype, self.dtype)

    def test_batch_mask_effect(self):
        memory_state = self.memory_unit.create_state(self.batch_shape)
        input_tensor = torch.randn(self.batch_shape + torch.Size([self.d_model]), dtype=self.dtype, device=self.device)
        # Create a batch mask that masks out the first element
        batch_mask = torch.zeros(self.batch_shape, dtype=torch.bool, device=self.device)
        batch_mask[0, 0] = True  # Mask out the first element
        # Perform forward pass
        output_tensor, next_memory_state = self.memory_unit.forward(input_tensor, batch_mask, memory_state)
        # Check that the memory at the masked position has not changed
        original_memory = memory_state.get_interpolation_states()['memory_tensor'][0, 0]
        updated_memory = next_memory_state.get_interpolation_states()['memory_tensor'][0, 0]
        self.assertTrue(torch.equal(original_memory, updated_memory))
        # Additionally, verify that 'cum_write_mass' and 'timestep' are updated correctly for masked and unmasked positions
        # Masked position: cum_write_mass and timestep should remain unchanged
        self.assertTrue(torch.equal(next_memory_state.cum_write_mass[0, 0], memory_state.cum_write_mass[0, 0]))
        self.assertTrue(torch.equal(next_memory_state.timestep[0, 0], memory_state.timestep[0, 0]))
        # Unmasked positions: cum_write_mass and timestep should be updated
        for i in range(self.batch_shape[0]):
            for j in range(self.batch_shape[1]):
                if not (i == 0 and j == 0):
                    expected_cum_write_mass = memory_state.cum_write_mass[i, j] + self.memory_unit.memory_writer._max_interpolation_rate * torch.sigmoid(input_tensor[i, j].mean())
                    expected_timestep = memory_state.timestep[i, j] + False  # batch_mask is False
                    self.assertTrue(torch.allclose(next_memory_state.cum_write_mass[i, j], expected_cum_write_mass, atol=1e-6))
                    self.assertTrue(torch.equal(next_memory_state.timestep[i, j], expected_timestep))

    def test_write_probability_range(self):
        write_unit = self.memory_unit.memory_writer
        input_tensor = torch.randn(self.batch_shape + torch.Size([self.d_model]), dtype=self.dtype, device=self.device)
        # Compute update and write probability
        update, write_probability = write_unit._compute_common(input_tensor, {})
        # Check that write_probability is between 0 and 1
        self.assertTrue(torch.all(write_probability >= 0))
        self.assertTrue(torch.all(write_probability <= 1))

    def test_torchscript(self):
        memory_unit = torch.jit.script(self.memory_unit)
        memory_state = memory_unit.create_state(self.batch_shape)
        input_tensor = torch.randn(self.batch_shape + torch.Size([self.d_model]), dtype=self.dtype, device=self.device)
        batch_mask = torch.zeros(self.batch_shape, dtype=torch.bool, device=self.device)
        # Perform forward pass to get next memory state
        _, next_memory_state = memory_unit.forward(input_tensor, batch_mask, memory_state)
        # Perform reverse pass
        (output_tensor, restored_memory_state), original_memory = memory_unit.reverse(input_tensor, batch_mask, next_memory_state)
        # Check that the restored memory matches the original
        for key in memory_state.get_interpolation_states():
            original = memory_state.get_interpolation_states()[key]
            derived = original_memory.get_interpolation_states()[key]
            self.assertTrue(torch.allclose(original, derived, atol=1e-6))
        for key in memory_state.get_persistent_state():
            original = memory_state.get_persistent_state()[key]
            derived = original_memory.get_persistent_state()[key]
            self.assertTrue(torch.allclose(original, derived, atol=1e-6))
        # Check that 'timestep' and 'cum_write_mass' have been correctly reverted
        self.assertTrue(torch.equal(restored_memory_state.timestep, memory_state.timestep))
        self.assertTrue(torch.equal(restored_memory_state.cum_write_mass, memory_state.cum_write_mass))
        # Check that the output tensor matches the one from forward pass
        self.assertEqual(output_tensor.shape, self.batch_shape + torch.Size([10]))

    def test_gradient_flow(self):
        memory_state = self.memory_unit.create_state(self.batch_shape)
        # Set requires_grad=True for input_tensor
        input_tensor = torch.randn(self.batch_shape + torch.Size([self.d_model]),
                                   dtype=self.dtype, device=self.device, requires_grad=True)
        batch_mask = torch.zeros(self.batch_shape, dtype=torch.bool, device=self.device)
        # Perform forward pass
        output_tensor, _ = self.memory_unit.forward(input_tensor, batch_mask, memory_state)
        # Define a simple loss function
        loss = output_tensor.sum()
        # Backward pass
        loss.backward()
        # Check that gradients have been computed for input_tensor
        self.assertIsNotNone(input_tensor.grad)
        # Additionally, check that gradients have been computed for interpolation_logits
        write_unit = self.memory_unit.memory_writer
        self.assertIsNotNone(write_unit._interpolation_logits.grad)
        # Optionally, check that gradients are non-zero
        self.assertTrue(torch.any(write_unit._interpolation_logits.grad != 0))

    def test_running_distance_update(self):
        memory_state = self.memory_unit.create_state(self.batch_shape)
        input_tensor = torch.randn(self.batch_shape + torch.Size([self.d_model]), dtype=self.dtype, device=self.device)
        batch_mask = torch.zeros(self.batch_shape, dtype=torch.bool, device=self.device)
        # Perform forward pass
        _, next_memory_state = self.memory_unit.forward(input_tensor, batch_mask, memory_state)
        # Check that 'running_distance' has been updated to current 'timestep'
        self.assertTrue(torch.allclose(next_memory_state.running_distance, next_memory_state.timestep, atol=1e-6))

    def test_normalized_timestep_distance(self):
        memory_state = self.memory_unit.create_state(self.batch_shape)
        input_tensor = torch.randn(self.batch_shape + torch.Size([self.d_model]), dtype=self.dtype, device=self.device)
        batch_mask = torch.zeros(self.batch_shape, dtype=torch.bool, device=self.device)
        # Perform forward pass
        _, next_memory_state = self.memory_unit.forward(input_tensor, batch_mask, memory_state)
        # Check normalized_timestep_distance
        normalized_distance = next_memory_state.normalized_timestep_distance
        expected = (next_memory_state.timestep - next_memory_state.running_distance) / (next_memory_state.timestep + 1e-6)
        self.assertTrue(torch.allclose(normalized_distance, expected, atol=1e-6))
