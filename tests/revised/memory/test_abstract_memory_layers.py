import unittest
import torch
from typing import List, Tuple, Dict
from dataclasses import dataclass
from src.main.arcAGI2024.memory.base import (
    MemoryState,
    MemoryData,
    AbstractCreateState,
    AbstractReadMemory,
    AbstractWriteMemory,
    AbstractMemoryUnit,
    AbstractMemoryConfig,
    GradientTimeLossConfig,
    MemRegularizationLossConfig,
)

# Minimal concrete implementation of AbstractMemoryConfig
@dataclass
class ConcreteMemoryConfig(AbstractMemoryConfig):
    min_write_half_life_init: float = 1.0
    max_write_half_life_init: float = 2.0
    erase_epsilon_factor: float = 0.1
    gradient_loss: GradientTimeLossConfig = None
    mem_regularization_loss: MemRegularizationLossConfig = None

    @property
    def interpolation_factor_shapes(self) -> torch.Size:
        return torch.Size([])

    def __post_init__(self):
        super().__post_init__()

# Minimal concrete implementation of AbstractCreateState
class ConcreteCreateState(AbstractCreateState):
    def __init__(self, d_model: int, dtype: torch.dtype, device: torch.device):
        super().__init__(dtype, device)
        self.d_model = d_model

    def forward(self, batch_shape: List[int]) -> MemoryState:
        batch_size = batch_shape[0]
        device = self.device
        dtype = self.dtype

        zeros_scalar = torch.zeros(batch_size, device=device, dtype=dtype)
        zeros_vector = torch.zeros(batch_size, self.d_model, device=device, dtype=dtype)

        metric_tensors = {
            'cum_write_mass': zeros_scalar.clone(),
            'cum_erase_mass': zeros_scalar.clone(),
            'effective_write_mass': zeros_scalar.clone(),
            'timestep': zeros_scalar.clone(),
            'average_timestep_distance': zeros_scalar.clone(),
        }

        memory_tensors = {
            'memory': zeros_vector.clone(),
        }

        persistent_tensors = {}

        return MemoryState(metric_tensors, memory_tensors, persistent_tensors)

# Minimal concrete implementation of AbstractReadMemory
class ConcreteReadMemory(AbstractReadMemory):
    def __init__(self, d_model: int, dtype: torch.dtype, device: torch.device):
        super().__init__(dtype, device)
        self.d_model = d_model

    def read_memory(self, query: torch.Tensor, memories: MemoryData, persistent: MemoryData) -> torch.Tensor:
        # For simplicity, return the memory tensor directly
        memory_tensor = memories['memory']
        return memory_tensor

# Minimal concrete implementation of AbstractWriteMemory
class ConcreteWriteMemory(AbstractWriteMemory):
    def __init__(self, d_model: int, dtype: torch.dtype, device: torch.device, config: AbstractMemoryConfig):
        super().__init__(dtype, device, config)
        self.d_model = d_model

    def _compute_common(self, query: torch.Tensor, persistent: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor],
                                                                                                 torch.Tensor,
                                                                                                 torch.Tensor]:
        # Define update_state as the query tensor
        update_state = {'memory': query}

        # Define write_probability and erase_probability as constants
        write_probability = torch.full([query.size(0)], 0.5, device=self.device, dtype=self.dtype)
        erase_probability = torch.full([query.size(0)], 0.5, device=self.device, dtype=self.dtype)

        return update_state, write_probability, erase_probability

# Concrete implementation of AbstractMemoryUnit
class ConcreteMemoryUnit(AbstractMemoryUnit):
    def __init__(self, d_model: int, dtype: torch.dtype, device: torch.device, config: AbstractMemoryConfig):
        create_state_unit = ConcreteCreateState(d_model, dtype, device)
        read_unit = ConcreteReadMemory(d_model, dtype, device)
        write_unit = ConcreteWriteMemory(d_model, dtype, device, config)
        super().__init__(create_state_unit, read_unit, write_unit)

# Test suite
class TestMemoryModule(unittest.TestCase):
    def setUp(self):
        # Common setup for tests
        self.d_model = 16
        self.batch_size = 4
        self.dtype = torch.float32
        self.device = torch.device('cpu')
        self.config = ConcreteMemoryConfig()
        self.memory_unit = ConcreteMemoryUnit(self.d_model, self.dtype, self.device, self.config)

    def test_create_state(self):
        # Test that the state can be created
        batch_shape = [self.batch_size]
        memory_state = self.memory_unit.create_state(batch_shape)
        self.assertIsInstance(memory_state, MemoryState)
        self.assertEqual(memory_state.get_memories()['memory'].shape, (self.batch_size, self.d_model))

    def test_read_memory(self):
        # Test the read_memory method
        batch_shape = [self.batch_size]
        memory_state = self.memory_unit.create_state(batch_shape)
        query = torch.randn(self.batch_size, self.d_model, device=self.device, dtype=self.dtype)
        read_result = self.memory_unit.memory_reader(query, memory_state)
        self.assertEqual(read_result.shape, (self.batch_size, self.d_model))

    def test_write_memory(self):
        # Test the write_memory method
        batch_shape = [self.batch_size]
        memory_state = self.memory_unit.create_state(batch_shape)
        query = torch.randn(self.batch_size, self.d_model, device=self.device, dtype=self.dtype)
        update_state, control_gates = self.memory_unit.memory_writer.compute_common(query, memory_state)
        self.assertIn('memory', update_state)
        self.assertEqual(update_state['memory'].shape, (self.batch_size, self.d_model))
        write_gate, erase_gate = control_gates
        self.assertEqual(write_gate.shape, (self.batch_size,))
        self.assertEqual(erase_gate.shape, (self.batch_size,))

    def test_memory_unit_forward(self):
        # Test the memory unit forward method
        batch_shape = [self.batch_size]
        memory_state = self.memory_unit.create_state(batch_shape)
        input_tensor = torch.randn(self.batch_size, self.d_model, device=self.device, dtype=self.dtype)
        batch_mask = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        read_result, new_memory_state = self.memory_unit.forward(input_tensor, batch_mask, memory_state)
        self.assertEqual(read_result.shape, (self.batch_size, self.d_model))
        self.assertIsInstance(new_memory_state, MemoryState)

    def test_memory_unit_reverse(self):
        # Test the memory unit reverse method
        batch_shape = [self.batch_size]
        memory_state = self.memory_unit.create_state(batch_shape)
        input_tensor = torch.randn(self.batch_size, self.d_model, device=self.device, dtype=self.dtype)
        batch_mask = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        # First, perform forward to get next_memory
        read_result, next_memory_state = self.memory_unit.forward(input_tensor, batch_mask, memory_state)
        # Now, perform reverse
        (read_result_reverse, memory_state_reverse), original_memory = self.memory_unit.reverse(
            input_tensor, batch_mask, next_memory_state)
        self.assertEqual(read_result_reverse.shape, (self.batch_size, self.d_model))
        self.assertIsInstance(memory_state_reverse, MemoryState)
        self.assertIsInstance(original_memory, MemoryState)

# Run the tests
if __name__ == '__main__':
    unittest.main()
