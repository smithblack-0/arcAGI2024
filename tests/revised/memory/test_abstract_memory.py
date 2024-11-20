import unittest
import torch
from torch import nn
from typing import Tuple, Dict, Any, List

# Assume the code provided is in a module named 'memory_module'
from src.main.arcAGI2024.memory.base import (
    AbstractMemoryConfig,
    MemoryState,
    AbstractCreateState,
    AbstractReadMemory,
    AbstractWriteMemory,
    AbstractMemoryUnit,
    register_concrete_implementation,
    make_memory_unit,
    _advance_memory_case,
    _retard_memory_case,
)

# For this example, I'll define minimal mock classes within the test code.
class MockMemoryConfig(AbstractMemoryConfig):
    def __init__(self,
                 max_interpolation_factor: float = 0.999,
                 min_write_half_life_init: float = 0.1,
                 max_write_half_life_init: float = 1000,
                 file_name: str = "mock_memory_config.json"
                 ):
        super().__init__(max_interpolation_factor,
                         min_write_half_life_init,
                         max_write_half_life_init,
                         file_name)

    @property
    def interpolation_factor_shapes(self) -> torch.Size:
        # For testing, let's assume the write factor is a scalar
        return torch.Size([])

class MockCreateState(AbstractCreateState):
    def forward(self, batch_shape: List[int]) -> MemoryState:
        # Create a simple memory state with zeros
        persistent_state = {'cum_write_factors': torch.zeros(batch_shape)}
        interpolation_state = {'memory_tensor': torch.zeros(batch_shape)}
        return MemoryState(persistent_state, interpolation_state)
class MockReadMemory(AbstractReadMemory):
    def forward(self, query: torch.Tensor, memory: MemoryState) -> torch.Tensor:
        # For testing, return the memory tensor directly
        interpolation_state = memory.get_interpolation_states()
        return interpolation_state['memory_tensor']
class MockWriteMemory(AbstractWriteMemory):
    def __init__(self, dtype: torch.dtype, device: torch.device, config: AbstractMemoryConfig):
        super().__init__(dtype, device, config)

    def _compute_common(self, query: torch.Tensor, persistent_state: Dict[str, torch.Tensor]
                        ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        # For testing, let's return the query as the update and a fixed write probability
        update_state = {'memory_tensor': query}
        write_probability = torch.tensor(0.5, device=query.device, dtype=query.dtype)
        return update_state, write_probability

class MockMemoryUnit(AbstractMemoryUnit):
    def __init__(self, d_model: int, dtype: torch.dtype, device: torch.device, config: AbstractMemoryConfig):
        create_state_unit = MockCreateState(dtype, device)
        read_unit = MockReadMemory(dtype, device)
        write_unit = MockWriteMemory(dtype, device, config)
        super().__init__(create_state_unit, read_unit, write_unit)


class TestMemoryModule(unittest.TestCase):
    def setUp(self):
        # Common setup for all tests
        self.device = torch.device('cpu')
        self.dtype = torch.float32
        self.d_model = 4  # Small dimension for testing
        self.batch_shape = [2, 3]  # Example batch shape
        self.config = MockMemoryConfig()
        self.memory_unit = make_memory_unit(self.d_model, self.dtype, self.device, self.config)

    def test_memory_state_initialization(self):
        # Create a memory state
        memory_state = self.memory_unit.create_state(self.batch_shape)

        # Check types
        self.assertIsInstance(memory_state, MemoryState)
        self.assertIsInstance(memory_state.persistent_state, dict)
        self.assertIsInstance(memory_state.interpolation_state, dict)

        # Check keys
        self.assertIn('cum_write_factors', memory_state.persistent_state)
        self.assertIn('memory_tensor', memory_state.interpolation_state)

        # Check tensor shapes
        self.assertEqual(memory_state.persistent_state['cum_write_factors'].shape, torch.Size(self.batch_shape))
        self.assertEqual(memory_state.interpolation_state['memory_tensor'].shape, torch.Size(self.batch_shape))
