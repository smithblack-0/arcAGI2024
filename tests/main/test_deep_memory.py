import unittest
import torch
from src.main.model.deep_memories.linear_memory_banks import (LinearKernelMemoryBank,
                                                              CreateState, ReadState, WriteState, MemoryState)
from src.main.model.virtual_layers import SelectionSpec
from src.main.model.deep_memories.abstract import deep_memory_registry
class TestLinearKernelMemoryBank(unittest.TestCase):

    def setUp(self):
        # Define model dimensions
        self.d_model = 16
        self.d_key = 8
        self.d_value = 8
        self.num_memories = 4
        self.bank_size = 3
        self.dropout = 0.1
        self.kernel_activation = torch.relu

        # Initialize memory bank unit
        self.memory_bank = deep_memory_registry.build("LinearKernelMemoryBank",
                                                      d_model=self.d_model,
                                                      mem_d_value=self.d_value,
                                                      num_memories=self.num_memories,
                                                      bank_size=self.bank_size,
                                                      dropout=self.dropout,
                                                      kernel_activation=self.kernel_activation
                                                      )

        # Extra batch dimensions
        self.extra_batch_dims = (2, 3)

        # SelectionSpec for testing
        batch_shape = self.extra_batch_dims
        selection_indices = torch.randint(0, self.bank_size, [*batch_shape, 2])
        selection_probabilities = torch.ones_like(selection_indices, dtype=torch.float)
        self.selection_spec = SelectionSpec(selection_index=selection_indices,
                                            selection_probabilities=selection_probabilities)

    def test_create_state(self):
        # Test state creation with CreateState class
        create_state = CreateState(self.d_key, self.d_value, self.num_memories)
        state = create_state(self.extra_batch_dims)

        # Validate state shapes
        matrix, normalizer = state.get()
        expected_matrix_shape = (*self.extra_batch_dims, self.num_memories, self.d_key, self.d_value)
        expected_normalizer_shape = (*self.extra_batch_dims, self.num_memories, self.d_key)
        self.assertEqual(matrix.shape, expected_matrix_shape)
        self.assertEqual(normalizer.shape, expected_normalizer_shape)

    def test_read_state(self):
        # Test the read functionality with ReadState class
        read_state = ReadState(self.d_model, self.d_key, self.d_value, self.num_memories, self.bank_size, self.dropout,
                               self.kernel_activation)

        # Prepare input tensor and state
        input_tensor = torch.randn(*self.extra_batch_dims, self.d_model)
        matrix = torch.randn(*self.extra_batch_dims, self.num_memories, self.d_key, self.d_value)
        normalizer = torch.randn(*self.extra_batch_dims, self.num_memories, self.d_key)
        state = (matrix, normalizer)
        state = MemoryState(*state)

        # Run read state and check output shape
        output = read_state(input_tensor, state, self.selection_spec)
        expected_shape = (*self.extra_batch_dims, self.d_model)
        self.assertEqual(expected_shape, output.shape)

    def test_write_state(self):
        # Test the write functionality with WriteState class
        write_state = WriteState(self.d_model, self.d_key, self.d_value, self.num_memories, self.bank_size,
                                 self.dropout, self.kernel_activation)

        # Prepare input tensor and state
        input_tensor = torch.randn(*self.extra_batch_dims, self.d_model)
        matrix = torch.randn(*self.extra_batch_dims, self.num_memories, self.d_key, self.d_value)
        normalizer = torch.randn(*self.extra_batch_dims, self.num_memories, self.d_key)
        state = (matrix, normalizer)
        state = MemoryState(*state)

        # Run write state and verify state update shape
        write_state(input_tensor, state, self.selection_spec)
        matrix_updated, normalizer_updated = state.get()
        self.assertEqual(matrix_updated.shape, matrix.shape)
        self.assertEqual(normalizer_updated.shape, normalizer.shape)

    def test_linear_kernel_memory_bank_forward(self):
        # Test the forward process of LinearKernelMemoryBank, combining read and write states
        input_tensor = torch.randn(*self.extra_batch_dims, self.d_model)
        state = self.memory_bank.create_state(self.extra_batch_dims)

        # Run the forward pass with no initial state
        response = self.memory_bank(input_tensor, self.selection_spec, state)

        # Check response shape
        expected_response_shape = (*self.extra_batch_dims, self.d_model)
        self.assertEqual(response.shape, expected_response_shape)

        # Validate updated state shapes
        matrix, normalizer = state.get()
        expected_matrix_shape = torch.Size([*self.extra_batch_dims, self.num_memories, self.d_model, self.d_value])
        expected_normalizer_shape = torch.Size([*self.extra_batch_dims, self.num_memories, self.d_model])
        self.assertEqual(matrix.shape, expected_matrix_shape)
        self.assertEqual(normalizer.shape, expected_normalizer_shape)

