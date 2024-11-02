import unittest
import torch
from src.main.argAGI2024.deep_memories.fast_linear_memory import (CreateState, LinearAttention,
                                                                  ReadMemory, WriteMemory, FastLinearMemory)
from src.main.argAGI2024.virtual_layers import SelectionSpec

class TestCreateState(unittest.TestCase):

    def setUp(self):
        self.d_address = 16
        self.d_memory = 64
        self.num_memories = 10
        self.device = torch.device('cpu')
        self.dtype = torch.float32
        self.batch_shape = torch.Size([2])
        self.create_state = CreateState(self.d_address, self.d_memory, self.num_memories,
                                        dtype=self.dtype, device=self.device)

    def test_initialization_shapes(self):
        state = self.create_state(self.batch_shape)

        # Check shapes
        self.assertEqual(state.matrix.shape, (2, self.num_memories, self.d_address, self.d_memory))
        self.assertEqual(state.normalizer.shape, (2, self.num_memories, self.d_address))

    def test_dtype_and_device(self):
        state = self.create_state(self.batch_shape)

        # Check dtype and device
        self.assertEqual(state.matrix.dtype, self.dtype)
        self.assertEqual(state.normalizer.device, self.device)


class TestLinearAttention(unittest.TestCase):

    def setUp(self):
        self.attention = LinearAttention(torch.relu)
        self.d_address = 16
        self.d_memory = 64
        self.device = torch.device('cpu')
        self.dtype = torch.float32

    def test_make_kernel_shapes(self):
        key = torch.randn(2, 10, self.d_address, device=self.device, dtype=self.dtype)
        value = torch.randn(2, 10, self.d_memory, device=self.device, dtype=self.dtype)

        matrix, normalizer = self.attention.make_kernel(key, value)

        # Check shapes
        self.assertEqual(matrix.shape, (2, self.d_address, self.d_memory))
        self.assertEqual(normalizer.shape, (2, self.d_address))

    def test_read_from_kernel_shape(self):
        query = torch.randn(2, 4, self.d_address, device=self.device, dtype=self.dtype)
        matrix = torch.randn(2, self.d_address, self.d_memory, device=self.device, dtype=self.dtype)
        normalizer = torch.randn(2, self.d_address, device=self.device, dtype=self.dtype)

        response = self.attention.read_from_kernel(query, matrix, normalizer)

        # Check response shape
        self.assertEqual(response.shape, (2, 4, self.d_memory))

class TestReadMemory(unittest.TestCase):

    def setUp(self):
        self.d_model = 32
        self.d_address = 16
        self.d_memory = 64
        self.num_read_heads = 4
        self.device = torch.device('cpu')
        self.dtype = torch.float32
        self.reader = ReadMemory(self.d_model, self.d_address, self.d_memory, self.num_read_heads,
                                 torch.relu, dtype=self.dtype, device=self.device)
        self.memory_state = CreateState(self.d_address, self.d_memory, 10,
                                        dtype=self.dtype, device=self.device)(torch.Size([2]))

    def test_read_memory_shape(self):
        query = torch.randn(2, self.d_model, device=self.device, dtype=self.dtype)
        addresses = torch.randn(2, 10, self.d_address, device=self.device, dtype=self.dtype)

        response = self.reader(query, addresses, self.memory_state)

        # Check response shape
        self.assertEqual(response.shape, (2, self.d_model))


class TestWriteMemory(unittest.TestCase):

    def setUp(self):
        self.d_model = 32
        self.d_address = 16
        self.d_memory = 64
        self.num_write_heads = 4
        self.device = torch.device('cpu')
        self.dtype = torch.float32
        self.writer = WriteMemory(self.d_model, self.d_address, self.d_memory, self.num_write_heads,
                                  10, torch.relu, dtype=self.dtype, device=self.device)
        self.memory_state = CreateState(self.d_address, self.d_memory, 10,
                                        dtype=self.dtype, device=self.device)(torch.Size([2]))

    def test_memory_update(self):
        key = torch.randn(2, self.d_model, device=self.device, dtype=self.dtype)
        values = torch.randn(2, self.d_model, device=self.device, dtype=self.dtype)
        addresses = torch.randn(2, 10, self.d_address, device=self.device, dtype=self.dtype)

        original_matrix, original_normalizer = self.memory_state.get()

        # Perform write operation
        self.writer(key, values, addresses, self.memory_state)

        updated_matrix, updated_normalizer = self.memory_state.get()

        # Check if memory state was updated
        self.assertFalse(torch.equal(original_matrix, updated_matrix))
        self.assertFalse(torch.equal(original_normalizer, updated_normalizer))

class TestFastLinearMemory(unittest.TestCase):

    def setUp(self):
        self.d_model = 32
        self.d_address = 16
        self.d_memory = 64
        self.num_read_heads = 4
        self.num_write_heads = 4
        self.num_memories = 10
        self.bank_size = 5
        self.device = torch.device('cpu')
        self.dtype = torch.float32
        self.fast_memory = FastLinearMemory(self.d_model, self.d_address, self.d_memory,
                                            self.num_read_heads, self.num_write_heads,
                                            self.num_memories, self.bank_size,
                                            dtype=self.dtype, device=self.device)
        self.selection_spec = SelectionSpec(
            selection_index=torch.randint(0, self.bank_size, (self.num_memories,)),
            selection_probabilities=torch.rand(self.num_memories)
        )
        self.memory_state = self.fast_memory.create_state(torch.Size([2]))
        print(self.memory_state.normalizer.numel())
        print(self.memory_state.matrix.numel())

    def test_fast_memory_forward(self):
        tensor = torch.randn(2, self.d_model, device=self.device, dtype=self.dtype)

        # Run forward pass
        response = self.fast_memory(tensor, self.selection_spec, self.memory_state)

        # Check output shape
        self.assertEqual(response.shape, (2, self.d_model))

    def test_memory_state_update(self):
        tensor = torch.randn(2, self.d_model, device=self.device, dtype=self.dtype)

        original_matrix, original_normalizer = self.memory_state.get()

        # Run forward pass to induce update
        _ = self.fast_memory(tensor, self.selection_spec, self.memory_state)

        updated_matrix, updated_normalizer = self.memory_state.get()

        # Verify memory state update
        self.assertFalse(torch.equal(original_matrix, updated_matrix))
        self.assertFalse(torch.equal(original_normalizer, updated_normalizer))


