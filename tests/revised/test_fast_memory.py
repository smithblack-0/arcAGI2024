import unittest
import torch
from src.main.arcAGI2024.memory.fast_memory import (CreateState, LinearAttention,
                                                    ReadMemory, WriteMemory, FastLinearMemory)
from src.main.arcAGI2024.base import set_rng_state, get_rng_state
class TestCreateState(unittest.TestCase):

    def setUp(self):
        self.d_memory = 64
        self.num_memories = 10
        self.device = torch.device('cpu')
        self.dtype = torch.float32
        self.batch_shape = torch.Size([2])
        self.create_state = CreateState(self.d_memory, self.num_memories,
                                        dtype=self.dtype, device=self.device)

    def test_initialization_shapes(self):
        state = self.create_state(self.batch_shape)

        # Check shapes
        self.assertEqual(state.memories.shape, (2, self.num_memories, self.d_memory))

    def test_dtype_and_device(self):
        state = self.create_state(self.batch_shape)

        # Check dtype and device
        self.assertEqual(state.memories.dtype, self.dtype)


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
        self.d_memory = 64
        self.num_read_heads = 4
        self.device = torch.device('cpu')
        self.dtype = torch.float32
        self.reader = ReadMemory(self.d_model, self.d_memory, self.num_read_heads,
                                 torch.relu, dtype=self.dtype, device=self.device)
        self.memory_state = CreateState(self.d_memory, 10,
                                        dtype=self.dtype, device=self.device)(torch.Size([2]))

    def test_read_memory_shape(self):
        query = torch.randn(2, self.d_model, device=self.device, dtype=self.dtype)
        addresses = torch.randn(2, 10, self.d_memory, device=self.device, dtype=self.dtype)

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
        self.writer = WriteMemory(self.d_model, self.d_memory, self.num_write_heads,
                                  10, dropout_rate= 0.3, max_write_factor= 0.9,
                                  linear_kernel_activation=torch.relu, dtype=self.dtype, device=self.device)
        self.memory_state = CreateState(self.d_memory, 10,
                                        dtype=self.dtype, device=self.device)(torch.Size([2]))

    def test_memory_advance(self):
        key = torch.randn(2, self.d_model, device=self.device, dtype=self.dtype)
        values = torch.randn(2, self.d_model, device=self.device, dtype=self.dtype)
        addresses = torch.randn(2, 10, self.d_memory, device=self.device, dtype=self.dtype)

        original_memory, original_cum_prob = self.memory_state.get()
        batch_mask = torch.zeros_like(key[..., 0], dtype=torch.bool)

        # Perform write operation
        update,write_factor = self.writer.compute_common(key, values, addresses)
        memory_state = self.writer.advance_memory(update, write_factor, batch_mask, self.memory_state)
        updated_memory, updated_cum_prob = memory_state.get()

        # Check if memory state was updated
        self.assertFalse(torch.equal(original_memory, updated_memory))
        self.assertFalse(torch.equal(original_cum_prob, updated_cum_prob))

    def test_memory_reverse(self):
        key = torch.randn(2, self.d_model, device=self.device, dtype=self.dtype)
        values = torch.randn(2, self.d_model, device=self.device, dtype=self.dtype)
        addresses = torch.randn(2, 10, self.d_memory, device=self.device, dtype=self.dtype)

        original_memory, original_cum_prob = self.memory_state.get()
        batch_mask = torch.zeros_like(key[..., 0], dtype=torch.bool)

        # Perform write operation
        update, write_factor = self.writer.compute_common(key, values, addresses)
        memory_state = self.writer.reverse_memory(update, write_factor, batch_mask, self.memory_state)
        updated_memory, updated_cum_prob = memory_state.get()

        # Check if memory state was updated
        self.assertFalse(torch.equal(original_memory, updated_memory))
        self.assertFalse(torch.equal(original_cum_prob, updated_cum_prob))

    def test_reversability(self):
        key = torch.randn(2, self.d_model, device=self.device, dtype=self.dtype)
        values = torch.randn(2, self.d_model, device=self.device, dtype=self.dtype)
        addresses = torch.randn(2, 10, self.d_memory, device=self.device, dtype=self.dtype)

        original_memory, original_cum_prob = self.memory_state.get()
        batch_mask = torch.zeros_like(key[..., 0], dtype=torch.bool)

        # Perform write forward
        rng_state = get_rng_state(key.device)
        update, write_factor = self.writer.compute_common(key, values, addresses)
        memory_state = self.writer.advance_memory(update, write_factor, batch_mask, self.memory_state)

        # Perform reverse
        set_rng_state(rng_state, key.device)
        memory_state = self.writer.reverse_memory(update, write_factor, batch_mask, memory_state)
        updated_memory, updated_cum_prob = memory_state.get()

        # Check if memory states are about the same
        self.assertTrue(torch.allclose(original_memory, updated_memory))
        self.assertTrue(torch.allclose(original_cum_prob, updated_cum_prob))

class TestFastLinearMemory(unittest.TestCase):

    def setUp(self):
        self.d_model = 32
        self.d_address = 16
        self.d_memory = 64
        self.num_read_heads = 4
        self.num_write_heads = 4
        self.num_memories = 10
        self.bank_size = 5
        self.dropout_rate = 0.01
        self.device = torch.device('cpu')
        self.dtype = torch.float32
        self.fast_memory = FastLinearMemory(self.d_model,  self.d_memory,
                                            self.num_read_heads, self.num_write_heads,
                                            self.num_memories, self.dropout_rate,
                                            dtype=self.dtype, device=self.device)
        self.memory_state = self.fast_memory.create_state(torch.Size([2]))

    def test_fast_memory_forward(self):
        tensor = torch.randn(10, self.d_model, device=self.device, dtype=self.dtype)
        batch_mask = torch.rand([10]) > 0.5

        # Run forward pass
        memory_state = self.fast_memory.create_state([10])
        response, memory = self.fast_memory(tensor, batch_mask, memory_state)

        # Check output shape
        self.assertEqual(response.shape, (10, self.d_model))

    def test_memory_gradients(self):
        tensor = torch.randn(2, self.d_model, device=self.device, dtype=self.dtype, requires_grad=True)
        batch_mask = torch.rand([2]) > 0.5

        # Run forward pass
        response, memory = self.fast_memory(tensor, batch_mask, self.memory_state)

        # Run backward pass
        response = response.sum().backward()
