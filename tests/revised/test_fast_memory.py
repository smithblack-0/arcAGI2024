import copy
import unittest
import torch
from src.main.ArcAGI2024.deep_memory import (CreateState, LinearAttention,
                                             ReadMemory, WriteMemory, FastLinearMemory)

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

        original_matrix, original_normalizer, original_cum_prob = self.memory_state.get()

        # Perform write operation
        memory_state = self.writer(key, values, addresses, self.memory_state)

        updated_matrix, updated_normalizer, updated_cum_prob = memory_state.get()

        # Check if memory state was updated
        self.assertFalse(torch.equal(original_matrix, updated_matrix))
        self.assertFalse(torch.equal(original_normalizer, updated_normalizer))
        self.assertFalse(torch.equal(original_cum_prob, updated_cum_prob))

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
                                            self.num_memories,
                                            dtype=self.dtype, device=self.device)
        self.memory_state = self.fast_memory.create_state(torch.Size([2]))

    def test_fast_memory_forward(self):
        tensor = torch.randn(2, self.d_model, device=self.device, dtype=self.dtype)

        # Run forward pass
        response, memory = self.fast_memory(tensor, self.memory_state)

        # Check output shape
        self.assertEqual(response.shape, (2, self.d_model))

    def test_memory_gradients(self):
        tensor = torch.randn(2, self.d_model, device=self.device, dtype=self.dtype, requires_grad=True)

        # Run forward pass
        response, memory = self.fast_memory(tensor, self.memory_state)

        # Run backward pass
        response = response.sum().backward()

    def test_memory_gradient_equivalency(self):
        # Test that we can rebuild a reasonably
        # close graph using the forward memory state,
        # and manually perform backprop.
        # Mark inputs so we catch gradients
        tensor = torch.randn(2, self.d_model, device=self.device, dtype=self.dtype)
        memory_state = copy.deepcopy(self.memory_state)

        memory_state.matrix.requires_grad_(True)
        memory_state.normalizer.requires_grad_(True)
        memory_state.write_probability_mass.requires_grad_(True)

        # Run forward pass.
        #
        # Also, mark the memory so we can set aside gradients during the back pass
        response, memory = self.fast_memory(tensor, memory_state)
        memory.write_probability_mass.retain_grad()
        memory.normalizer.retain_grad()
        memory.matrix.retain_grad()
        response.retain_grad()


        response.register_hook(lambda x : print("1"))
        memory.write_probability_mass.register_hook(lambda x : print("2"))
        memory.normalizer.register_hook(lambda x : print("3"))
        memory.write_probability_mass.register_hook(lambda x : print("4"))

        # Run backwards pass. Then set aside the expected gradients.
        loss = response.sum() + memory.matrix.sum() + memory.normalizer.sum() + memory.write_probability_mass.sum()
        loss.backward(retain_graph=True)

        input_matrix_gradient = memory.matrix.grad
        input_normalizer_gradient = memory.normalizer.grad
        input_write_probability_mass_gradient = memory.write_probability_mass.grad
        input_response_gradient = response.grad

        expected_matrix_gradients = memory_state.matrix.grad
        expected_normalizer_gradients = memory_state.normalizer.grad
        expected_write_probability_mass_gradients = memory_state.write_probability_mass.grad

        # Now we figure this out using the reconstructed graph.
        response, memory = self.fast_memory(tensor, memory_state)
        (revised, graph_memory), synthesized_memory_state = self.fast_memory.reverse(tensor, memory)


        # We run the backwards pass. We need to both reverse along the response,
        # which would be token loss, and along the memories

        revised.backward(input_response_gradient, retain_graph=True)
        graph_memory.write_probability_mass.backward(input_write_probability_mass_gradient, retain_graph=True)
        graph_memory.matrix.backward(input_matrix_gradient, retain_graph=True)
        graph_memory.normalizer.backward(input_normalizer_gradient, retain_graph=True)
