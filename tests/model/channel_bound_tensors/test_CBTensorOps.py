import unittest
import torch
from src.old.CBTensors import CBTensor, CBTensorSpec  # Replace with your actual module


class TestCBTensorConcatOperator(unittest.TestCase):

    def setUp(self):
        # Define the tensor channels for use in the test methods
        # Each tensor has shape (3, 2, 5, 6, 4, channel_size)
        self.tensor_channels_1 = {
            'channel_1': torch.rand([3, 2, 5, 6, 4, 5]),
            'channel_2': torch.rand([3, 2, 5, 6, 4, 3])
        }
        self.tensor_channels_2 = {
            'channel_1': torch.rand([3, 2, 5, 6, 4, 5]),
            'channel_2': torch.rand([3, 2, 5, 6, 4, 3])
        }

    def test_concat_by_valid_positive_dim(self):
        # Concatenate along a valid positive dimension (dim=1)
        cb_tensor_1 = CBTensor.create_from_channels(self.tensor_channels_1)
        cb_tensor_2 = CBTensor.create_from_channels(self.tensor_channels_2)

        result = torch.cat([cb_tensor_1, cb_tensor_2], dim=0)
        expected_tensor = torch.cat([cb_tensor_1.get_tensor(), cb_tensor_2.get_tensor()], dim=0)
        self.assertTrue(torch.equal(result.get_tensor(), expected_tensor),
                        "Concatenation along a valid positive dimension failed.")
    def test_concat_by_last_dim(self):
        cb_tensor_1 = CBTensor.create_from_channels(self.tensor_channels_1)
        cb_tensor_2 = CBTensor.create_from_channels(self.tensor_channels_2)

        result = torch.cat([cb_tensor_1, cb_tensor_2], dim=-1)
        expected_tensor = torch.cat([cb_tensor_1.get_tensor(), cb_tensor_2.get_tensor()], dim=-2)
        self.assertTrue(torch.all(result.tensor == expected_tensor))
class TestCBTensorEqualityOperator(unittest.TestCase):

    def setUp(self):
        # Define tensor channels for testing with shapes (3, 2, 5) and (3, 2, 3)
        self.tensor_channels_a = {
            'channel_1': torch.ones([2, 3, 5]),  # Tensor full of ones
            'channel_2': torch.ones([2, 3, 3])  # Tensor full of ones
        }

        self.tensor_channels_b = {
            'channel_1': torch.ones([2, 3, 5]),  # Tensor full of ones
            'channel_2': torch.ones([2, 3, 3])  # Tensor full of ones
        }

        self.tensor_channels_c = {
            'channel_1': torch.zeros([2, 3, 5]),  # Tensor full of zeros
            'channel_2': torch.zeros([2, 3, 3])  # Tensor full of zeros
        }

    def test_eq_identical_tensors(self):
        # Create identical CBTensors
        tensor_a = CBTensor.create_from_channels(self.tensor_channels_a)
        tensor_b = CBTensor.create_from_channels(self.tensor_channels_b)

        # Test equality
        result = torch.eq(tensor_a, tensor_b)
        self.assertTrue(result.all(), "Expected tensors to be element-wise equal.")

    def test_eq_different_tensors(self):
        # Create different CBTensors (ones vs zeros)
        tensor_a = CBTensor.create_from_channels(self.tensor_channels_a)
        tensor_c = CBTensor.create_from_channels(self.tensor_channels_c)

        # Test inequality
        result = torch.eq(tensor_a, tensor_c)
        self.assertFalse(result.any(), "Expected tensors to be element-wise unequal.")

    def test_eq_incompatible_specs(self):
        # Create tensors with incompatible specs
        tensor_a = CBTensor.create_from_channels(self.tensor_channels_a)

        # Change the spec of the second tensor
        tensor_d = CBTensor(CBTensorSpec({'channel_1': 5}), torch.ones([2, 3, 5]))

        # Test ValueError when specs are incompatible
        with self.assertRaises(ValueError):
            torch.eq(tensor_a, tensor_d)

    def test_eq_with_broadcasting(self):
        # Test broadcasting capabilities of torch.eq
        tensor_a = CBTensor.create_from_channels(self.tensor_channels_a)

        # Create a tensor that will broadcast (with batch size 1 instead of 2)
        tensor_broadcast = CBTensor.create_from_channels({
            'channel_1': torch.ones([1, 3, 5]),  # Shape (1, 3, 5)
            'channel_2': torch.ones([1, 3, 3])  # Shape (1, 3, 3)
        })

        # Test equality with broadcasting
        result = torch.eq(tensor_a, tensor_broadcast)
        self.assertTrue(result.all(), "Expected tensors to be element-wise equal with broadcasting.")

class TestCBTensorUnsqueeze(unittest.TestCase):

    def setUp(self):
        # Set up tensor channels for each test to use
        # Tensor shape is (3, 2, 5, 6, 4, channel)
        self.tensor_channels = {
            'channel_1': torch.rand([3, 2, 5, 6, 4, 5]),  # Shape (3, 2, 5, 6, 4, 5) for channel_1
            'channel_2': torch.rand([3, 2, 5, 6, 4, 3])   # Shape (3, 2, 5, 6, 4, 3) for channel_2
        }

    def test_unsqueeze_positive_dim(self):
        # Initialize CBTensor within the method
        cb_tensor = CBTensor.create_from_channels(self.tensor_channels)

        # Test unsqueezing a valid positive dimension (before the channel dimension)
        unsqueezed_tensor = torch.unsqueeze(cb_tensor, dim=1)
        expected_shape = list(cb_tensor.get_tensor().shape)
        expected_shape.insert(1, 1)  # Insert a new dimension at position 1

        # Check the shape of the unsqueezed tensor
        self.assertEqual(unsqueezed_tensor.get_tensor().shape, tuple(expected_shape),
                         "Unsqueezing along positive dimension failed.")

    def test_unsqueeze_negative_dim(self):
        # Initialize CBTensor within the method
        cb_tensor = CBTensor.create_from_channels(self.tensor_channels)

        # Test unsqueezing a valid negative dimension (before the channel dimension)
        unsqueezed_tensor = torch.unsqueeze(cb_tensor, dim=-2)
        expected_shape = list(cb_tensor.get_tensor().shape)
        expected_shape.insert(-2, 1)  # Insert a new dimension at the second to last position (excluding the channel)

        # Check the shape of the unsqueezed tensor
        self.assertEqual(unsqueezed_tensor.get_tensor().shape, tuple(expected_shape),
                         "Unsqueezing along negative dimension failed.")

    def test_unsqueeze_last_dim(self):
        cb_tensor = CBTensor.create_from_channels(self.tensor_channels)

        # Test unsqueezing a valid negative dimension (before the channel dimension)
        unsqueezed_tensor = torch.unsqueeze(cb_tensor, dim=-1)
        expected_shape = list(cb_tensor.get_tensor().shape)
        expected_shape.insert(-1, 1)  # Insert a new dimension at the second to last position (excluding the channel)

        # Check the shape of the unsqueezed tensor
        self.assertEqual(unsqueezed_tensor.get_tensor().shape, tuple(expected_shape),
                         "Unsqueezing along negative dimension failed.")
    def test_unsqueeze_invalid_dim_error(self):
        # Initialize CBTensor within the method
        cb_tensor = CBTensor.create_from_channels(self.tensor_channels)

        # Test unsqueezing an invalid dimension (greater than the available dimensions)
        invalid_dim = cb_tensor.dim() + 1  # Out of range dimension
        with self.assertRaises(ValueError):
            torch.unsqueeze(cb_tensor, dim=invalid_dim)

    def test_unsqueeze_first_dim(self):
        # Initialize CBTensor within the method
        cb_tensor = CBTensor.create_from_channels(self.tensor_channels)

        # Test unsqueezing along the first dimension
        unsqueezed_tensor = torch.unsqueeze(cb_tensor, dim=0)
        expected_shape = (1,) + cb_tensor.get_tensor().shape

        # Check that the first dimension is successfully unsqueezed
        self.assertEqual(unsqueezed_tensor.get_tensor().shape, expected_shape,
                         "Unsqueezing along the first dimension failed.")
