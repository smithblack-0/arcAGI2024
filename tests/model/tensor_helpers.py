import unittest
import torch

from src.model.channel_bound_tensors import TensorChannelManager


class TestTensorChannelManager(unittest.TestCase):

    def setUp(self):
        # Initialize TensorChannelManager with a channel specification
        self.channel_spec = {
            "position": 2,
            "velocity": 3,
            "acceleration": 1
        }
        self.manager = TensorChannelManager(self.channel_spec)

    def test_channel_allocs(self):
        expected_allocs = ["position", "velocity", "acceleration"]
        self.assertEqual(self.manager.channel_allocs, expected_allocs)

    def test_channel_length(self):
        self.assertEqual(self.manager.channel_length, 6)  # 2 + 3 + 1

    def test_slices(self):
        expected_slices = {
            "position": slice(0, 2),
            "velocity": slice(2, 5),
            "acceleration": slice(5, 6)
        }
        self.assertEqual(self.manager.slices, expected_slices)

    def test_get_common_shape(self):
        tensor1 = torch.randn(10, 2)
        tensor2 = torch.randn(10, 3)
        tensor3 = torch.randn(10, 1)
        tensors = {
            "position": tensor1,
            "velocity": tensor2,
            "acceleration": tensor3
        }
        common_shape, dtype, device = self.manager.get_common_shape(tensors)
        self.assertEqual(common_shape, (10,))
        self.assertEqual(dtype, torch.float32)
        self.assertEqual(device, tensor1.device)

    def test_combine(self):
        tensor1 = torch.randn(10, 2)  # Position
        tensor2 = torch.randn(10, 3)  # Velocity
        tensor3 = torch.randn(10, 1)  # Acceleration
        combined = self.manager.combine({
            "position": tensor1,
            "velocity": tensor2,
            "acceleration": tensor3
        })
        self.assertEqual(combined.shape, (10, 6))  # Should be combined along the last dimension

    def test_extract(self):
        combined_tensor = torch.cat([
            torch.randn(10, 2),
            torch.randn(10, 3),
            torch.randn(10, 1)
        ], dim=-1)

        position = self.manager.extract(combined_tensor, "position")
        velocity = self.manager.extract(combined_tensor, "velocity")
        acceleration = self.manager.extract(combined_tensor, "acceleration")

        self.assertEqual(position.shape, (10, 2))
        self.assertEqual(velocity.shape, (10, 3))
        self.assertEqual(acceleration.shape, (10, 1))

    def test_replace(self):
        original_tensor = torch.randn(10, 6)
        replacement_tensor = torch.ones(10, 2)  # New position
        modified_tensor = self.manager.replace(original_tensor, "position", replacement_tensor)

        self.assertTrue(torch.equal(modified_tensor[..., self.manager.slices["position"]], replacement_tensor))
        self.assertFalse(torch.equal(modified_tensor[..., self.manager.slices["position"]],
                                     original_tensor[..., self.manager.slices["position"]]))


if __name__ == '__main__':
    unittest.main()
