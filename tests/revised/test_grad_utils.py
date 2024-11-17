import unittest
import torch
from typing import Any
from src.main.arcAGI2024.grad_utils import BatchCollectiveReductiveGradNorm, BatchCollectiveQuantileClipping
from src.main.arcAGI2024.base import middle_quantiles_mean
class TestBatchCollectiveReductiveGradNorm(unittest.TestCase):

    def setUp(self):
        # Initialize default parameters for the normalization class
        self.num_batch_dims = 1
        self.rescale_threshold = 0.1

    def test_mean_reduction_mode(self):
        # Initialize with mean reduction mode
        grad_norm = BatchCollectiveReductiveGradNorm(
            num_batch_dims=self.num_batch_dims,
            rescale_threshold=self.rescale_threshold,
            reduction_mode='mean',
            verbose=True
        )

        # Create sample gradients with a range of values
        grad_tree = torch.tensor([[0.05, 0.2, 0.1], [0.15, 0.05, 0.3]])

        # Run the normalization
        result = grad_norm(grad_tree)

    def test_max_reduction_mode(self):
        # Initialize with max reduction mode
        grad_norm = BatchCollectiveReductiveGradNorm(
            num_batch_dims=self.num_batch_dims,
            rescale_threshold=self.rescale_threshold,
            reduction_mode='max',
            verbose=True
        )

        # Create sample gradients with values exceeding the threshold
        grad_tree = torch.tensor([[0.15, 0.2, 0.25], [0.3, 0.05, 0.1]])

        # Run the normalization
        result = grad_norm(grad_tree)

        # Ensure the largest gradient in each batch does not exceed the threshold
        max_value = result.abs().max()
        self.assertLessEqual(max_value.item(), self.rescale_threshold+1e-3)

    def test_sum_reduction_mode(self):
        # Initialize with sum reduction mode
        grad_norm = BatchCollectiveReductiveGradNorm(
            num_batch_dims=self.num_batch_dims,
            rescale_threshold=self.rescale_threshold,
            reduction_mode='sum',
            verbose=True
        )

        # Create sample gradients that will likely exceed the threshold when summed
        grad_tree = torch.tensor([[0.2, 0.3, 0.1], [0.25, 0.35, 0.4]])

        # Run the normalization
        result = grad_norm(grad_tree)

        # Ensure that summed gradients are scaled to meet the threshold
        max_value = result.abs().max()
        self.assertLessEqual(max_value, self.rescale_threshold)


    def test_multiple_batch_dimensions(self):
        # Test with two batch dimensions and max reduction mode
        grad_norm = BatchCollectiveReductiveGradNorm(
            num_batch_dims=2,
            rescale_threshold=self.rescale_threshold,
            reduction_mode='max',
            verbose=True
        )

        # Create a 3D gradient tensor simulating two batch dimensions
        grad_tree = torch.tensor([[[0.1, 0.15, 0.2], [0.05, 0.3, 0.25]], [[0.2, 0.3, 0.4], [0.15, 0.1, 0.05]]])

        # Run the normalization
        result = grad_norm(grad_tree)


    def test_no_rescale_needed(self):
        # Test with gradients that do not require rescaling
        grad_norm = BatchCollectiveReductiveGradNorm(
            num_batch_dims=self.num_batch_dims,
            rescale_threshold=self.rescale_threshold,
            reduction_mode='mean',
            verbose=True
        )

        # Create sample gradients all below the threshold
        grad_tree = torch.tensor([[0.05, 0.08, 0.09], [0.04, 0.03, 0.02]])

        # Run the normalization
        result = grad_norm(grad_tree)

        # Ensure no rescaling occurred by comparing input and output tensors
        self.assertTrue(torch.allclose(grad_tree, result))


class TestBatchCollectiveQuantileClipping(unittest.TestCase):

    def setUp(self):
        # Initialize default parameters for the clipping class
        self.num_batch_dims = 1
        self.clip_factor = 1000.0
        self.protection_threshold = 0.0001

    def test_mean_mode_clipping(self):
        # Initialize with mean mode
        grad_clipper = BatchCollectiveQuantileClipping(
            num_batch_dims=self.num_batch_dims,
            clip_factor=self.clip_factor,
            protection_threshold=self.protection_threshold,
            mean_mode='mean',
            verbose=True
        )

        # Create sample gradients that are all below threshold
        grad_tree = torch.tensor([[0.05, 0.08, 0.07], [0.06, 0.09, 0.04]])

        # Run the clipping algorithm
        result = grad_clipper(grad_tree)

        # Ensure gradients are not clipped since they're all below the threshold
        self.assertTrue(torch.allclose(grad_tree, result))

    def test_quantiles_mean_mode_clipping(self):
        # Initialize with quantiles mean mode
        grad_clipper = BatchCollectiveQuantileClipping(
            num_batch_dims=self.num_batch_dims,
            clip_factor=self.clip_factor,
            protection_threshold=self.protection_threshold,
            mean_mode='quantiles_mean',
            verbose=True
        )

        # Create sample gradients with some values above the threshold
        grad_tree = torch.tensor([[0.00005, 0.0001, 0.05], [0.0001, 0.02, 0.00003]])

        # Run the clipping algorithm
        result = grad_clipper(grad_tree)

        # Ensure gradients are clipped properly based on quantiles
        clip_value = grad_clipper.clip_factor * middle_quantiles_mean(grad_tree.abs(), dim=-1)
        clip_value = clip_value.clip(min= grad_clipper.protection_threshold)
        self.assertTrue(torch.all(result <= clip_value.unsqueeze(-1)))

    def test_clip_factor_effectiveness(self):
        # Initialize with a very low clip factor
        grad_clipper = BatchCollectiveQuantileClipping(
            num_batch_dims=self.num_batch_dims,
            clip_factor=2.0,  # low clip factor to ensure clipping
            protection_threshold=self.protection_threshold,
            mean_mode='mean',
            verbose=True
        )

        # Create gradients where some values should exceed the clip factor mean threshold
        grad_tree = torch.tensor([[0.05, 0.2, 0.3], [0.4, 0.05, 0.1]])

        # Run the clipping algorithm
        result = grad_clipper(grad_tree)

        # Calculate the expected clip threshold for verification
        clip_value = grad_clipper.clip_factor * grad_tree.abs().mean(dim=-1)
        clip_value = clip_value.clip(min= grad_clipper.protection_threshold)

        # Ensure no gradient exceeds the calculated threshold
        self.assertTrue(torch.all(result <= clip_value.unsqueeze(-1)))

    def test_protection_threshold(self):
        # Initialize with mean mode and a low clip threshold
        grad_clipper = BatchCollectiveQuantileClipping(
            num_batch_dims=self.num_batch_dims,
            clip_factor=self.clip_factor,
            protection_threshold=0.1,  # higher threshold to avoid clipping smaller values
            mean_mode='mean',
            verbose=True
        )

        # Create gradients with values below the protection threshold
        grad_tree = torch.tensor([[0.05, 0.06, 0.04], [0.07, 0.03, 0.02]])

        # Run the clipping algorithm
        result = grad_clipper(grad_tree)

        # Ensure no gradients are clipped due to the protection threshold
        self.assertTrue(torch.allclose(grad_tree, result))

    def test_multiple_batch_dimensions(self):
        # Initialize with two batch dimensions and quantiles mean mode
        grad_clipper = BatchCollectiveQuantileClipping(
            num_batch_dims=2,
            clip_factor=self.clip_factor,
            protection_threshold=self.protection_threshold,
            mean_mode='quantiles_mean',
            verbose=True
        )

        # Create a 3D gradient tensor simulating two batch dimensions
        grad_tree = torch.tensor([[[0.1, 0.05, 0.2], [0.15, 0.25, 0.3]], [[0.01, 0.02, 0.1], [0.04, 0.01, 0.3]]])

        # Run the clipping algorithm
        result = grad_clipper(grad_tree)

        # Verify that clipping applies within each batch independently
        clip_value = grad_clipper.clip_factor * middle_quantiles_mean(grad_tree.abs(), dim=-1)
        clip_value = clip_value.clip(min= grad_clipper.protection_threshold)
        self.assertTrue(torch.all(result <= clip_value.unsqueeze(-1)))