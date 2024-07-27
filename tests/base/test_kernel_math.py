import torch
import unittest
from src.base import kernel_math

class TestKernelOperations(unittest.TestCase):

    def setUp(self):
        self.kernel1 = [torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[5.0, 6.0], [7.0, 8.0]])]
        self.kernel2 = [torch.tensor([[0.5, 1.5], [2.5, 3.5]]), torch.tensor([[4.5, 5.5], [6.5, 7.5]])]

    def test_transform_kernel(self):
        transform = lambda x: x * 2
        expected_result = [torch.tensor([[2.0, 4.0], [6.0, 8.0]]), torch.tensor([[10.0, 12.0], [14.0, 16.0]])]
        transformed_kernel = kernel_math.transform_kernel(self.kernel1, transform)
        for original, transformed in zip(expected_result, transformed_kernel):
            self.assertTrue(torch.equal(transformed, original))

    def test_add_kernels(self):
        expected_result = [torch.tensor([[1.5, 3.5], [5.5, 7.5]]), torch.tensor([[9.5, 11.5], [13.5, 15.5]])]
        added_kernel = kernel_math.add_kernels(self.kernel1, self.kernel2)
        for result, expected in zip(added_kernel, expected_result):
            self.assertTrue(torch.equal(result, expected))

    def test_stack_kernels(self):
        expected_result = [torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[0.5, 1.5], [2.5, 3.5]]]),
                           torch.tensor([[[5.0, 6.0], [7.0, 8.0]], [[4.5, 5.5], [6.5, 7.5]]])]
        stacked_kernel = kernel_math.stack_kernels(self.kernel1, self.kernel2, dim=0)
        for result, expected in zip(stacked_kernel, expected_result):
            self.assertTrue(torch.equal(result, expected))

    def test_concat_kernels(self):
        expected_result = [torch.tensor([[1.0, 2.0], [3.0, 4.0], [0.5, 1.5], [2.5, 3.5]]),
                           torch.tensor([[5.0, 6.0], [7.0, 8.0], [4.5, 5.5], [6.5, 7.5]])]
        concated_kernel = kernel_math.concat_kernels(self.kernel1, self.kernel2, dim=0)
        for result, expected in zip(concated_kernel, expected_result):
            self.assertTrue(torch.equal(result, expected))

    def test_split_kernel(self):
        kernel = [torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]), torch.tensor([[9.0, 10.0], [11.0, 12.0], [13.0, 14.0], [15.0, 16.0]])]
        expected_result = [
            [torch.tensor([1.0, 2.0]), torch.tensor([9.0, 10.0])],
            [torch.tensor([3.0, 4.0]), torch.tensor([11.0, 12.0])],
            [torch.tensor([5.0, 6.0]), torch.tensor([13.0, 14.0])],
            [torch.tensor([7.0, 8.0]), torch.tensor([15.0, 16.0])]
        ]
        split_kernels = kernel_math.split_kernel(kernel, dim=0)
        for result, expected in zip(split_kernels, expected_result):
            for r, e in zip(result, expected):
                self.assertTrue(torch.equal(r, e))
