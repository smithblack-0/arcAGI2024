import copy
import unittest
import torch
from typing import Any, Tuple
from src.main.arcAGI2024.base import PytreeState, TensorTree, parallel_pytree_map, GradientSubstitutionEndpoint


class MySavableState(PytreeState):
    """
    A concrete implementation of SavableState for testing.
    This class holds a tensor and demonstrates how to save and load state.
    """
    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor

    def save_state(self) -> Tuple[TensorTree, None]:
        # Simply return the tensor itself as the state
        return self.tensor, None

    def load_state(self, pytree: TensorTree, bypass: None) -> 'MySavableState':
        # Restore from the given pytree, which should be a tensor
        self.tensor = pytree
        return self
    def __eq__(self, other: Any) -> bool:
        # Equality check to facilitate testing
        if not isinstance(other, MySavableState):
            return False
        return torch.equal(self.tensor, other.tensor)


class TestParallelPytreeMapWithSavableState(unittest.TestCase):
    """
    Unit tests for parallel_pytree_map function with support for SavableState.
    """

    def test_parallel_pytree_map_with_savable_state(self):
        """
        Test parallel_pytree_map with nested structures containing SavableState instances.
        """
        # Create test data: two SavableState objects
        tensor1 = torch.tensor([1.0, 2.0, 3.0])
        tensor2 = torch.tensor([4.0, 5.0, 6.0])
        state1 = MySavableState(tensor1)
        state2 = MySavableState(tensor2)

        # Function to add two tensors
        def add_tensors(x, y):
            if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
                return x + y
            return None

        # Apply parallel_pytree_map with SavableState instances
        result = parallel_pytree_map(add_tensors, state1, state2)

        # Verify that the result is a tuple of SavableState objects
        self.assertIsInstance(result, MySavableState)
        self.assertTrue(torch.equal(result.tensor, tensor1 + tensor2))

    def test_parallel_pytree_map_with_mixed_structures(self):
        """
        Test parallel_pytree_map with mixed nested structures containing lists, dicts, and SavableState.
        """
        # Create test data
        tensor1 = torch.tensor([1.0, 2.0, 3.0])
        tensor2 = torch.tensor([4.0, 5.0, 6.0])
        state1 = MySavableState(tensor1)
        state2 = MySavableState(tensor2)

        nested_structure1 = {'a': [copy.deepcopy(state1), torch.tensor([7.0, 8.0])], 'b': (torch.tensor([9.0]), state1)}
        nested_structure2 = {'a': [copy.deepcopy(state2), torch.tensor([1.0, 2.0])], 'b': (torch.tensor([3.0]), state2)}

        # Function to add two tensors
        def add_tensors(x, y):
            if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
                return x + y
            return None

        # Apply parallel_pytree_map with mixed structures
        result = parallel_pytree_map(add_tensors, nested_structure1, nested_structure2)

        # Verify the result structure matches the original nested structures
        self.assertIsInstance(result, dict)
        self.assertIn('a', result)
        self.assertIn('b', result)
        self.assertIsInstance(result['a'], list)
        self.assertIsInstance(result['b'], tuple)

        print(result["a"][0])

        # Verify the SavableState instances in the result
        expected_tensor = tensor1 + tensor2
        self.assertIsInstance(result['a'][0], MySavableState)
        self.assertTrue(torch.equal(result['a'][0].tensor, expected_tensor))

        # Verify the remaining tensor results
        self.assertTrue(torch.equal(result['a'][1], torch.tensor([8.0, 10.0])))
        self.assertTrue(torch.equal(result['b'][0], torch.tensor([12.0])))
        self.assertTrue(torch.equal(result['b'][1].tensor, expected_tensor))

class GradientSubstitutionEndpoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, desired_gradients):
        ctx.save_for_backward(desired_gradients)
        return torch.tensor(0.0, requires_grad=True, device=tensor.device)

    @staticmethod
    def backward(ctx, grad_output):
        desired_gradients, = ctx.saved_tensors
        return desired_gradients, None

class TestGradientSubstitutionEndpoint(unittest.TestCase):
    def test_gradient_substitution(self):
        # Create an input tensor with requires_grad=True
        input_tensor = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

        # Define desired gradients
        desired_grads = torch.tensor([0.1, 0.2, 0.3])

        # Use GradientSubstitutionEndpoint in the computation graph
        output = GradientSubstitutionEndpoint.apply(input_tensor, desired_grads)

        # Combine output with a dummy loss to trigger backpropagation
        loss = output
        loss.backward()

        # Check that the gradients of input_tensor match the desired gradients
        self.assertTrue(torch.allclose(input_tensor.grad, desired_grads),
                        f"Expected gradients: {desired_grads}, but got: {input_tensor.grad}")