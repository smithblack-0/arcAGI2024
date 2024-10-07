import torch
import unittest
from src.main.model.core.subroutine_driver import SubroutineStackFactory, DifferentiableSubroutineStack


class TestSubroutineStackFactory(unittest.TestCase):

    def test_stack_creation(self):
        """
        Tests if the SubroutineStackFactory correctly creates a DifferentiableSubroutineStack
        instance with properly initialized components.
        """
        stack_depth = 5
        d_model = 4
        tensor = torch.randn([2, d_model])  # Simulating a batch of size 2
        min_iterations_before_destack = 2
        computation_limit = 10

        factory = SubroutineStackFactory(stack_depth, d_model)

        # Create the differentiable subroutine stack
        subroutine_stack = factory(tensor, min_iterations_before_destack, computation_limit)

        # Check if the created object is a DifferentiableSubroutineStack instance
        self.assertIsInstance(subroutine_stack, DifferentiableSubroutineStack)

        # Ensure that the stack depth matches what was passed to the factory
        self.assertEqual(subroutine_stack.stack_depth, stack_depth)

        # Check that initial stack probabilities are set correctly (focused on the first level)
        self.assertTrue(torch.allclose(subroutine_stack.pointers.get()[0], torch.ones_like(subroutine_stack.pointers.get()[0])))

    def test_initial_stack_setup(self):
        """
        Verifies that the tensor passed to the factory is correctly placed in the stack
        and that all other levels are initialized to zeros.
        """
        stack_depth = 5
        d_model = 3
        tensor = torch.randn([1, d_model])  # Simulating a tensor of shape (1, d_model)
        min_iterations_before_destack = 1
        computation_limit = 8

        factory = SubroutineStackFactory(stack_depth, d_model)

        # Create the differentiable subroutine stack
        subroutine_stack = factory(tensor, min_iterations_before_destack, computation_limit)

        # Ensure the tensor is correctly placed at the first stack level
        self.assertTrue(torch.allclose(subroutine_stack.stack.stack[0], tensor))

        # Ensure all other stack levels are initialized to zeros
        for i in range(1, stack_depth):
            self.assertTrue(torch.allclose(subroutine_stack.stack.stack[i], torch.zeros_like(tensor)))

    def test_position_markers_and_layernorm_setup(self):
        """
        Checks if position markers and layer normalization are correctly set up.
        """
        stack_depth = 4
        d_model = 3
        tensor = torch.randn([1, d_model])
        min_iterations_before_destack = 0
        computation_limit = 5

        factory = SubroutineStackFactory(stack_depth, d_model)

        # Create the differentiable subroutine stack
        subroutine_stack = factory(tensor, min_iterations_before_destack, computation_limit)

        # Ensure that the position markers have been expanded correctly across the stack
        self.assertEqual(subroutine_stack.stack.positions.shape[0], stack_depth)
        self.assertEqual(subroutine_stack.stack.positions.shape[-1], d_model)

        # Check that layer normalization was set up with the correct dimension
        self.assertIsInstance(subroutine_stack.stack.layernorm, torch.nn.LayerNorm)
        self.assertEqual(subroutine_stack.stack.layernorm.normalized_shape, (d_model,))
