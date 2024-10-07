import torch
import unittest
from src.main.model.subroutine_driver import DifferentiableSubroutineStack, ActionsManagement, ProbabilisticPointers, \
    SubroutineStackTracker


class TestDifferentiableSubroutineStack(unittest.TestCase):

    def create_stack_components(self, stack_depth, d_model, min_iterations, max_iterations):
        """
        Helper function to create necessary components for DifferentiableSubroutineStack.
        """
        # Mock statistics and initialize action manager
        action_stats = torch.zeros([stack_depth, 1, 3])
        action_manager = ActionsManagement(min_iterations, max_iterations, stack_depth, action_stats)

        # Mock probabilistic pointers (focused entirely on the first stack level)
        pointers = torch.zeros([stack_depth, 1])
        pointers[0] = 1.0  # Full probability at stack level 0
        pointer_manager = ProbabilisticPointers(pointers)

        # Mock stack setup (random tensor and layer norm)
        stack_tensor = torch.randn([stack_depth, 1, d_model])
        layernorm = torch.nn.LayerNorm(d_model)
        positions = torch.zeros([stack_depth, d_model])

        stack_tracker = SubroutineStackTracker(stack_tensor, layernorm, positions)

        return action_manager, pointer_manager, stack_tracker

    def test_stack_passthrough_properties(self):
        """
        Unified test for the stack depth, stack probabilities, and stack empty properties.
        Ensures that the passthroughs for these properties work correctly.
        """
        stack_depth = 5
        d_model = 3
        min_iterations = 2
        max_iterations = 10

        # Create components
        action_manager, pointer_manager, stack_tracker = self.create_stack_components(stack_depth, d_model,
                                                                                      min_iterations, max_iterations)

        # Initialize DifferentiableSubroutineStack
        subroutine_stack = DifferentiableSubroutineStack(action_manager, pointer_manager, stack_tracker)

        # Test the `stack_depth` property
        self.assertEqual(subroutine_stack.stack_depth, stack_depth)

        # Test the `stack_probabilities` property (sum should be 1, since we focused on stack level 0)
        stack_probs = subroutine_stack.stack_probabilities
        self.assertTrue(torch.allclose(stack_probs, torch.ones(1)))

        # Test the `stack_empty` property
        # Initially, the stack is not empty since we have non-zero probabilities.
        self.assertFalse(subroutine_stack.stack_empty)

        # Modify the pointers to simulate an empty stack
        subroutine_stack.pointers.pointers = torch.zeros([stack_depth, 1])
        self.assertTrue(subroutine_stack.stack_empty)

    def test_get_method_and_tensor_passthrough(self):
        """
        Test the get method and tensor passthrough functionality, ensuring tensors are
        correctly retrieved based on the probabilistic pointers and updated during stack actions.
        """
        stack_depth = 4
        d_model = 3
        min_iterations = 0
        max_iterations = 10

        # Create components
        action_manager, pointer_manager, stack_tracker = self.create_stack_components(stack_depth, d_model,
                                                                                      min_iterations, max_iterations)

        # Initialize DifferentiableSubroutineStack
        subroutine_stack = DifferentiableSubroutineStack(action_manager, pointer_manager, stack_tracker)

        # Check the `get` method
        retrieved_tensor = subroutine_stack.get()
        expected_tensor = stack_tracker.stack[0]  # Since pointer is focused on stack level 0
        self.assertTrue(torch.allclose(retrieved_tensor, expected_tensor))

        # Simulate action logits for enstack action
        action_logits = torch.zeros([1, 3])
        action_logits[..., 2] = 10.0  # Enstack action has highest logit

        # Simulate a new tensor to be added to the stack
        new_tensor = torch.randn([1, d_model])

        # Run the update and verify that the stack is updated correctly
        destack_probabilities, tensor = subroutine_stack.update(action_logits, new_tensor)

        self.assertTrue(torch.all(tensor == new_tensor))
        self.assertTrue(torch.all(destack_probabilities > 0))
