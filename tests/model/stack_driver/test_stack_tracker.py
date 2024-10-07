import unittest
import torch
from src.main.model.subroutine_driver import SubroutineStateTracker, SubroutineEmbeddingTracker

class TestSubroutineStateTracker(unittest.TestCase):

    def create_state_tracker(self, stack):
        """
        Helper function to create a fresh SubroutineStateTracker instance for each test.
        """
        return SubroutineStateTracker(stack)

    def test_get_method(self):
        """
        Test that the get method correctly retrieves the superimposed stack state
        based on the probabilistic pointer values.
        """
        stack = torch.randn(4, 5)  # (stack_depth, d_model)
        tracker = self.create_state_tracker(stack)

        pointer_probabilities = torch.zeros(4)
        pointer_probabilities[1] = 1.0  # Expect full attention on stack[1]

        result = tracker.get(pointer_probabilities)

        self.assertTrue(torch.allclose(result, stack[1]), "Expected full attention on stack[1].")

    def test_change_superposition(self):
        """
        Test the change_superposition method that modifies the stack based on
        pointer and action probabilities.
        """
        stack = torch.ones(4, 5)  # (stack_depth, d_model)
        tracker = self.create_state_tracker(stack)

        pointer_probabilities = torch.zeros(4)
        pointer_probabilities[1] = 1.0  # Assume focus on stack[1]

        # Simulate action where destack should erase current context
        action_probabilities = torch.zeros(4, 3)
        action_probabilities[1, 0] = 1.0  # Destack action

        tracker.change_superposition(pointer_probabilities, action_probabilities)

        # After destack, stack[1] should be erased (i.e., set to zero)
        self.assertTrue(torch.all(tracker.stack[1] == 0), "stack[1] should be erased.")

    def test_update_method(self):
        """
        Test that the update method correctly integrates a new tensor into the stack,
        and that the relevant stack position changes. We will place the pointer at
        one location (100% probability at that position), perform an update, and
        verify that only that position is updated.
        """
        stack_depth = 4
        d_model = 5
        stack = torch.ones(stack_depth, d_model)  # (stack_depth, d_model)
        tracker = self.create_state_tracker(stack)

        # Set the pointer probabilities to fully focus on stack[2]
        pointer_probabilities = torch.zeros(stack_depth)
        pointer_probabilities[2] = 1.0  # Full focus on stack[2]

        tensor_to_integrate = torch.randn(d_model)  # Random tensor to integrate

        # Save the original stack before update
        original_stack_copy = stack.clone()

        # Perform the update
        tracker.update(pointer_probabilities, tensor_to_integrate)


        # Ensure other positions in the stack remain unchanged
        for i in range(stack_depth):
            if i != 2:
                self.assertTrue(torch.allclose(original_stack_copy[i], tracker.stack[i]),
                                f"stack[{i}] should not have changed during the update.")

        # Check if the stack[2] was updated correctly according to the logic
        expected_update = tensor_to_integrate # Expected result after the update
        self.assertTrue(torch.allclose(tracker.stack[2], expected_update),
                        "stack[2] was not updated as expected.")


class TestEmbeddingTracker(unittest.TestCase):

    def create_stack_tracker(self, stack, layernorm, positions):
        """
        Helper function to create a fresh SubroutineStackTracker instance for each test.
        """
        return SubroutineEmbeddingTracker(stack, layernorm, positions)

    def test_get_method(self):
        """
        Test that the get method correctly retrieves the superimposed stack state
        based on the probabilistic pointer values.
        """
        stack = torch.randn(4, 5)  # (stack_depth, d_model)
        layernorm = torch.nn.LayerNorm(5)
        positions = torch.zeros_like(stack)  # Positional encodings

        tracker = self.create_stack_tracker(stack, layernorm, positions)

        pointer_probabilities = torch.zeros(4)
        pointer_probabilities[1] = 1.0  # Expect full attention on stack[1]

        result = tracker.get(pointer_probabilities)

        self.assertTrue(torch.allclose(result, stack[1]), "Expected full attention on stack[1].")

    def test_change_superposition(self):
        """
        Test the change_superposition method that modifies the stack based on
        pointer and action probabilities.
        """
        stack = torch.ones(4, 5)  # (stack_depth, d_model)
        layernorm = torch.nn.LayerNorm(5)
        positions = torch.zeros_like(stack)  # Positional encodings

        tracker = self.create_stack_tracker(stack, layernorm, positions)

        pointer_probabilities = torch.zeros(4)
        pointer_probabilities[1] = 1.0  # Assume focus on stack[1]

        # Simulate action where destack should erase current context
        action_probabilities = torch.zeros(4, 3)
        action_probabilities[1, 0] = 1.0  # Destack action

        tracker.change_superposition(pointer_probabilities, action_probabilities)

        # After destack, stack[1] should be erased (i.e., set to zero)
        self.assertTrue(torch.all(tracker.stack[1] == 0), "stack[1] should be erased.")

    def test_update_method(self):
        """
        Test that the update method correctly integrates a new tensor into the stack,
        and that the relevant stack position changes. We will store the original value
        of the stack at the position being updated and verify that it is different after
        the update.
        """
        stack = torch.ones(4, 5)  # (stack_depth, d_model)
        layernorm = torch.nn.LayerNorm(5)
        positions = torch.randn(4, 5)  # Some positional encodings

        tracker = self.create_stack_tracker(stack, layernorm, positions)

        pointer_probabilities = torch.zeros(4)
        pointer_probabilities[2] = 1.0  # Assume focus on stack[2]

        tensor_to_integrate = torch.ones(5)  # (d_model)

        # Save a copy of the stack at position 2 before the update
        original_stack_copy = tracker.stack[2].clone()

        tracker.update(pointer_probabilities, tensor_to_integrate)

        # Verify that stack[2] has changed after the update
        self.assertFalse(torch.allclose(original_stack_copy, tracker.stack[2]),
                         "stack[2] should have been updated and should no longer match the original.")

        # Ensure other positions in the stack remain unchanged
        for i in range(stack.size(0)):
            if i != 2:
                self.assertTrue(torch.allclose(tracker.stack[i], stack[i]),
                                f"stack[{i}] should not have changed during the update.")
