import unittest
import torch
from src.main.model.support_datastructure.computation_stack import AdaptiveComputationStack  # Adjust the import path accordingly

class TestAdaptiveComputationStack(unittest.TestCase):

    @staticmethod
    def create_mock_data(batch_size, d_model, stack_depth):
        embedding_shape = torch.Size([batch_size, d_model])
        return embedding_shape, stack_depth

    def test_initialization(self):
        """
        Test the initialization of AdaptiveComputationStack.
        Verify the shape of the stack and pointers, and ensure they start with the correct values.
        """
        embedding_shape, stack_depth = self.create_mock_data(batch_size=2, d_model=4, stack_depth=3)
        stack = AdaptiveComputationStack(stack_depth, embedding_shape)

        # Check if stack is initialized with correct shape and zeros
        self.assertEqual(stack.stack.shape, torch.Size([stack_depth, *embedding_shape]))
        self.assertTrue(torch.all(stack.stack == 0))

        # Check if pointers are initialized with correct shape and pointer[0] == 1
        self.assertEqual(stack.pointers.shape, torch.Size([stack_depth, *embedding_shape[:-1]]))
        self.assertTrue(torch.all(stack.pointers[0] == 1.0))
        self.assertTrue(torch.all(stack.pointers[1:] == 0.0))

    def test_set_expression(self):
        """
        Test if the set_expression function correctly updates the stack based on embedding and pointer probabilities.
        """
        embedding_shape, stack_depth = self.create_mock_data(batch_size=2, d_model=4, stack_depth=3)
        stack = AdaptiveComputationStack(stack_depth, embedding_shape)

        embedding = torch.randn(*embedding_shape)
        batch_mask = torch.zeros(embedding_shape[:-1], dtype=torch.bool)


        # Call set_expression and verify the stack content changes as expected
        original = stack.get_expression()
        stack.set_expression(embedding, batch_mask)

        self.assertTrue(torch.any(original != stack.get_expression()))

        # Now test with a mask that skips updates
        batch_mask = torch.ones(embedding_shape[:-1], dtype=torch.bool)  # Skip updates
        previous_stack = stack.stack.clone()
        stack.set_expression(embedding, batch_mask)

        self.assertTrue(torch.all(previous_stack == stack.stack), "Stack values should not update when masked.")

    def test_adjust_stack(self):
        """
        Test the adjust_stack method to ensure correct behavior when applying enstack, no-op, and destack actions.
        """
        embedding_shape, stack_depth = self.create_mock_data(batch_size=2, d_model=4, stack_depth=3)
        stack = AdaptiveComputationStack(stack_depth, embedding_shape)

        action_probabilities = torch.tensor([[0.7, 0.2, 0.1], [0.4, 0.4, 0.2]])
        batch_mask = torch.zeros(embedding_shape[:-1], dtype=torch.bool)

        # Adjust the stack and verify that pointers have been correctly updated
        original_pointers = stack.pointers.clone()
        stack.adjust_stack(action_probabilities, batch_mask)

        # Check that the stack probabilities have changed based on the action probabilities
        self.assertFalse(torch.equal(stack.pointers, original_pointers), "Pointers were not updated correctly.")

        # Now test with a mask that skips updates
        batch_mask = torch.ones(embedding_shape[:-1], dtype=torch.bool)  # Skip updates
        original_pointers = stack.pointers.clone()
        stack.adjust_stack(action_probabilities, batch_mask)

        self.assertTrue(torch.all(original_pointers == stack.pointers), "Pointers should not update when masked.")

    def test_statistics_accumulation(self):
        """
        Test if statistics are properly accumulated when adjust_stack is called.
        """
        embedding_shape, stack_depth = self.create_mock_data(batch_size=2, d_model=4, stack_depth=3)
        stack = AdaptiveComputationStack(stack_depth, embedding_shape)

        action_probabilities = torch.tensor([[0.7, 0.2, 0.1], [0.4, 0.4, 0.2]])
        batch_mask = torch.zeros(embedding_shape[:-1], dtype=torch.bool)  # Allow updates

        # Perform multiple adjust_stack calls to accumulate statistics
        stack.adjust_stack(action_probabilities, batch_mask)
        original_statistics = stack.probability_statistics.clone()

        # Run the adjust_stack again to accumulate more statistics
        stack.adjust_stack(action_probabilities, batch_mask)

        # Verify statistics have been accumulated correctly
        self.assertTrue(torch.all(original_statistics <= stack.probability_statistics), "Statistics were not accumulated correctly.")

        # Now test with a mask that skips updates
        batch_mask = torch.ones(embedding_shape[:-1], dtype=torch.bool)  # Skip updates
        original_statistics = stack.probability_statistics.clone()
        stack.adjust_stack(action_probabilities, batch_mask)

        # Ensure statistics haven't changed
        self.assertTrue(torch.all(original_statistics == stack.probability_statistics),
                        "Statistics should not update when masked.")

    def test_call_method(self):
        """
        Test the __call__ method, which performs set, adjust, and get in one step.
        """
        embedding_shape, stack_depth = self.create_mock_data(batch_size=2, d_model=4, stack_depth=3)
        stack = AdaptiveComputationStack(stack_depth, embedding_shape)

        embedding = torch.randn(*embedding_shape)
        action_probabilities = torch.tensor([[0.5, 0.3, 0.2], [0.2, 0.5, 0.3]])
        batch_mask = torch.zeros(embedding_shape[:-1], dtype=torch.bool)

        # Call the __call__ method
        result = stack(embedding, action_probabilities, batch_mask)

        # Verify that result has the correct shape and matches expected stack expression
        self.assertEqual(result.shape, embedding_shape, "The result shape is incorrect.")
        expected_expression = stack.get_expression()
        self.assertTrue(torch.allclose(result, expected_expression), "The result from __call__ is incorrect.")

        # Test with a mask that skips updates
        batch_mask = torch.ones(embedding_shape[:-1], dtype=torch.bool)  # Skip updates
        previous_result = result.clone()
        result = stack(embedding, action_probabilities, batch_mask)

        self.assertTrue(torch.all(previous_result == result), "Result should not update when masked.")
