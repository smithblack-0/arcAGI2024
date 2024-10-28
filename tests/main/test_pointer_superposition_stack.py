import unittest
import torch
from torch import nn

from src.main.model.computation_support_stack.pointer_superposition_stack import PointerSuperpositionStack


class TestPointerSuperpositionStack(unittest.TestCase):
    """
    Unit tests for the PointerSuperpositionStack class.
    """

    def setUp(self):
        """
        Sets up the basic parameters used for creating PointerSuperpositionStack instances.
        """
        self.stack_depth = 5
        self.d_model = 8
        self.batch_shape = torch.Size([4, 3])
        self.dtype = torch.float32
        self.device = torch.device('cpu')

        # Default tensor initialization for stack setup
        self.defaults = {
            'layer1': torch.zeros(*self.batch_shape, self.d_model, dtype=self.dtype, device=self.device),
            'layer2': torch.ones(*self.batch_shape, self.d_model, dtype=self.dtype, device=self.device) * 0.5
        }

        # Action and focus projectors
        self.action_projector = nn.Linear(self.d_model, 3)
        self.focus_projector = nn.Linear(self.d_model, 1)

    def create_stack(self) -> PointerSuperpositionStack:
        """
        Creates a new PointerSuperpositionStack instance for each test case.
        """
        return PointerSuperpositionStack(
            stack_depth=self.stack_depth,
            batch_shape=self.batch_shape,
            action_projector=self.action_projector,
            focus_projector=self.focus_projector,
            **self.defaults
        )

    def test_initialization(self):
        """
        Tests whether the stack initializes correctly.
        """
        stack = self.create_stack()

        # Check if the probability pointers are initialized correctly
        self.assertEqual(stack.pointers.shape, torch.Size([self.stack_depth, *self.batch_shape]))
        self.assertEqual(stack.pointers.dtype, self.dtype)
        self.assertEqual(stack.pointers.device, self.device)
        self.assertTrue(torch.allclose(stack.pointers[0], torch.ones_like(stack.pointers[0])))

    def test_adjust_stack(self):
        """
        Tests the adjust_stack method to ensure the stack updates correctly.
        """
        stack = self.create_stack()

        # Create random embeddings and batch mask
        action_probabilities = torch.rand(*self.batch_shape, 3, device=self.device, dtype=self.dtype)
        sharpening = torch.ones(*self.batch_shape, device=self.device, dtype=self.dtype)
        controls = action_probabilities, sharpening

        batch_mask = torch.zeros(self.batch_shape, dtype=torch.bool, device=self.device)

        # Perform adjustment with various iteration limits
        lost_prob1 = stack.adjust_stack(controls, batch_mask, min_iterations=1)
        self.assertEqual(lost_prob1.shape, self.batch_shape)

        lost_prob2 = stack.adjust_stack(controls, batch_mask, min_iterations=0)
        self.assertEqual(lost_prob2.shape, self.batch_shape)

        # Ensure the probability masses are updated
        self.assertTrue(torch.all(stack.pointer_prob_masses >= 0))

    def test_get_expression(self):
        """
        Tests the get_expression method to ensure the correct weighted output is produced.
        """
        stack = self.create_stack()

        # Call get_expression and check if the output matches the batch shape and d_model
        expressions = stack.get_expression()
        for name, expression in expressions.items():
            self.assertEqual(expression.shape, torch.Size([*self.batch_shape, self.d_model]))
            self.assertEqual(expression.dtype, self.dtype)
            self.assertEqual(expression.device, self.device)

    def test_set_expression(self):
        """
        Tests the set_expression method to ensure the stack is updated correctly.
        """
        stack = self.create_stack()

        # Create random tensor inputs matching the default shapes
        new_tensors = {
            'layer1': torch.randn(*self.batch_shape, self.d_model, dtype=self.dtype, device=self.device),
            'layer2': torch.randn(*self.batch_shape, self.d_model, dtype=self.dtype, device=self.device)
        }
        batch_mask = torch.zeros(self.batch_shape, dtype=torch.bool, device=self.device)

        # Perform set_expression
        stack.set_expression(batch_mask, **new_tensors)

        # Check if the stack updated correctly
        for name, expression in new_tensors.items():
            updated_stack = stack.stack[name][0]
            self.assertTrue(torch.allclose(updated_stack, expression, atol=1e-6))

    def test_call(self):
        """
        Tests the __call__ method to ensure the stack updates and returns the correct output.
        """
        stack = self.create_stack()

        # Create random inputs
        embedding = torch.randn(*self.batch_shape, self.d_model, dtype=self.dtype, device=self.device)
        batch_mask = torch.zeros(self.batch_shape, dtype=torch.bool, device=self.device)
        tensors = {
            'layer1': torch.randn(*self.batch_shape, self.d_model, dtype=self.dtype, device=self.device),
            'layer2': torch.randn(*self.batch_shape, self.d_model, dtype=self.dtype, device=self.device)
        }

        # Call the stack and get the outputs
        expressions, lost_probability = stack(embedding, batch_mask,5, 1, **tensors)

        # Validate the shape of the output and lost probability
        for name, expression in expressions.items():
            self.assertEqual(expression.shape, torch.Size([*self.batch_shape, self.d_model]))
        self.assertEqual(lost_probability.shape, self.batch_shape)

    def test_manual_pointer_activity(self):
        """
        Test manual manipulation of the pointer stack, to make sure
        everything is coded sanely
        """
        stack = PointerSuperpositionStack(stack_depth=4,
                                          batch_shape = [3],
                                          action_projector=self.action_projector,
                                          focus_projector=self.focus_projector,
                                          test_tensor = torch.zeros([3])
                                          )

        # Test that the default was coded as filled with zeros
        self.assertTrue(torch.all(stack.stack["test_tensor"] == torch.zeros([3])))
        base = stack.stack["test_tensor"].clone()

        # Define operands

        enstack = torch.tensor([1.0, 0.0, 0.0])
        enstack = torch.stack([enstack]*3, dim=0)

        no_op = torch.tensor([0.0, 1.0, 0.0])
        no_op = torch.stack([no_op]*3, dim=0)

        destack = torch.tensor([0.0, 0.0, 1.0])
        destack = torch.stack([destack]*3, dim=0)

        sharpening  = torch.ones([3])

        # Insert an entry into the first slot. Then check if it was
        # actually written correctly
        stack.set_expression(torch.tensor([False, False, True]), test_tensor=torch.Tensor([1.0, 1.0, 1.0]))
        expected = base.clone()
        expected[0] = torch.tensor([1.0, 1.0, 0.0])
        self.assertTrue(torch.all(stack.stack["test_tensor"] == expected))

        # Manually change it so that I am pointing at
        # position 1. Insert. 2. Read back. Ensure it is there
        batch_mask = torch.tensor([False, False, True])
        update = torch.Tensor([2.0, 2.0, 2.0])

        control = enstack, sharpening
        stack.adjust_stack(control, batch_mask, 10)
        stack.set_expression(batch_mask, test_tensor=update)
        expected[1] = torch.tensor([2.0, 2.0, 0.0])
        self.assertTrue(torch.all(stack.stack["test_tensor"] == expected))
        self.assertTrue(torch.allclose(stack.get_expression()["test_tensor"], torch.tensor([2.0, 2.0, 0.0])))

        # Run a no op. Verify it does not change the position or contents of the stack
        control = no_op, sharpening
        stack.adjust_stack(control, batch_mask, 10)
        self.assertTrue(torch.all(stack.stack["test_tensor"] == expected))
        self.assertTrue(torch.allclose(stack.get_expression()["test_tensor"], torch.tensor([2.0, 2.0, 0.0])))

        # Run a destack. Ensure it erases. Ensure we end up pointing back at the
        # top.
        control = destack, sharpening
        stack.adjust_stack(control, batch_mask, 10)
        expected[1] = 0.0
        self.assertTrue(torch.all(stack.stack["test_tensor"] == expected))
        self.assertTrue(torch.allclose(stack.get_expression()["test_tensor"], torch.tensor([1.0, 1.0, 0.0])))
