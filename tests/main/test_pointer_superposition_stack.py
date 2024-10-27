import unittest
import torch
from torch import nn

from src.main.model.pointer_superposition_stack import PointerSuperpositionStack
from src.main.model.base import parallel_pytree_map

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
            defaults=list(self.defaults.values()),
            dtype=self.dtype,
            device=self.device
        )

    def test_initialization(self):
        """
        Tests whether the stack initializes correctly.
        """
        stack = self.create_stack()

        # Check if the stack was initialized with the correct depth and batch shape
        for i, default in enumerate(self.defaults.values()):
            self.assertEqual(stack.stack[i].shape, torch.Size([self.stack_depth, *self.batch_shape, self.d_model]))
            self.assertEqual(stack.stack[i].dtype, self.dtype)
            self.assertEqual(stack.stack[i].device, self.device)

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
        embedding = torch.randn(*self.batch_shape, self.d_model, dtype=self.dtype, device=self.device)
        batch_mask = torch.zeros(self.batch_shape, dtype=torch.bool, device=self.device)

        # Perform adjustment with various iteration limits
        lost_prob1 = stack.adjust_stack(embedding, batch_mask, max_iterations=3, min_iterations=1)
        self.assertEqual(lost_prob1.shape, self.batch_shape)

        lost_prob2 = stack.adjust_stack(embedding, batch_mask, max_iterations=5, min_iterations=0)
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
        for i, expression in enumerate(expressions):
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
        stack.set_expression(batch_mask, *new_tensors.values())

        # Check if the stack updated correctly
        for i, new_tensor in enumerate(new_tensors.values()):
            updated_stack = stack.stack[i][0]
            self.assertTrue(torch.allclose(updated_stack, new_tensor, atol=1e-6))

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
        expressions, lost_probability = stack(embedding, batch_mask,5, 1, *tensors.values())

        # Validate the shape of the output and lost probability
        for i, expression in enumerate(expressions):
            self.assertEqual(expression.shape, torch.Size([*self.batch_shape, self.d_model]))
        self.assertEqual(lost_probability.shape, self.batch_shape)


# TODO: Finish verifying behavioral aspect.