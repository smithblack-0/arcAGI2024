import unittest
import torch
from torch import nn

from src.old.arcAGI2024 import (stack_controller_registry, AbstractSupportStack,
                                AbstractStackController)

class TestPointerSuperpositionStack(unittest.TestCase):
    """
    Unit tests for the PointerSuperpositionStack class.
    """

    def setUp(self):
        """
        Sets up the basic parameters used for creating PointerSuperpositionStack instances via the factory.
        """
        self.stack_depth = 5
        self.d_model = 8
        self.batch_shape = torch.Size([4, 3])
        self.dtype = torch.float32
        self.device = torch.device('cpu')
        self.control_dropout = 0.0

        # Default tensor initialization for stack setup
        self.defaults = {
            'layer1': torch.zeros(*self.batch_shape, self.d_model, dtype=self.dtype, device=self.device),
            'layer2': torch.ones(*self.batch_shape, self.d_model, dtype=self.dtype, device=self.device) * 0.5
        }

        # Initialize the stack controller using the registry
        self.stack_controller = stack_controller_registry.build(
            dtype=self.dtype,
            device=self.device,
            d_model=self.d_model,
            control_dropout=self.control_dropout,

        )

    def create_stack(self):
        """
        Utility method to create a fresh PointerSuperpositionStack instance for each test case.
        """
        return self.stack_controller.create_state(self.batch_shape, self.stack_depth, **self.defaults)

    def test_initialization(self):
        """
        Tests whether the stack initializes correctly when created via the factory.
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

        # Perform adjustment with various iteration limits
        stack.adjust_stack(controls)
        self.assertEqual(stack.pointer_prob_masses.shape, torch.Size([self.stack_depth, *self.batch_shape]))
        self.assertTrue(torch.all(stack.pointer_prob_masses >= 0))

    def test_pop(self):
        """
        Tests the pop method to ensure the correct weighted output is produced.
        """
        stack = self.create_stack()

        # Call pop and check if the output matches the batch shape and d_model
        expressions = stack.pop()
        for name, expression in expressions.items():
            self.assertEqual(expression.shape, torch.Size([*self.batch_shape, self.d_model]))
            self.assertEqual(expression.dtype, self.dtype)
            self.assertEqual(expression.device, self.device)

    def test_push(self):
        """
        Tests the push method to ensure the stack is updated correctly.
        """
        stack = self.create_stack()

        # Create random tensor inputs matching the default shapes
        new_tensors = {
            'layer1': torch.randn(*self.batch_shape, self.d_model, dtype=self.dtype, device=self.device),
            'layer2': torch.randn(*self.batch_shape, self.d_model, dtype=self.dtype, device=self.device)
        }
        # Perform push
        stack.push(**new_tensors)

        # Check if the stack updated correctly
        for name, expression in new_tensors.items():
            updated_stack = stack.stack[name][0]
            self.assertTrue(torch.allclose(updated_stack, expression, atol=1e-6))

    def test_integration(self):
        """
        Tests the broader integration method to ensure the stack updates and returns the correct output.
        """
        # Create random inputs
        embedding = torch.randn(*self.batch_shape, self.d_model, dtype=self.dtype, device=self.device)
        tensors = {
            'layer1': torch.randn(*self.batch_shape, self.d_model, dtype=self.dtype, device=self.device),
            'layer2': torch.randn(*self.batch_shape, self.d_model, dtype=self.dtype, device=self.device)
        }

        stack = self.stack_controller.create_state(self.batch_shape, self.stack_depth, **tensors)
        expressions = self.stack_controller(embedding, stack, **tensors)

        for name, expression in expressions.items():
            self.assertEqual(expression.shape, torch.Size([*self.batch_shape, self.d_model]))

    def test_manual_pointer_activity(self):
        """
        Test manual manipulation of the pointer stack, to make sure everything is coded sanely.
        """
        stack = self.stack_controller.create_state(batch_shape= [3],
                                                   stack_depth = 4,
                                                   test_tensor=torch.zeros([3]))

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
        stack.push(test_tensor=torch.Tensor([1.0, 1.0, 2.0]))
        expected = base.clone()
        expected[0] = torch.tensor([1.0, 1.0, 2.0])
        self.assertTrue(torch.all(stack.stack["test_tensor"] == expected))

        # Manually change it so that I am pointing at
        # position 1. Insert. 2. Read back. Ensure it is there
        update = torch.Tensor([2.0, 2.0, 2.0])

        control = enstack, sharpening
        stack.adjust_stack(control)
        stack.push(test_tensor=update)
        expected[1] = torch.tensor([2.0, 2.0, 2.0])
        self.assertTrue(torch.all(stack.stack["test_tensor"] == expected))
        self.assertTrue(torch.allclose(stack.pop()["test_tensor"], torch.tensor([2.0, 2.0, 2.0])))

        # Run a no op. Verify it does not change the position or contents of the stack
        control = no_op, sharpening
        stack.adjust_stack(control)
        self.assertTrue(torch.all(stack.stack["test_tensor"] == expected))
        self.assertTrue(torch.allclose(stack.pop()["test_tensor"], torch.tensor([2.0, 2.0, 2.0])))

        # Run a destack. Ensure it erases. Ensure we end up pointing back at the
        # top.
        control = destack, sharpening
        stack.adjust_stack(control)
        expected[1] = 0.0
        self.assertTrue(torch.all(stack.stack["test_tensor"] == expected))
        self.assertTrue(torch.allclose(stack.pop()["test_tensor"], torch.tensor([1.0, 1.0, 2.0])))
