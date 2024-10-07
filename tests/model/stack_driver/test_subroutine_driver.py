import torch
import unittest
from src.main.model.subroutine_driver import (SubroutineDriver, SubroutineCore,
                                              DifferentiableSubroutineStack)
from src.main.model.base import TensorTree
from typing import Tuple

class MockSubroutineCore(SubroutineCore):
    """
    A mock implementation of SubroutineCore for testing purposes.
    """
    def setup_state(self, tensor: torch.Tensor) -> TensorTree:
        # Simulate setting up a state as a simple tensor
        return torch.zeros_like(tensor)

    def forward(self, tensor: torch.Tensor, states: TensorTree) -> Tuple[torch.Tensor, TensorTree]:
        # Mock forward pass that returns the tensor and states unchanged
        return tensor + 1, states


class MockComplexSubroutineCore(SubroutineCore):
    """
    A mock implementation of SubroutineCore for sophisticated state testing.
    """
    def setup_state(self, tensor: torch.Tensor) -> TensorTree:
        # Simulate setting up a complex state with dictionaries and lists
        state = {
            "layer1": tensor,
            "layer2": [tensor.clone(), tensor.clone()]
        }
        return state

    def forward(self, tensor: torch.Tensor, states: TensorTree) -> Tuple[torch.Tensor, TensorTree]:
        # Mock forward pass that modifies the state slightly
        new_states = {
            "layer1": states["layer1"] + 1,
            "layer2": [s + 1 for s in states["layer2"]]
        }
        return tensor + 1, new_states


class TestSubroutineDriver(unittest.TestCase):

    def test_initialize_stacks(self):
        """
        Test if the stacks are initialized correctly using the tensor and state.
        """
        d_model = 4
        stack_depth = 3
        core = MockSubroutineCore()
        driver = SubroutineDriver(d_model, stack_depth, core)
        tensor = torch.randn([2, d_model])
        state = torch.randn([2, d_model])
        min_iterations_before_destack = 2
        max_iterations_before_flush = 10

        tensor_stack, state_stack = driver.initialize_stacks(tensor, state,
                                                             min_iterations_before_destack,
                                                             max_iterations_before_flush)

        self.assertIsInstance(tensor_stack, DifferentiableSubroutineStack)
        self.assertIsInstance(state_stack, DifferentiableSubroutineStack)

    def test_initialize_accumulators(self):
        """
        Test if accumulators for tensors and states are initialized correctly.
        """
        d_model = 4
        stack_depth = 3
        core = MockSubroutineCore()
        driver = SubroutineDriver(d_model, stack_depth, core)
        tensor = torch.randn([2, d_model])
        state = torch.randn([2, d_model])

        tensor_acc, state_acc = driver.initialize_accumulators(tensor, state)

        # The accumulator should be zero tensors of the same shape
        self.assertTrue(torch.all(tensor_acc == 0))
        self.assertTrue(torch.all(state_acc == 0))

    def test_forward_pass_simple(self):
        """
        Test forward pass with a simple state to ensure stack-based computation is correct.
        """
        d_model = 4
        stack_depth = 3
        core = MockSubroutineCore()
        driver = SubroutineDriver(d_model, stack_depth, core)
        tensor = torch.randn([2, d_model])
        state = torch.randn([2, d_model])
        max_computation_iterations = 5
        min_iterations_before_destack = 2

        # Perform the forward pass
        output_tensor, output_state = driver.forward(tensor, max_computation_iterations,
                                                     state, min_iterations_before_destack)

        # Ensure that the output tensor is processed correctly
        self.assertTrue(torch.any(output_tensor != tensor))

        # Ensure that the state was interacted with
        self.assertTrue(torch.any(output_state != state))

    def test_forward_pass_complex_state(self):
        """
        Test forward pass with a sophisticated state (list, dict) to ensure state updates work correctly.
        """
        d_model = 4
        stack_depth = 3
        core = MockComplexSubroutineCore()
        driver = SubroutineDriver(d_model, stack_depth, core)
        tensor = torch.randn([2, d_model])
        complex_state = {
            "layer1": torch.randn([2, d_model]),
            "layer2": [torch.randn([2, d_model]), torch.randn([2, d_model])]
        }
        max_computation_iterations = 5
        min_iterations_before_destack = 2

        # Perform the forward pass
        output_tensor, output_state = driver.forward(tensor, max_computation_iterations,
                                                     complex_state, min_iterations_before_destack)

        # Ensure that the output tensor is processed correctly
        self.assertTrue(torch.all(output_tensor != tensor))

        # Ensure that the state was modified correctly
        self.assertTrue(torch.all(output_state["layer1"] != complex_state["layer1"]))
        for new_s, old_s in zip(output_state["layer2"], complex_state["layer2"]):
            self.assertTrue(torch.all(new_s != old_s))

    def test_sophisticated_state_storage(self):
        """
        Test that the SubroutineDriver handles sophisticated nested states correctly.
        """
        d_model = 4
        stack_depth = 3
        core = MockComplexSubroutineCore()
        driver = SubroutineDriver(d_model, stack_depth, core)
        tensor = torch.randn([2, d_model])
        complex_state = {
            "layer1": torch.randn([2, d_model]),
            "layer2": [torch.randn([2, d_model]), torch.randn([2, d_model])]
        }
        max_computation_iterations = 5
        min_iterations_before_destack = 2

        # Perform the forward pass
        output_tensor, output_state = driver.forward(tensor, max_computation_iterations,
                                                     complex_state, min_iterations_before_destack)

        # Ensure that the output tensor is processed correctly
        self.assertTrue(torch.all(output_tensor != tensor))

        # Verify the state was properly handled and modified
        self.assertTrue(torch.all(output_state["layer1"] != complex_state["layer1"]))
        for new_s, old_s in zip(output_state["layer2"], complex_state["layer2"]):
            self.assertTrue(torch.all(new_s != old_s))
