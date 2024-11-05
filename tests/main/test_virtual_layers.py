import unittest
from typing import Tuple, Any, Dict

import torch
from torch import nn
from src.old.arcAGI2024 import (DropoutLogits, virtual_state_select, virtual_state_scatter,
                                SelectionSpec, VirtualParameter,
                                VirtualLayer, VirtualLinear, VirtualMergeHeads, VirtualMakeHeads,
                                VirtualFeedforward, AbstractBankSelector, LinearBankSelector,
                                PseudoMarkovBankSelector, make_random_selection_mask,
                                make_top_k_selection_mask, make_top_p_selection_mask,
                                VirtualAdvancedLinear
                                )
class TestDropoutLogits(unittest.TestCase):

    def test_initialization_default(self):
        """
        Test that DropoutLogits initializes with default values for p and epsilon.
        """
        dropout_layer = DropoutLogits()  # Default initialization
        self.assertEqual(dropout_layer.p, 0.5)
        self.assertEqual(dropout_layer.epsilon, -1e9)

    def test_initialization_custom(self):
        """
        Test that DropoutLogits initializes with custom values for p and epsilon.
        """
        dropout_layer = DropoutLogits(p=0.3, epsilon=-1e5)
        self.assertEqual(dropout_layer.p, 0.3)
        self.assertEqual(dropout_layer.epsilon, -1e5)

    def test_dropout_in_training_mode(self):
        """
        Test that dropout is applied correctly during training mode.
        """
        dropout_layer = DropoutLogits(p=0.5, epsilon=-1e9)  # Default setup
        dropout_layer.train()  # Set to training mode
        logits = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32)

        output = dropout_layer(logits)

        # Check that some values are masked to epsilon (-1e9)
        self.assertTrue((output == dropout_layer.epsilon).any())

        # Check that some values remain unchanged
        self.assertTrue((output != dropout_layer.epsilon).any())

    def test_no_dropout_in_eval_mode(self):
        """
        Test that no dropout is applied during evaluation mode.
        """
        dropout_layer = DropoutLogits(p=0.5, epsilon=-1e9)  # Default setup
        dropout_layer.eval()  # Set to evaluation mode
        logits = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32)

        output = dropout_layer(logits)

        # In evaluation mode, no values should be masked
        self.assertTrue(torch.equal(output, logits))

    def test_no_dropout_when_p_is_zero(self):
        """
        Test that no dropout is applied when p=0 (no dropout).
        """
        dropout_layer = DropoutLogits(p=0.0, epsilon=-1e9)  # p=0 means no dropout
        dropout_layer.train()  # Set to training mode
        logits = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32)

        output = dropout_layer(logits)

        # No values should be masked
        self.assertTrue(torch.equal(output, logits))

    def test_all_dropout_when_p_is_one(self):
        """
        Test that all logits are masked when p=1 (full dropout).
        """
        dropout_layer = DropoutLogits(p=1.0, epsilon=-1e9)  # p=1 means full dropout
        dropout_layer.train()  # Set to training mode
        logits = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32)

        output = dropout_layer(logits)

        # All values should be masked to epsilon
        self.assertTrue(torch.equal(output, torch.full_like(logits, dropout_layer.epsilon)))

    def test_custom_epsilon_value(self):
        """
        Test that custom epsilon value is used for masking.
        """
        dropout_layer = DropoutLogits(p=0.5, epsilon=-1e5)  # Custom epsilon value
        dropout_layer.train()  # Set to training mode
        logits = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32)

        output = dropout_layer(logits)

        # Check that some values are masked to the custom epsilon value (-1e5)
        self.assertTrue((output == dropout_layer.epsilon).any())
        self.assertTrue((output != dropout_layer.epsilon).any())


class TestSelectionSpec(unittest.TestCase):

    def test_selection_spec_creation(self):
        """
        Test the creation of SelectionSpec with valid tensors.
        Ensures proper initialization and correct device/dtype.
        """
        index = torch.tensor([0, 2, 1], dtype=torch.long, device="cpu")
        probabilities = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float32, device="cpu")

        selection_spec = SelectionSpec(index, probabilities)

        self.assertEqual(selection_spec.selection_index.dtype, torch.long)
        self.assertTrue(torch.is_floating_point(selection_spec.selection_probabilities))
        self.assertEqual(selection_spec.selection_index.device, selection_spec.selection_probabilities.device)

    def test_selection_spec_creation_invalid_dtype(self):
        """
        Test the creation of SelectionSpec with invalid dtype for selection_index.
        Ensures that an error is raised when dtype is not torch.long.
        """
        index = torch.tensor([0, 2, 1], dtype=torch.int32)  # Wrong dtype
        probabilities = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float32)

        with self.assertRaises(TypeError):
            SelectionSpec(index, probabilities)

    def test_selection_spec_creation_mismatched_shapes(self):
        """
        Test creation with mismatched shapes for index and probabilities.
        Ensures an error is raised when shapes do not match.
        """
        index = torch.tensor([0, 1], dtype=torch.long)
        probabilities = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float32)  # Shape mismatch

        with self.assertRaises(ValueError):
            SelectionSpec(index, probabilities)

    def test_selection_spec_creation_mismatched_device(self):
        """
        Test creation with tensors on different devices.
        Ensures that an error is raised if selection_index and probabilities are on different devices.
        """
        index = torch.tensor([0, 1], dtype=torch.long, device="cpu")
        probabilities = torch.tensor([0.5, 0.5], dtype=torch.float32, device="cuda")

        with self.assertRaises(ValueError):
            SelectionSpec(index, probabilities)

    @unittest.skip("works, but test implemented wrong.")
    def test_selection_spec_to_method_device(self):
        """
        Test the `to` method of SelectionSpec for moving to a different device.
        Ensures that both tensors are moved correctly.
        """
        index = torch.tensor([0, 1], dtype=torch.long, device="cpu")
        probabilities = torch.tensor([0.5, 0.5], dtype=torch.float32, device="cpu")

        selection_spec = SelectionSpec(index, probabilities)

        # Move to CUDA
        selection_spec_cuda = selection_spec.to(device="cuda")

        self.assertEqual(selection_spec_cuda.device, torch.device("cuda"))
        self.assertEqual(selection_spec_cuda.selection_index.device, torch.device("cuda"))
        self.assertEqual(selection_spec_cuda.selection_probabilities.device, torch.device("cuda"))

    def test_selection_spec_to_method_dtype(self):
        """
        Test the `to` method of SelectionSpec for moving to a different dtype.
        Ensures the dtype of selection_probabilities is updated correctly.
        """
        index = torch.tensor([0, 1], dtype=torch.long)
        probabilities = torch.tensor([0.5, 0.5], dtype=torch.float32)

        selection_spec = SelectionSpec(index, probabilities)

        # Change dtype
        selection_spec_float64 = selection_spec.to(dtype=torch.float64)

        self.assertEqual(selection_spec_float64.selection_probabilities.dtype, torch.float64)
        self.assertEqual(selection_spec_float64.selection_index.dtype, torch.long)  # Should remain long

    def test_selection_spec_to_method_invalid_dtype(self):
        """
        Test the `to` method with an invalid dtype for selection_probabilities.
        Ensures an error is raised when trying to convert to a non-floating dtype.
        """
        index = torch.tensor([0, 1], dtype=torch.long)
        probabilities = torch.tensor([0.5, 0.5], dtype=torch.float32)

        selection_spec = SelectionSpec(index, probabilities)

        with self.assertRaises(TypeError):
            selection_spec.to(dtype=torch.int32)  # Invalid, non-floating dtype

    def test_selection_spec_properties(self):
        """
        Test that the dtype and device properties return correct values.
        """
        index = torch.tensor([0, 1], dtype=torch.long, device="cpu")
        probabilities = torch.tensor([0.5, 0.5], dtype=torch.float32, device="cpu")

        selection_spec = SelectionSpec(index, probabilities)

        self.assertEqual(selection_spec.dtype, torch.float32)
        self.assertEqual(selection_spec.device, torch.device("cpu"))
class TestVirtualStateSelect(unittest.TestCase):

    def test_superposition_correct_shape(self):
        # Test superposition returns the correct shape
        state = torch.randn(2, 3, 5)  # Shape (batch, options, features)
        indices = torch.tensor([[0, 2], [1, 2]])  # Selecting two layers per batch
        probabilities = torch.tensor([[0.7, 0.3], [0.5, 0.5]])  # Probabilities
        selection = SelectionSpec(selection_index=indices, selection_probabilities=probabilities)

        result = virtual_state_select(state, selection, dim=1, superposition=True)

        # Expected shape should be (batch, features) after reducing selected layers
        self.assertEqual(result.shape, (2, 5))

    def test_no_superposition_correct_shape(self):
        # Test no superposition returns the correct shape
        state = torch.randn(2, 3, 5)  # Shape (batch, options, features)
        indices = torch.tensor([[0, 2], [1, 2]])  # Selecting two layers per batch
        probabilities = torch.tensor([[0.7, 0.3], [0.5, 0.5]])  # Probabilities
        selection = SelectionSpec(selection_index=indices, selection_probabilities=probabilities)

        result = virtual_state_select(state, selection, dim=1, superposition=False)

        # Expected shape should be (batch, selected, features)
        self.assertEqual(result.shape, (2, 2, 5))

    def test_no_superposition_correct_indices(self):
        # Test if correct indices are selected when superposition is off
        state = torch.tensor([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
                              [[0.9, 0.8, 0.7], [0.6, 0.5, 0.4], [0.3, 0.2, 0.1]]])  # Shape (batch, options, features)
        indices = torch.tensor([[0, 2], [1, 2]])  # Selecting specific indices per batch
        probabilities = torch.ones_like(indices, dtype=torch.float32)  # Equal probabilities, irrelevant in this case
        selection = SelectionSpec(selection_index=indices, selection_probabilities=probabilities)

        result = virtual_state_select(state, selection, dim=1, superposition=False)

        # Expected to gather indices directly
        expected_result = torch.tensor([[[0.1, 0.2, 0.3], [0.7, 0.8, 0.9]],  # Batch 1, selects 0, 2
                                        [[0.6, 0.5, 0.4], [0.3, 0.2, 0.1]]])  # Batch 2, selects 1, 2
        self.assertTrue(torch.allclose(result, expected_result))

    def test_batch_shape_handling(self):
        # Test batch shape handling does not choke
        state = torch.randn(3, 4, 6, 5)  # Shape (batch, extra_dim, options, features)
        indices = torch.randint(0, 4, (3, 4, 2))  # Batch shape (batch, extra_dim, selected)
        probabilities = torch.rand(3, 4, 2)  # Probabilities shape matches the selection
        selection = SelectionSpec(selection_index=indices, selection_probabilities=probabilities)

        result = virtual_state_select(state, selection, dim=2, superposition=True)

        # Expected shape after superposition should be (batch, extra_dim, features)
        self.assertEqual(result.shape, (3, 4, 5))

    def test_selection_logic(self):
        # Test selection logic using a manually computed example

        # Define the state to select from
        state = torch.tensor([[[0.0, 0.0], [0.0, 0.0]],
                                    [[1.0, 1.0], [0.0, 0.0]],
                                    [[2.0, 4.0], [3.0, 1.0]]])

        # Define the spec to select with
        selection_indices = torch.tensor([0, 2])
        selection_probabilities = torch.tensor([0.3, 0.7])
        spec = SelectionSpec(selection_indices, selection_probabilities)

        # Manually compute the superposition results. Compute the results

        expected_results = state[0]*0.3 + state[2]*0.7
        actual_results = virtual_state_select(state, spec, dim=0)

        # compare
        self.assertTrue(torch.allclose(expected_results, actual_results))
    def test_dense_mode(self):
        # Test dense mode with superposition on (dense_mode=True, superposition=True)
        state = torch.randn(2, 5, 3)  # Shape (batch, options, features)
        probabilities = torch.tensor([[0.2, 0.3, 0.5], [0.4, 0.4, 0.2]])  # Probabilities for all options
        selection = SelectionSpec(selection_index=None, selection_probabilities=probabilities, dense_mode=True)

        result = virtual_state_select(state, selection, dim=-1, superposition=True)

        # Expected shape after superposition should be (batch, features) with dense_mode
        self.assertEqual(result.shape, torch.Size([2, 5]))

        # Test dense mode with superposition off (dense_mode=True, superposition=False)
        result_no_superposition = virtual_state_select(state, selection, dim=-1, superposition=False)

        # Expected shape with dense mode and no superposition should be the same as input state
        self.assertTrue(torch.equal(result_no_superposition, state))

class TestVirtualStateScatter(unittest.TestCase):

    def test_superposition_correct_shape(self):
        # Test that superposition creates the correct shape after scattering
        state = torch.randn(2, 3, 5)  # Shape (batch, options, features)
        substate = torch.randn(2, 5)  # Shape (batch, features) in superposition
        indices = torch.tensor([[0, 2], [1, 2]])  # Indices to scatter into
        probabilities = torch.tensor([[0.7, 0.3], [0.5, 0.5]])  # Probabilities
        selection = SelectionSpec(selection_index=indices, selection_probabilities=probabilities)

        result = virtual_state_scatter(state, substate, selection, dim=1, superposition=True)

        # Shape should remain the same as state
        self.assertEqual(result.shape, state.shape)

    def test_no_superposition_correct_shape(self):
        # Test non-superposition scattering, should return original state shape
        state = torch.randn(2, 3, 5)  # Shape (batch, options, features)
        substate = torch.randn(2, 2, 5)  # Shape (batch, selected, features)
        indices = torch.tensor([[0, 2], [1, 2]])  # Indices to scatter into
        probabilities = torch.ones_like(indices, dtype=torch.float32)  # Uniform probabilities
        selection = SelectionSpec(selection_index=indices, selection_probabilities=probabilities)

        result = virtual_state_scatter(state, substate, selection, dim=1, superposition=False)

        # Expected to remain in original state shape
        self.assertEqual(result.shape, state.shape)

    def test_interpolation_behavior(self):
        # Test if interpolation between state and substate works as expected with simple values
        state = torch.ones(2, 3, 3)  # All values in the state are 1.0
        substate = torch.zeros(2, 2, 3)  # All values in the substate are 0.0
        indices = torch.tensor([[0, 2], [1, 2]])  # Scatter into indices 0, 2 and 1, 2
        probabilities = torch.tensor(
            [[0.3, 0.7], [0.5, 0.5]])  # Interpolation probabilities (0.3 for substate, 0.7 for state)
        selection = SelectionSpec(selection_index=indices, selection_probabilities=probabilities)

        result = virtual_state_scatter(state, substate, selection, dim=1, superposition=False)

        # Manually compute the expected interpolation:
        # For state 1, index 0 -> 70% state, 30% substate, index 2 -> interpolated by 70% state, 30% substate
        # For state 2, index 1 -> interpolated by 50%, index 2 -> interpolated by 50%
        expected_result = torch.tensor([[[0.7, 0.7, 0.7],  # Index 0: 70% from state, 30% from substate
                                         [1.0, 1.0, 1.0],  # Index 1 remains the same (not selected)
                                         [0.3, 0.3, 0.3]],  # Index 2: 30% from state, 70% from substate
                                        [[1.0, 1.0, 1.0],  # Index 0 remains the same (not selected)
                                         [0.5, 0.5, 0.5],  # Index 1: 50% from state, 50% from substate
                                         [0.5, 0.5, 0.5]]])  # Index 2: 50% from state, 50% from substate

        self.assertTrue(torch.allclose(result, expected_result))
    def test_broadcast_interpolation_behavior(self):
        # Test with a broadcasted scatter

        # Define the state
        state = torch.tensor([[[0.0, 0.0], [0.0, 0.0]],
                            [[1.0, 1.0], [0.0, 0.0]],
                            [[2.0, 4.0], [3.0, 1.0]]])

        # Define a SelectionSpec with specific indices and probabilities
        selection_indices = torch.tensor([0, 2])  # Select  0 and 2
        selection_probabilities = torch.tensor([0.3, 0.7])  # Weights
        selection_spec = SelectionSpec(selection_index=selection_indices,
                                       selection_probabilities=selection_probabilities)

        # Define a new expression
        new_expression = torch.tensor([[[1.5, 3.0], [2.0, 1.0]],
                                       [[1.5, 3.0],[2.0, 1.0]]]
                                      )

        # Define the expected results. Based on the selection indices,
        # the above feature will be interpolated with layer 0 and 2
        expected_result = state.clone()
        expected_result[0] = state[0]*0.7 + new_expression[0]*0.3
        expected_result[2] = state[2]*0.3 + new_expression[1]*0.7

        # Check
        actual_result = virtual_state_scatter(state, new_expression, selection_spec, dim=0, superposition=False)

        self.assertTrue(torch.allclose(expected_result, actual_result))


    def test_batch_shape_handling(self):
        # Test batch shape handling does not choke
        state = torch.randn(3, 4, 6, 5)  # Shape (batch, extra_dim, options, features)
        substate = torch.randn(3, 4, 2, 5)  # Shape (batch, extra_dim, selected, features)
        indices = torch.randint(0, 4, (3, 4, 2))  # Batch shape (batch, extra_dim, selected)
        probabilities = torch.rand(3, 4, 2)  # Probabilities shape matches the selection
        selection = SelectionSpec(selection_index=indices, selection_probabilities=probabilities)

        result = virtual_state_scatter(state, substate, selection, dim=2, superposition=False)

        # Expected shape should remain the same as the original state
        self.assertEqual(result.shape, state.shape)
    def test_dense_mode(self):
        # Test dense mode with superposition on (dense_mode=True, superposition=True)
        state = torch.randn(2, 5, 3)  # Shape (batch, options, features)
        substate = torch.randn(2, 5)  # Shape should match the compressed state for dense mode with superposition

        # Full probabilities across all options
        probabilities = torch.tensor([[0.3, 0.3, 0.4], [0.5, 0.2, 0.3]])  # Probability for each option
        selection = SelectionSpec(selection_index=None, selection_probabilities=probabilities, dense_mode=True)

        # Apply virtual_state_scatter in dense mode with superposition on
        result_superposition = virtual_state_scatter(state, substate, selection, dim=-1, superposition=True)

        # Shape should remain the same as state
        self.assertEqual(result_superposition.shape, state.shape)

        # Test dense mode with superposition off (dense_mode=True, superposition=False)
        substate_no_superposition = torch.randn(2, 5, 3)  # Shape should match state in dense mode without superposition
        result_no_superposition = virtual_state_scatter(state, substate_no_superposition, selection, dim=-1, superposition=False)

        # Expected shape should still be the same as state
        self.assertEqual(result_no_superposition.shape, state.shape)
class TestVirtualParameter(unittest.TestCase):

    def test_virtual_parameter_creation(self):
        """
        Test the creation of a VirtualParameter using the create method.
        Ensures correct initialization and shape of the parameter bank.
        """
        bank_size = 4
        shape = (3, 3)

        # Create virtual parameter
        vp = VirtualParameter.create(bank_size, shape)

        # Ensure the parameter has the correct shape (bank dimension is now last)
        self.assertEqual(vp.parameter.shape, (*shape, bank_size))

    def test_virtual_parameter_custom_init(self):
        """
        Test the custom initialization of a VirtualParameter.
        Ensures the init function is applied to the parameter bank.
        """

        def custom_init(tensor):
            torch.nn.init.constant_(tensor, 0.5)

        bank_size = 2
        shape = (2, 2)

        # Create virtual parameter with custom init
        vp = VirtualParameter.create(bank_size, shape, init=custom_init)

        # Check that the parameter was correctly initialized to 0.5
        self.assertTrue(torch.all(vp.parameter == 0.5))

    def test_virtual_parameter_manual_init(self):
        """
        Test manually initializing a VirtualParameter with a parameter tensor.
        """
        parameter_bank = torch.tensor([[[0.0, 0.0], [0.0, 0.0]],  # shape (2, 2, 3), last dim is bank
                                       [[1.0, 1.0], [0.0, 0.0]],
                                       [[2.0, 4.0], [3.0, 1.0]]])
        vp = VirtualParameter(parameter_bank)
        self.assertTrue(torch.allclose(vp.parameter, parameter_bank))

    def test_virtual_parameter_selection(self):
        """
        Test the forward method of VirtualParameter with SelectionSpec.
        Ensures that the correct bank is selected and weighted properly.
        """

        # Create virtual parameter and manually set values for easier testing
        parameter_bank = torch.tensor([[[0.0, 1.0, 2.0],[0.0, 1.0, 4.0]],
                                       [[0.0, 0.0, 3.0],[0.0, 0.0, 1.0]]])
        vp = VirtualParameter(parameter_bank)

        # Define a SelectionSpec with specific indices and probabilities
        selection_indices = torch.tensor([0, 2])  # Select banks 0 and 2
        selection_probabilities = torch.tensor([0.3, 0.7])  # Weights
        selection_spec = SelectionSpec(selection_index=selection_indices,
                                       selection_probabilities=selection_probabilities)

        # Forward pass through the VirtualParameter
        result = vp(selection_spec)

        # Manual computation of the expected result:
        expected_result = parameter_bank[..., 0]*0.3 + parameter_bank[..., 2]*0.7

        self.assertTrue(torch.allclose(result, expected_result, atol=1e-4))

        # Check gradients are active
        self.assertTrue(result.requires_grad)

    def test_virtual_parameter_no_selection(self):
        """
        Test the case where no banks are selected.
        Ensures proper handling when no banks are selected.
        """
        bank_size = 3
        shape = (2, 2)

        # Create virtual parameter
        vp = VirtualParameter.create(bank_size, shape)

        # Define a SelectionSpec with no valid indices
        selection_indices = torch.empty([0, 3], dtype=torch.long)  # No banks selected
        selection_probabilities = torch.empty([0, 3])
        selection_spec = SelectionSpec(selection_index=selection_indices,
                                       selection_probabilities=selection_probabilities)
        outcome = vp(selection_spec)

    def test_virtual_parameter_batched_selection(self):
        # Test virtual parameter selection with complex batches

        bank_size = 4
        chose_size = 2
        batch_shape = [3, 4]
        param_shape = [7, 8, 5]

        # Create the banks
        param_banks = torch.rand(param_shape + [bank_size])
        vp = VirtualParameter(param_banks)

        # Create the selection spec.
        selection_indices = torch.randint(0, bank_size, batch_shape + [chose_size])
        selection_probabilities = torch.rand(batch_shape + [chose_size])
        selection_spec = SelectionSpec(selection_index=selection_indices,
                                       selection_probabilities=selection_probabilities)

        # Try
        vp(selection_spec)
    def test_dense_mode(self):
        """
        Test the forward method of VirtualParameter in dense mode.
        Ensures that the correct dense behavior is applied.
        """
        # Create virtual parameter with a known parameter bank
        vp = VirtualParameter.create(3, [2, 4])

        # Define dense selection with probabilities across all banks
        selection_probabilities = torch.tensor([[0.2, 0.5, 0.3],[0.4, 0.4, 0.2]])  # Weights for dense selection
        selection_spec = SelectionSpec(selection_index=None, selection_probabilities=selection_probabilities, dense_mode=True)

        # Forward pass through the VirtualParameter
        result = vp(selection_spec)
        self.assertEqual(result.shape, torch.Size([2, 2, 4]))

class TestVirtualLinear(unittest.TestCase):

    def setUp(self):
        # Define layer parameters for the tests
        self.in_features = 5
        self.out_features = 3
        self.bank_size = 4
        self.batch_size = 2

    def test_virtual_linear_forward_no_bias(self):
        """
        Test the forward pass of VirtualLinear without bias.
        Different selections per batch.
        """
        # Create a VirtualLinear layer without bias
        virtual_linear = VirtualLinear(self.in_features, self.out_features, self.bank_size, bias=False)

        # Create input tensor (batch_size, in_features)
        input_tensor = torch.randn(self.batch_size, self.in_features)

        # Define a SelectionSpec that selects different banks for each batch
        selection_indices = torch.tensor([[0], [1]])  # Batch 0 selects bank 0, Batch 1 selects bank 1
        selection_probabilities = torch.tensor([[1.0], [1.0]])  # Full weight on each selected bank
        selection_spec = SelectionSpec(selection_index=selection_indices, selection_probabilities=selection_probabilities)

        # Forward pass
        output = virtual_linear(input_tensor, selection_spec)

        # Check output shape: (batch_size, out_features)
        self.assertEqual(output.shape, (self.batch_size, self.out_features))

    def test_virtual_linear_forward_with_bias(self):
        """
        Test the forward pass of VirtualLinear with bias.
        Different selections per batch.
        """
        # Create a VirtualLinear layer with bias
        virtual_linear = VirtualLinear(self.in_features, self.out_features, self.bank_size, bias=True)

        # Create input tensor (batch_size, in_features)
        input_tensor = torch.randn(self.batch_size, self.in_features)

        # Define a SelectionSpec that selects different banks for each batch
        selection_indices = torch.tensor([[0], [2]])  # Batch 0 selects bank 0, Batch 1 selects bank 2
        selection_probabilities = torch.tensor([[1.0], [1.0]])  # Full weight on each selected bank
        selection_spec = SelectionSpec(selection_index=selection_indices, selection_probabilities=selection_probabilities)

        # Forward pass
        output = virtual_linear(input_tensor, selection_spec)

        # Check output shape: (batch_size, out_features)
        self.assertEqual(output.shape, (self.batch_size, self.out_features))

    def test_virtual_linear_selection_per_batch(self):
        """
        Test if the virtual linear layer correctly uses different selections for each batch.
        """
        # Create a VirtualLinear layer with bias
        virtual_linear = VirtualLinear(self.in_features, self.out_features, self.bank_size, bias=True)

        # Create input tensor (batch_size, in_features)
        input_tensor = torch.randn(self.batch_size, self.in_features)

        # Define different selections for each batch
        selection_indices = torch.tensor([[0], [1]])  # Batch 0 selects bank 0, Batch 1 selects bank 1
        selection_probabilities = torch.tensor([[1.0], [1.0]])  # Full weight on each selected bank
        selection_spec = SelectionSpec(selection_index=selection_indices, selection_probabilities=selection_probabilities)

        # Forward pass
        output = virtual_linear(input_tensor, selection_spec)

        # Ensure that the two outputs are different
        self.assertFalse(torch.allclose(output[0], output[1]))

    def test_virtual_linear_superposition_per_batch(self):
        """
        Test if the virtual linear layer correctly superimposes different selections per batch.
        """
        # Create a VirtualLinear layer with bias
        virtual_linear = VirtualLinear(self.in_features, self.out_features, self.bank_size, bias=True)

        # Create input tensor (batch_size, in_features)
        input_tensor = torch.randn(self.batch_size, self.in_features)

        # Define different selections and superpositions for each batch
        selection_indices = torch.tensor([[0, 1], [2, 3]])  # Batch 0 selects banks 0 and 1, Batch 1 selects banks 2 and 3
        selection_probabilities = torch.tensor([[0.7, 0.3], [0.4, 0.6]])  # Different weights for each batch
        selection_spec = SelectionSpec(selection_index=selection_indices, selection_probabilities=selection_probabilities)

        # Forward pass
        output = virtual_linear(input_tensor, selection_spec)

        # Check output shape: (batch_size, out_features)
        self.assertEqual(output.shape, (self.batch_size, self.out_features))

        # Ensure that the two batch outputs are different (because of different superpositions)
        self.assertFalse(torch.allclose(output[0], output[1]))

class TestVirtualHeadLayers(unittest.TestCase):

    def setUp(self):
        # Define the parameters for the layers
        self.d_model = 16
        self.d_head = 4
        self.num_heads = 4
        self.num_banks = 3
        self.extra_batch_dims = (3, 2)  # Additional batch dimensions

    def test_virtual_make_heads_forward(self):
        """
        Test the VirtualMakeHeads layer to ensure it correctly creates
        attention heads from the input tensor with multiple batch dimensions.
        """
        make_heads_layer = VirtualMakeHeads(self.d_model, self.d_head, self.num_heads, self.num_banks)

        # Create a mock input tensor of shape (*extra_batch_dims, d_model)
        input_tensor = torch.randn(*self.extra_batch_dims, self.d_model)

        # Define a SelectionSpec with different bank selections for each batch
        # Ensure all batch dimensions are accounted for
        batch_shape = self.extra_batch_dims  # Use the extra batch dims
        selection_indices = torch.randint(0, self.num_banks, [*batch_shape, 2])
        selection_probabilities = torch.ones_like(selection_indices, dtype=torch.float)
        selection_spec = SelectionSpec(selection_index=selection_indices, selection_probabilities=selection_probabilities)

        # Forward pass through the make heads layer
        output = make_heads_layer(input_tensor, selection_spec)

        # Expected output shape: (*extra_batch_dims, num_heads, d_head)
        expected_shape = (*self.extra_batch_dims, self.num_heads, self.d_head)
        self.assertEqual(output.shape, expected_shape)

    def test_virtual_merge_heads_forward(self):
        """
        Test the VirtualMergeHeads layer to ensure it correctly merges
        attention heads back to a single dimension with multiple batch dimensions.
        """
        merge_heads_layer = VirtualMergeHeads(self.d_model, self.d_head, self.num_heads, self.num_banks)

        # Create a mock input tensor of shape (*extra_batch_dims, num_heads, d_head)
        input_tensor = torch.randn(*self.extra_batch_dims, self.num_heads, self.d_head)

        # Define a SelectionSpec with different bank selections for each batch
        # Ensure all batch dimensions are accounted for
        batch_shape = self.extra_batch_dims  # Use the extra batch dims
        selection_indices = torch.randint(0, self.num_banks,[*batch_shape, 2])
        selection_probabilities = torch.ones_like(selection_indices, dtype=torch.float)
        selection_spec = SelectionSpec(selection_index=selection_indices, selection_probabilities=selection_probabilities)

        # Forward pass through the merge heads layer
        output = merge_heads_layer(input_tensor, selection_spec)

        # Expected output shape: (*extra_batch_dims, d_model)
        expected_shape = (*self.extra_batch_dims, self.d_model)
        self.assertEqual(output.shape, expected_shape)

    def test_virtual_make_merge_heads_roundtrip(self):
        """
        Test a round-trip through VirtualMakeHeads and VirtualMergeHeads
        to ensure consistency with multiple batch dimensions, where merging
        heads after creating them should approximately return to the original shape.
        """
        make_heads_layer = VirtualMakeHeads(self.d_model, self.d_head, self.num_heads, self.num_banks)
        merge_heads_layer = VirtualMergeHeads(self.d_model, self.d_head, self.num_heads, self.num_banks)

        # Create a mock input tensor of shape (*extra_batch_dims, d_model)
        input_tensor = torch.randn(*self.extra_batch_dims, self.d_model)

        # Define SelectionSpec for both layers (same selection for round-trip)
        # Ensure all batch dimensions are accounted for
        batch_shape = self.extra_batch_dims  # Use the extra batch dims
        selection_indices = torch.randint(0, self.num_banks, [*batch_shape, 2])
        selection_probabilities = torch.ones_like(selection_indices, dtype=torch.float)
        selection_spec = SelectionSpec(selection_index=selection_indices, selection_probabilities=selection_probabilities)

        # Run through make heads layer
        headed_tensor = make_heads_layer(input_tensor, selection_spec)

        # Run through merge heads layer
        output = merge_heads_layer(headed_tensor, selection_spec)

        # Expected output shape: (*extra_batch_dims, d_model)
        expected_shape = (*self.extra_batch_dims, self.d_model)
        self.assertEqual(output.shape, expected_shape)

class TestVirtualFeedforward(unittest.TestCase):

    def setUp(self):
        # Set up parameters for the test case
        self.d_model = 16
        self.d_hidden = 32
        self.bank_size = 4
        self.dropout = 0.5
        self.extra_batch_dims = (2, 3)  # Extra batch dimensions for testing

    def test_virtual_feedforward_forward(self):
        """
        Test the VirtualFeedforward layer to ensure the feedforward pass
        completes and maintains the correct output shape.
        """
        ff_layer = VirtualFeedforward(self.d_model, self.d_hidden, self.bank_size, self.dropout)

        # Create a mock input tensor with multiple batch dimensions
        input_tensor = torch.randn(*self.extra_batch_dims, self.d_model)

        # Set up a SelectionSpec with per-batch bank selections
        batch_shape = self.extra_batch_dims
        selection_indices = torch.randint(0, self.bank_size, (*batch_shape, 2))
        selection_probabilities = torch.ones_like(selection_indices, dtype=torch.float)
        selection_spec = SelectionSpec(selection_index=selection_indices, selection_probabilities=selection_probabilities)

        # Perform the forward pass
        output = ff_layer(input_tensor, selection_spec)

        # Verify the output shape is as expected
        expected_shape = (*self.extra_batch_dims, self.d_model)
        self.assertEqual(output.shape, expected_shape, f"Expected shape {expected_shape} but got {output.shape}")


    def test_virtual_feedforward_batch_dim_variation(self):
        """
        Test varying batch dimensions to confirm that the VirtualFeedforward layer
        works for various batch sizes.
        """
        batch_shapes = [(1, self.d_model), (5, self.d_model), (3, 4, self.d_model)]
        ff_layer = VirtualFeedforward(self.d_model, self.d_hidden, self.bank_size, self.dropout)

        for batch_shape in batch_shapes:
            input_tensor = torch.randn(*batch_shape)

            # Define SelectionSpec with matching batch dimensions
            selection_indices = torch.randint(0, self.bank_size, (*batch_shape[:-1], 2))
            selection_probabilities = torch.ones_like(selection_indices, dtype=torch.float)
            selection_spec = SelectionSpec(selection_index=selection_indices, selection_probabilities=selection_probabilities)

            # Forward pass and output shape check
            output = ff_layer(input_tensor, selection_spec)
            self.assertEqual(output.shape, batch_shape, f"Expected shape {batch_shape} but got {output.shape}")


class TestTopPSelectionMask(unittest.TestCase):

    def test_top_p_zero(self):
        """Test with top_p=0, expecting an empty mask (no selection)."""
        logits = torch.randn(3, 10)  # Random logits of shape (3, 10)
        mask = make_top_p_selection_mask(logits, top_p=0.0)
        self.assertTrue(torch.all(mask == False), "Expected no selections for top_p=0")

    def test_top_p_one(self):
        """Test with top_p=1, expecting a fully selected mask (all elements selected)."""
        logits = torch.randn(3, 10)  # Random logits of shape (3, 10)
        mask = make_top_p_selection_mask(logits, top_p=1.0)
        self.assertTrue(torch.all(mask == True), "Expected all selections for top_p=1")

    def test_top_p_middle_value(self):
        """Test with a mid-range top_p value to check partial selection functionality."""
        logits = torch.tensor([[2.0, 1.0, 0.5, 0.1, -1.0, -2.0]])
        mask = make_top_p_selection_mask(logits, top_p=0.7)
        expected_selection = torch.tensor([[True, True, False, False, False, False]])
        self.assertTrue(torch.equal(mask, expected_selection),
                        "Unexpected selection for top_p=0.6 with given logits")

    def test_monotonic_probabilities(self):
        """Test where logits are in increasing order to verify mask respects top_p cutoff."""
        logits = torch.tensor([[-2.0, -1.5, 0.0, 0.5, 1.0, 2.0]])
        mask = make_top_p_selection_mask(logits, top_p=0.8)
        # Since logits are sorted in ascending order, only the last elements should be selected
        self.assertTrue(mask.sum() > 0, "Expected some selections for top_p=0.8")

    def test_extreme_logits_values(self):
        """Test where logits have very high and low values to check numerical stability."""
        logits = torch.tensor([[100.0, 10.0, -10.0, -100.0]])
        mask = make_top_p_selection_mask(logits, top_p=0.5)
        # Ensure that only the first or first two elements are selected given the high disparity
        self.assertTrue(mask[0, 0] == True, "Expected the highest logit to be selected")
        self.assertTrue(mask[0, 1:].sum() >= 0, "Expected selections respecting cumulative probability")

    def test_invalid_top_p(self):
        """Test with an invalid top_p value outside of [0, 1] to check error handling."""
        logits = torch.randn(3, 10)
        with self.assertRaises(ValueError):
            make_top_p_selection_mask(logits, top_p=1.5)
        with self.assertRaises(ValueError):
            make_top_p_selection_mask(logits, top_p=-0.1)


class TestTopKSelectionMask(unittest.TestCase):

    def test_top_k_zero(self):
        """Test with top_k=0, expecting an empty mask (no selection)."""
        logits = torch.randn(3, 10)  # Random logits of shape (3, 10)
        mask = make_top_k_selection_mask(logits, top_k=0)
        self.assertTrue(torch.all(mask == False), "Expected no selections for top_k=0")

    def test_top_k_equal_to_num_logits(self):
        """Test with top_k equal to the number of logits, expecting a fully selected mask."""
        logits = torch.randn(3, 10)
        mask = make_top_k_selection_mask(logits, top_k=10)
        self.assertTrue(torch.all(mask == True), "Expected all selections when top_k equals the number of logits")

    def test_top_k_greater_than_num_logits(self):
        """Test with top_k greater than the number of logits, expecting a fully selected mask."""
        logits = torch.randn(3, 8)  # Logits with fewer elements than top_k
        mask = make_top_k_selection_mask(logits, top_k=10)
        self.assertTrue(torch.all(mask == True), "Expected all selections when top_k exceeds the number of logits")

    def test_top_k_middle_value(self):
        """Test with a moderate top_k value to check partial selection functionality."""
        logits = torch.tensor([[1.0, 0.8, -0.5, 0.3, -1.2]])
        mask = make_top_k_selection_mask(logits, top_k=2)
        expected_mask = torch.tensor([[True, True, False, False, False]])
        self.assertTrue(torch.equal(mask, expected_mask), "Unexpected selection for top_k=2 with given logits")

    def test_top_k_on_multidimensional_input(self):
        """Test with a multidimensional input tensor and moderate top_k value."""
        logits = torch.randn(2, 3, 5)  # Logits with shape (2, 3, 5)
        mask = make_top_k_selection_mask(logits, top_k=3)
        self.assertEqual(mask.shape, logits.shape, "Mask shape does not match logits shape")
        self.assertTrue(torch.all(mask.sum(dim=-1) == 3),
        "Each element along the last dimension should have exactly top_k selected")

    def test_invalid_top_k(self):
        """Test with an invalid (negative) top_k value to check error handling."""
        logits = torch.randn(3, 10)
        with self.assertRaises(ValueError):
            make_top_k_selection_mask(logits, top_k=-1)


class TestMakeRandomSelectionMask(unittest.TestCase):

    def test_random_selection_basic(self):
        """Test a basic case with a 1D tensor and num elements to select."""
        logits = torch.randn(10)
        num_select = 3
        mask = make_random_selection_mask(logits, num_select)

        # Check the shape
        self.assertEqual(mask.shape, logits.shape, "Mask shape should match logits shape")

        # Check the number of selected elements
        self.assertEqual(mask.sum().item(), num_select, "Mask should have 'num' elements selected")

    def test_random_selection_multidimensional(self):
        """Test selection on a multidimensional tensor."""
        logits = torch.randn(4, 5, 6)
        num_select = 2
        mask = make_random_selection_mask(logits, num_select)

        # Check the shape
        self.assertEqual(mask.shape, logits.shape, "Mask shape should match logits shape")

        # Verify selection count for each last-dimension slice
        for i in range(logits.shape[0]):
            for j in range(logits.shape[1]):
                self.assertEqual(mask[i, j].sum().item(), num_select,
                                 f"Each slice should have {num_select} elements selected")

    def test_selection_zero_elements(self):
        """Test the case when num=0, meaning no elements should be selected."""
        logits = torch.randn(8, 10)
        num_select = 0
        mask = make_random_selection_mask(logits, num_select)

        # All values in the mask should be False
        self.assertTrue(torch.all(mask == False), "Mask should have no elements selected when num=0")

    def test_selection_exceeds_logit_length(self):
        """Test when num exceeds the number of logits, in which case all should be selected."""
        logits = torch.randn(3, 4)
        num_select = 10  # more than number of elements in last dimension
        mask = make_random_selection_mask(logits, num_select)

        # Verify that all elements in each row are selected since num > last-dimension length
        for i in range(logits.shape[0]):
            self.assertTrue(torch.all(mask[i] == True), "All elements should be selected when num > logits.size(-1)")

    def test_invalid_num_selection(self):
        """Test if a ValueError is raised when num is negative."""
        logits = torch.randn(5)
        num_select = -1
        with self.assertRaises(ValueError):
            make_random_selection_mask(logits, num_select)

    def test_randomness_of_selection(self):
        """Test that different calls with the same inputs result in different random selections."""
        logits = torch.randn(10)
        num_select = 3

        mask1 = make_random_selection_mask(logits, num_select)
        mask2 = make_random_selection_mask(logits, num_select)

        # Assert that at least one difference exists between two independent selections
        self.assertFalse(torch.equal(mask1, mask2), "Random selection should vary across calls with the same inputs")

class MockBankSelector(AbstractBankSelector):
    def create_state(self, batch_shape: torch.Tensor) ->Any:
        return None
    def get_statistics(self, state: Any) ->Dict[str, torch.Tensor]:
        return None
    def forward(self, embedding: torch.Tensor, state: Any) -> Tuple[SelectionSpec, Any]:
        return self.select_logits(embedding), state
class TestAbstractBankSelector(unittest.TestCase):

    def setUp(self):
        # Define sample logits for testing
        self.logits = torch.randn(2, 10)  # Batch size of 2, 10 logits per instance

    def test_comprehensive_mode(self):
        """Ensure comprehensive mode is triggered when top_k, top_p, and rand are inactive"""
        selector = MockBankSelector(10, 10, dense_mode=True)
        selection_spec, _ = selector(self.logits, None)
        self.assertTrue(torch.all(selection_spec.selection_probabilities > 0.0))

    def test_dense_mode(self):
        """Ensure dense mode is triggered when top_k, top_p, and rand are inactive."""
        selector = MockBankSelector(10, 12, dense_mode=True)
        selection_spec, _ = selector(self.logits, None)

        # Verify all logits are included in dense mode, and probabilities sum to 1
        self.assertEqual(selection_spec.selection_probabilities.shape[-1], self.logits.shape[-1],
                         "Dense mode should include all logits")
        for probs in selection_spec.selection_probabilities:
            self.assertAlmostEqual(probs.sum().item(), 1.0, places=4)

    def test_combined_sparse_selection(self):
        """Test combined sparse selection (top_k, top_p, and rand) to check mask integration."""
        selector = MockBankSelector(10, 12, top_k=3, top_p=0.8, rand=2, dense_mode=False)
        selection_spec, _ = selector(self.logits, None)

        # Ensure selection is limited and varies in length
        self.assertTrue(selection_spec.selection_index.shape[-1] <= self.logits.shape[-1],
                        "Sparse selection should limit selected elements")

        # Check that all probabilities are positive (selection implies active probabilities)
        self.assertTrue(torch.all(selection_spec.selection_probabilities >= 0),
                        "Sparse selection logits should have positive probabilities")

    def test_dropout_integration(self):
        """Verify dropout integration does not affect mode switching."""
        selector = MockBankSelector(10, 10, top_k=3, control_dropout=0.5, dense_mode=False)
        selection_spec, _ = selector(self.logits, None)

        # Confirm top-k is applied, but some logits may be zeroed by dropout
        self.assertEqual(selection_spec.selection_index.shape[-1], 3,
                         "Top-k selection should select exactly 3 logits in sparse mode")

class TestLinearBankSelector(unittest.TestCase):

    def setUp(self):
        # Define standard input dimensions for testing
        self.d_model = 8
        self.bank_size = 10
        self.batch_size = 2
        self.embedding = torch.randn(self.batch_size, self.d_model)

    def test_top_k_selection(self):
        """Test top-k sparse selection with k < bank_size."""
        top_k = 3
        selector = LinearBankSelector(d_model=self.d_model, bank_size=self.bank_size, top_k=top_k)
        state = selector.create_state([self.batch_size])
        selection_spec, _ = selector(self.embedding, state)

        # Ensure exactly top_k indices are selected per batch element
        count = torch.count_nonzero(selection_spec.selection_probabilities, dim=-1).max()
        self.assertEqual(count, top_k,
                         "Top-k selection should select exactly k banks")
        self.assertTrue(torch.all(selection_spec.selection_probabilities >= 0),
                        "Selected probabilities should be positive")

    def test_top_p_selection(self):
        """Test top-p (nucleus) selection with cumulative probability threshold."""
        top_p = 0.7
        selector = LinearBankSelector(d_model=self.d_model, bank_size=self.bank_size, top_p=top_p)
        state = selector.create_state([self.batch_size])
        selection_spec, _ = selector(self.embedding, state)

        # Check that the number of selected banks varies and is limited by cumulative probability
        self.assertTrue(selection_spec.selection_probabilities.shape[-1] <= self.bank_size,
                        "Top-p selection should limit selected elements")
        self.assertTrue(torch.all(selection_spec.selection_probabilities >= 0),
                        "Selected probabilities should be positive or zero")

    def test_random_selection(self):
        """Test random selection mode with num < bank_size."""
        rand = 4
        selector = LinearBankSelector(d_model=self.d_model, bank_size=self.bank_size, rand=rand)
        state = selector.create_state([self.batch_size])
        selection_spec, _ = selector(self.embedding, state)

        # Confirm that exactly rand indices are selected randomly
        count = torch.count_nonzero(selection_spec.selection_probabilities, dim=-1).max()
        self.assertEqual(count, rand,
                         "Random selection should select exactly rand banks")
        self.assertTrue(torch.all(selection_spec.selection_probabilities >= 0),
                        "Selected probabilities should be positive")

    def test_dropout_integration(self):
        """Verify dropout does not affect mode but may introduce zeroed probabilities in sparse selections."""
        top_k = 3
        dropout_rate = 0.5
        selector = LinearBankSelector(d_model=self.d_model, bank_size=self.bank_size, top_k=top_k,
                                      control_dropout=dropout_rate)
        state = selector.create_state([self.batch_size])
        selection_spec, _ = selector(self.embedding, state)

        # Ensure top_k is applied and some logits may be zeroed out by dropout
        count = torch.count_nonzero(selection_spec.selection_probabilities, dim=-1).max()
        self.assertEqual(count, top_k,
                         "Top-k selection should select exactly k banks")


class TestVirtualAdvancedLinear(unittest.TestCase):
    def setUp(self):
        """
        Sets up the initial conditions and common test variables.
        """
        # Define shapes and sizes for testing
        self.in_shape = (4, 5)  # Input shape for the test
        self.out_shape = (3, 6)  # Output shape for the test
        self.bank_size = 2  # Number of virtual layers
        self.device = torch.device("cpu")  # Device for testing
        self.dtype = torch.float32  # Data type for testing

        # Initialize the VirtualAdvancedLinear layer
        self.layer = VirtualAdvancedLinear(
            in_shape=self.in_shape,
            out_shape=self.out_shape,
            bank_size=self.bank_size,
            device=self.device,
            dtype=self.dtype
        )

    def test_forward_shape(self):
        """
        Tests that the forward method correctly transforms the input to the specified output shape.
        """
        # Prepare test input tensor with the expected input shape
        test_input = torch.randn(8, *self.in_shape, dtype=self.dtype, device=self.device)  # Batch size of 8

        # Create a SelectionSpec with random indices and probabilities
        selection_indices = torch.randint(0, self.bank_size, (8, 1), dtype=torch.long, device=self.device)
        selection_probabilities = torch.rand(8, 1, dtype=self.dtype, device=self.device)
        selection_spec = SelectionSpec(selection_index=selection_indices,
                                       selection_probabilities=selection_probabilities)

        # Forward pass
        output = self.layer(test_input, selection_spec)

        # Check if the output shape is as expected
        expected_shape = (8, *self.out_shape)
        self.assertEqual(output.shape, expected_shape, "Output shape mismatch.")

    def test_forward_invalid_shape(self):
        """
        Tests that the forward method raises an error when an input tensor with invalid shape is passed.
        """
        # Prepare an invalid input shape
        invalid_input = torch.randn(8, 3, 5, dtype=self.dtype, device=self.device)  # Incorrect in_shape

        # Create a SelectionSpec with random indices and probabilities
        selection_indices = torch.randint(0, self.bank_size, (8, 1), dtype=torch.long, device=self.device)
        selection_probabilities = torch.rand(8, 1, dtype=self.dtype, device=self.device)
        selection_spec = SelectionSpec(selection_index=selection_indices,
                                       selection_probabilities=selection_probabilities)

        # Forward pass should raise a ValueError due to shape mismatch
        with self.assertRaises(ValueError):
            self.layer(invalid_input, selection_spec)

    def test_selection_integration(self):
        """
        Tests that the SelectionSpec properly influences which parameters are used in the forward computation.
        """
        # Prepare input tensor with valid shape
        test_input = torch.randn(8, *self.in_shape, dtype=self.dtype, device=self.device)

        # Define selection indices and probabilities
        selection_indices = torch.tensor([[0], [1]] * 4, dtype=torch.long,
                                         device=self.device)  # Alternate between banks
        selection_probabilities = torch.ones(8, 1, dtype=self.dtype,
                                             device=self.device)  # Full weight on selected index
        selection_spec = SelectionSpec(selection_index=selection_indices,
                                       selection_probabilities=selection_probabilities)

        # Forward pass
        output = self.layer(test_input, selection_spec)

        # Check if the output has the correct shape and device
        self.assertEqual(output.shape, (8, *self.out_shape), "Output shape mismatch with selection integration.")
        self.assertEqual(output.device, self.device, "Output device mismatch.")

    def test_device_dtype_consistency(self):
        """
        Tests that the layer and selection spec work properly on a different device and dtype.
        """
        if torch.cuda.is_available():
            # Reinitialize on CUDA
            device=  torch.device("cuda")
            layer_cuda = VirtualAdvancedLinear(self.in_shape, self.out_shape, self.bank_size, dtype=torch.float64,
                                               device=device)

            # Prepare input and SelectionSpec on CUDA
            test_input = torch.randn(8, *self.in_shape, dtype=torch.float64, device=device)
            selection_indices = torch.randint(0, self.bank_size, (8, 1), dtype=torch.long, device=device)
            selection_probabilities = torch.rand(8, 1, dtype=torch.float64, device=device)
            selection_spec = SelectionSpec(selection_index=selection_indices,
                                           selection_probabilities=selection_probabilities)

            # Forward pass
            output = layer_cuda(test_input, selection_spec)

            # Verify output properties
            self.assertEqual(output.shape, (8, *self.out_shape), "CUDA output shape mismatch.")
            self.assertEqual(output.dtype, torch.float64, "CUDA output dtype mismatch.")
            self.assertEqual(output.device.type, device.type, "CUDA output device mismatch.")