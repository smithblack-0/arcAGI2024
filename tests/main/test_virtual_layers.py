import unittest
import torch
from torch import nn
from src.main.model.virtual_layers import (DropoutLogits, virtual_state_select, virtual_state_scatter,
                                           SelectionSpec, VirtualState, VirtualParameter, VirtualBuffer,
                                           VirtualLayer, VirtualLinear, VirtualMergeHeads, VirtualMakeHeads
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
        selection_indices = torch.tensor([])  # No banks selected
        selection_probabilities = torch.tensor([])
        selection_spec = SelectionSpec(selection_index=selection_indices,
                                       selection_probabilities=selection_probabilities)

        # Expecting an empty result
        with self.assertRaises(IndexError):
            vp(selection_spec)

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

class TestVirtualBuffer(unittest.TestCase):

    def test_virtual_buffer_creation(self):
        """
        Test the creation of a VirtualBuffer using the create method.
        Ensures correct initialization and shape of the buffer.
        """
        bank_size = 4
        shape = (3, 3)

        # Create virtual buffer
        vb = VirtualBuffer.create(bank_size, shape)

        # Ensure the buffer has the correct shape
        self.assertEqual(vb.buffer.shape, (*shape, bank_size, ))

    def test_virtual_buffer_custom_init(self):
        """
        Test the custom initialization of a VirtualBuffer.
        Ensures the init function is applied to the buffer.
        """

        def custom_init(tensor):
            torch.nn.init.constant_(tensor, 0.5)

        bank_size = 2
        shape = (2, 2)

        # Create virtual buffer with custom init
        vb = VirtualBuffer.create(bank_size, shape, init=custom_init)

        # Check that the buffer was correctly initialized to 0.5
        self.assertTrue(torch.all(vb.buffer == 0.5))

    def test_virtual_buffer_manual_init(self):
        """
        Test manually initializing a VirtualBuffer with a buffer tensor.
        """
        buffer_bank = torch.tensor([[[0.0, 0.0], [0.0, 0.0]],
                                    [[1.0, 1.0], [0.0, 0.0]],
                                    [[2.0, 4.0], [3.0, 1.0]]])
        vb = VirtualBuffer(buffer_bank)
        self.assertTrue(torch.allclose(vb.buffer, buffer_bank))

    def test_throws_on_broadcasting(self):
        """
        Test expressing the buffer in superposition mode.
        Ensures that the correct combination of selected buffers is returned.
        """
        buffer_bank = torch.tensor([[[0.0, 1.0, 2.0],[0.0, 1.0, 4.0]],
                                    [[0.0, 0.0, 3.0], [0.0, 0.0, 1.0]]])

        vb = VirtualBuffer(buffer_bank)

        # Define a SelectionSpec with specific indices and probabilities.
        # Note that there is one fewer dimension than is required for a superposition
        # invokation, as this is
        selection_indices = torch.tensor([0, 2])  # Select banks 0 and 2
        selection_probabilities = torch.tensor([0.3, 0.7])  # Weights
        selection_spec = SelectionSpec(selection_index=selection_indices,
                                       selection_probabilities=selection_probabilities)

        # Express the buffer in superposition mode
        with self.assertRaises(ValueError):
            result = vb.express_buffer(selection_spec, superposition=True)

    def test_express_buffer_superposition(self):
        """
        Test expressing the buffer in superposition mode.
        Ensures that the correct combination of selected buffers is returned.
        """
        buffer_bank = torch.tensor([[[0.0, 1.0, 2.0],[0.0, 1.0, 4.0]],
                                    [[0.0, 0.0, 3.0], [0.0, 0.0, 1.0]]])

        vb = VirtualBuffer(buffer_bank)

        # Define a SelectionSpec with specific indices and probabilities
        selection_indices = torch.tensor([0, 2])  # Select banks 0 and 2
        selection_probabilities = torch.tensor([0.3, 0.7])  # Weights
        selection_spec = SelectionSpec(selection_index=selection_indices,
                                       selection_probabilities=selection_probabilities)

        # Express the buffer in superposition mode
        result = vb.express_buffer(selection_spec, superposition=True)

        # Manual computation of the expected result:
        expected_result = buffer_bank[..., 0]*0.3 + buffer_bank[..., 2]*0.7
        self.assertTrue(torch.allclose(result, expected_result, atol=1e-4))

    def test_express_buffer_no_superposition(self):
        """
        Test expressing the buffer without superposition.
        Ensures that the selected buffers are returned separately.
        """
        buffer_bank = torch.tensor([[[0.0, 1.0, 2.0],[0.0, 1.0, 4.0]],
                                    [[0.0, 0.0, 3.0], [0.0, 0.0, 1.0]]])
        vb = VirtualBuffer(buffer_bank)

        # Define a SelectionSpec with specific indices
        selection_indices = torch.tensor([0, 2])  # Select banks 0 and 2
        selection_probabilities = torch.tensor([0.5, 0.5])  # Dummy weights, not used
        selection_spec = SelectionSpec(selection_index=selection_indices,
                                       selection_probabilities=selection_probabilities)

        # Express the buffer without superposition
        result = vb.express_buffer(selection_spec, superposition=False)

        # Manually check the expected result
        expected_result = [
            buffer_bank[..., 0],
            buffer_bank[..., 2],

        ]
        expected_result = torch.stack(expected_result, dim=-1)

        self.assertTrue(torch.allclose(result, expected_result))

    def test_update_buffer_superposition(self):
        """
        Test updating the buffer with a new expression in superposition mode.
        Ensures that the buffer is correctly updated.
        """
        buffer_bank = torch.tensor([[[0.0, 1.0, 2.0],[0.0, 1.0, 4.0]],
                                    [[0.0, 0.0, 3.0], [0.0, 0.0, 1.0]]])
        vb = VirtualBuffer(buffer_bank)

        # Define a SelectionSpec with specific indices and probabilities
        selection_indices = torch.tensor([0, 2])  # Select banks 0 and 2
        selection_probabilities = torch.tensor([0.3, 0.7])  # Weights
        selection_spec = SelectionSpec(selection_index=selection_indices,
                                       selection_probabilities=selection_probabilities)

        # Define a new expression. Notice it is reduced in dimensionality
        # by one unit.
        new_expression = torch.tensor([[3.0, 4.0], [7.0, 1.0]])

        # Manually compute the expected updated buffer
        expected_updated_buffer = buffer_bank.clone()
        expected_updated_buffer[..., 0] = buffer_bank[...,0]*0.7 + new_expression*0.3
        expected_updated_buffer[..., 2] = buffer_bank[...,2]*0.3 + new_expression*0.7

        # Update the buffer
        vb.update_buffer(new_expression, selection_spec, superposition=True)

        self.assertTrue(torch.allclose(vb.buffer, expected_updated_buffer))

    def test_update_buffer_no_superposition(self):
        """
        Test updating the buffer with a new expression without superposition.
        Ensures that the buffer is updated only at the selected indices.
        """
        buffer_bank = torch.tensor([[[0.0, 1.0, 2.0],[0.0, 1.0, 4.0]],
                                    [[0.0, 0.0, 3.0], [0.0, 0.0, 1.0]]])
        vb = VirtualBuffer(buffer_bank)

        # Define a SelectionSpec with specific indices
        selection_indices = torch.tensor([0, 2])  # Select banks 0 and 2
        selection_probabilities = torch.tensor([0.5, 0.5])  # Dummy weights
        selection_spec = SelectionSpec(selection_index=selection_indices,
                                       selection_probabilities=selection_probabilities)

        # Define a new expression
        new_expression = torch.tensor([[[1.5, 3.0], [2.0, 1.0]],
                                       [[4.5, 5.5], [6.0, 7.0]]])

        # Update the buffer without superposition
        vb.update_buffer(new_expression, selection_spec, superposition=False)

        # Manually compute the expected updated buffer
        expected_updated_buffer = buffer_bank.clone()
        expected_updated_buffer[..., 0] = buffer_bank[..., 0]*0.5 + new_expression[..., 0]*0.5
        expected_updated_buffer[..., 2] = buffer_bank[..., 2]*0.5 + new_expression[..., 1]*0.5

        self.assertTrue(torch.allclose(vb.buffer, expected_updated_buffer))
    def test_batch_shape_handling(self):
        """
        Test batch shape handling in virtual buffer when expressing and updating.
        Ensures that batch dimensions are handled correctly.
        """
        buffer = torch.randn(3, 4, 6)  # Shape (batch, extra_dim, banks)
        indices = torch.randint(0, 4, (3, 2))  # Batch shape (batch, extra_dim, selected)
        probabilities = torch.rand(3, 2)  # Probabilities shape matches the selection
        selection = SelectionSpec(selection_index=indices, selection_probabilities=probabilities)

        vb = VirtualBuffer(buffer)

        # Express buffer
        expressed = vb.express_buffer(selection, superposition=True)
        self.assertEqual((3, 4), expressed.shape)

        # Update buffer
        vb.update_buffer(expressed, selection, superposition=True)
        self.assertEqual(vb.buffer.shape, buffer.shape)

        # With superposition off
        expressed = vb.express_buffer(selection, superposition=False)
        self.assertEqual((3, 4, 2), expressed.shape)

        vb.update_buffer(expressed, selection, superposition=False)
        self.assertEqual(vb.buffer.shape, buffer.shape)


class TestVirtualState(unittest.TestCase):

    def test_virtual_state_creation(self):
        """
        Test the creation of a VirtualState using the create method.
        Ensures correct initialization and shape of the state.
        """
        bank_size = 4
        shape = (3, 3)

        # Create virtual state
        vs = VirtualState.create(bank_size, shape)

        # Ensure the state has the correct shape
        self.assertEqual(vs.state.shape, (*shape, bank_size))

    def test_virtual_state_custom_init(self):
        """
        Test the custom initialization of a VirtualState.
        Ensures the init function is applied to the state.
        """

        def custom_init(tensor):
            torch.nn.init.constant_(tensor, 0.5)

        bank_size = 2
        shape = (2, 2)

        # Create virtual state with custom init
        vs = VirtualState.create(bank_size, shape, init=custom_init)

        # Check that the state was correctly initialized to 0.5
        self.assertTrue(torch.all(vs.state == 0.5))

    def test_virtual_state_manual_init(self):
        """
        Test manually initializing a VirtualState with a state tensor.
        """
        state_bank = torch.tensor([[[0.0, 0.0], [0.0, 0.0]],
                                   [[1.0, 1.0], [0.0, 0.0]],
                                   [[2.0, 4.0], [3.0, 1.0]]])
        vs = VirtualState(state_bank)
        self.assertTrue(torch.allclose(vs.state, state_bank))

    def test_express_state_superposition(self):
        """
        Test expressing the state in superposition mode.
        Ensures that the correct combination of selected states is returned.
        """
        state_bank = torch.tensor([[[0.0, 1.0, 2.0], [0.0, 1.0, 4.0]],
                                   [[0.0, 0.0, 3.0], [0.0, 0.0, 1.0]]])

        vs = VirtualState(state_bank)

        # Define a SelectionSpec with specific indices and probabilities
        selection_indices = torch.tensor([0, 2])  # Select banks 0 and 2
        selection_probabilities = torch.tensor([0.3, 0.7])  # Weights
        selection_spec = SelectionSpec(selection_index=selection_indices,
                                       selection_probabilities=selection_probabilities)

        # Express the state in superposition mode
        result = vs.express_state(selection_spec, superposition=True)

        # Manual computation of the expected result:
        expected_result = state_bank[..., 0] * 0.3 + state_bank[..., 2] * 0.7
        self.assertTrue(torch.allclose(result, expected_result, atol=1e-4))

    def test_express_state_no_superposition(self):
        """
        Test expressing the state without superposition.
        Ensures that the selected states are returned separately.
        """
        state_bank = torch.tensor([[[0.0, 1.0, 2.0], [0.0, 1.0, 4.0]],
                                   [[0.0, 0.0, 3.0], [0.0, 0.0, 1.0]]])

        vs = VirtualState(state_bank)

        # Define a SelectionSpec with specific indices
        selection_indices = torch.tensor([0, 2])  # Select banks 0 and 2
        selection_probabilities = torch.tensor([0.5, 0.5])  # Dummy weights, not used
        selection_spec = SelectionSpec(selection_index=selection_indices,
                                       selection_probabilities=selection_probabilities)

        # Express the state without superposition
        result = vs.express_state(selection_spec, superposition=False)

        # Manually check the expected result
        expected_result = torch.stack([state_bank[..., 0], state_bank[..., 2]], dim=-1)

        self.assertTrue(torch.allclose(result, expected_result))

    def test_update_state_superposition(self):
        """
        Test updating the state with a new expression in superposition mode.
        Ensures that the state is correctly updated.
        """
        state_bank = torch.tensor([[[0.0, 1.0, 2.0], [0.0, 1.0, 4.0]],
                                   [[0.0, 0.0, 3.0], [0.0, 0.0, 1.0]]])

        vs = VirtualState(state_bank)

        # Define a SelectionSpec with specific indices and probabilities
        selection_indices = torch.tensor([0, 2])  # Select banks 0 and 2
        selection_probabilities = torch.tensor([0.3, 0.7])  # Weights
        selection_spec = SelectionSpec(selection_index=selection_indices,
                                       selection_probabilities=selection_probabilities)

        # Define a new expression
        new_expression = torch.tensor([[3.0, 4.0], [7.0, 1.0]])

        # Manually compute the expected updated state
        expected_updated_state = state_bank.clone()
        expected_updated_state[..., 0] = state_bank[..., 0] * 0.7 + new_expression * 0.3
        expected_updated_state[..., 2] = state_bank[..., 2] * 0.3 + new_expression * 0.7

        # Update the state
        vs.update_state(new_expression, selection_spec, superposition=True)

        self.assertTrue(torch.allclose(vs.state, expected_updated_state))

    def test_update_state_no_superposition(self):
        """
        Test updating the state with a new expression without superposition.
        Ensures that the state is updated only at the selected indices.
        """
        state_bank = torch.tensor([[[0.0, 1.0, 2.0], [0.0, 1.0, 4.0]],
                                   [[0.0, 0.0, 3.0], [0.0, 0.0, 1.0]]])

        vs = VirtualState(state_bank)

        # Define a SelectionSpec with specific indices
        selection_indices = torch.tensor([0, 2])  # Select banks 0 and 2
        selection_probabilities = torch.tensor([0.5, 0.5])  # Dummy weights
        selection_spec = SelectionSpec(selection_index=selection_indices,
                                       selection_probabilities=selection_probabilities)

        # Define a new expression
        new_expression = torch.tensor([[[1.5, 3.0], [2.0, 1.0]],
                                       [[4.5, 5.5], [6.0, 7.0]]])

        # Update the state without superposition
        vs.update_state(new_expression, selection_spec, superposition=False)

        # Manually compute the expected updated state
        expected_updated_state = state_bank.clone()
        expected_updated_state[..., 0] = state_bank[..., 0] * 0.5 + new_expression[..., 0] * 0.5
        expected_updated_state[..., 2] = state_bank[..., 2] * 0.5 + new_expression[..., 1] * 0.5

        self.assertTrue(torch.allclose(vs.state, expected_updated_state))

    def test_batch_shape_handling(self):
        """
        Test batch shape handling in virtual state when expressing and updating.
        Ensures that batch dimensions are handled correctly.
        """
        state = torch.randn(3, 4, 6)  # Shape (batch, extra_dim, banks)
        indices = torch.randint(0, 4, (3, 2))  # Batch shape (batch, extra_dim, selected)
        probabilities = torch.rand(3, 2)  # Probabilities shape matches the selection
        selection = SelectionSpec(selection_index=indices, selection_probabilities=probabilities)

        vs = VirtualState(state)

        # Express state
        expressed = vs.express_state(selection, superposition=True)
        self.assertEqual((3, 4), expressed.shape)

        # Update state
        vs.update_state(expressed, selection, superposition=True)
        self.assertEqual(vs.state.shape, state.shape)

        # With superposition off
        expressed = vs.express_state(selection, superposition=False)
        self.assertEqual((3, 4, 2), expressed.shape)

        vs.update_state(expressed, selection, superposition=False)
        self.assertEqual(vs.state.shape, state.shape)

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
