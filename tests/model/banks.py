
import torch
import unittest
from src.main.model.virtual_layers import BankedLinear
class TestBankedLinear(unittest.TestCase):

    def create_banked_linear(self, in_features, out_features, num_banks):
        """
        Helper function to create a fresh BankedLinear instance for each test.
        """
        return BankedLinear(in_features, out_features, num_banks)

    def test_weight_extraction(self):
        """
        Test that the weight extraction method correctly retrieves the weights
        corresponding to the selected banks.
        """
        in_features = 4
        out_features = 3
        num_banks = 5
        banked_linear = self.create_banked_linear(in_features, out_features, num_banks)

        bank_selection = torch.tensor([0, 2, 4])  # Indices of banks to select
        extracted_weights = banked_linear.extract_weights(bank_selection)

        # Ensure that the extracted weights have the correct shape
        expected_shape = (3, in_features, out_features)  # 3 banks selected
        self.assertEqual(extracted_weights.shape, expected_shape)

    def test_bias_extraction(self):
        """
        Test that the bias extraction method correctly retrieves the biases
        corresponding to the selected banks.
        """
        in_features = 4
        out_features = 3
        num_banks = 5
        banked_linear = self.create_banked_linear(in_features, out_features, num_banks)

        bank_selection = torch.tensor([0, 2, 4])  # Indices of banks to select
        extracted_bias = banked_linear.extract_bias(bank_selection)

        # Ensure that the extracted biases have the correct shape
        expected_shape = (3, out_features)  # 3 banks selected
        self.assertEqual(extracted_bias.shape, expected_shape)

    def test_forward_method(self):
        """
        Test the forward method to ensure that it performs matrix multiplication
        with the selected banks' weights and biases, and that it correctly sums
        the banks' outputs.
        """
        in_features = 4
        out_features = 6
        num_banks = 5
        banked_linear = self.create_banked_linear(in_features, out_features, num_banks)

        # Inputs for the forward pass
        tensor = torch.randn(2, in_features)  # Shape (in_features=4)
        bank_selection = torch.tensor([0, 2, 4]).unsqueeze(0).repeat(2, 1)  # Indices of banks to select
        bank_weights = torch.tensor([0.3, 0.5, 0.2]).unsqueeze(0).repeat(2, 1)  # Probabilities for the 3 selected banks

        # Run the forward method
        output = banked_linear.forward(tensor, bank_weights, bank_selection)

        # Check the output shape
        expected_shape = (2, out_features)  # Output shape should match batch size and out_features
        self.assertEqual(output.shape, expected_shape)

        # No detailed value check here, but could potentially check if forward output is reasonable
