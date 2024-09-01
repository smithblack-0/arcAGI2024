import unittest
import torch
from unittest.mock import MagicMock, patch
from typing import Any, Dict, Type
from src.model.adapters.io_adapters import IORegistry
from src.model.adapters.distribution_adapters import (DistributionAdapterRegistry, DistributionAdapter,
                                                      VocabularyDistributionAdapter, registry)

class MockDistributionAdapter:
    """
    This is a mock distribution adapter for testing.
    """

    @classmethod
    def setup(cls, use_hard_sample: bool, use_embedding_bags: bool, top_k: int, num_resamples: int):
        """
        Setup method for the mock distribution adapter.
        """

class TestDistributionAdapterRegistry(unittest.TestCase):

    def create_clean_registry(self) -> DistributionAdapterRegistry:
        # Create the registry
        io_registry = IORegistry()
        distribution_adapters_registry = DistributionAdapterRegistry(io_registry)

        # Mock up some IO registry cases
        mock_io_adapter = MagicMock()
        another_mock_io_adapter = MagicMock()

        # Use patch as a context manager
        with patch('builtins.issubclass', return_value=True):
            io_registry.register("mock_io_adapter", {}, mock_io_adapter)
            io_registry.register("another_mock_io_adapter", {}, another_mock_io_adapter)

        return distribution_adapters_registry

    def test_register_distribution_adapter(self):
        distribution_registry = self.create_clean_registry()
        config_spec = {"use_hard_sample": bool, "use_embedding_bags": bool, "top_k": int, "num_resamples": int}

        # Register a mock distribution adapter
        mock_distribution_adapter = MagicMock()
        with patch('builtins.issubclass', return_value=True):
            distribution_registry.register("mock_distribution_adapter", "mock_io_adapter", config_spec,
                                           mock_distribution_adapter)

        # Setup the adapter and capture the call to the mock's setup method
        adapter_instance = distribution_registry.setup("mock_distribution_adapter", {
            "use_hard_sample": True,
            "use_embedding_bags": False,
            "top_k": 10,
            "num_resamples": 5
        })

        self.assertIsInstance(adapter_instance, MagicMock)

        # Verify that the setup method was called with the correct parameters
        mock_distribution_adapter.setup.assert_called_once_with(
            use_hard_sample=True,
            use_embedding_bags=False,
            top_k=10,
            num_resamples=5
        )

        # Registering another adapter with the same name should warn
        another_mock_distribution_adapter = MagicMock()
        with patch('builtins.issubclass', return_value=True):
            with self.assertWarns(UserWarning):
                distribution_registry.register("mock_distribution_adapter", "mock_io_adapter", config_spec,
                                               another_mock_distribution_adapter)

    def test_register_association(self):
        distribution_registry = self.create_clean_registry()
        config_spec = {"use_hard_sample": bool, "use_embedding_bags": bool, "top_k": int, "num_resamples": int}

        # Register MockDistributionAdapter
        mock_distribution_adapter = MagicMock()
        with patch('builtins.issubclass', return_value=True):
            distribution_registry.register("mock_distribution_adapter", "mock_io_adapter", config_spec, mock_distribution_adapter)

        # Register a new association with another mock IO adapter
        with patch('builtins.issubclass', return_value=True):
            distribution_registry.register_association("mock_distribution_adapter", "another_mock_io_adapter")

        # Ensure the new association is correctly recorded
        associations = distribution_registry.get_associations("mock_distribution_adapter")
        self.assertIn("another_mock_io_adapter", associations)
        self.assertIn("mock_io_adapter", associations)

    def test_setup_with_missing_config(self):
        distribution_registry = self.create_clean_registry()
        config_spec = {"use_hard_sample": bool, "use_embedding_bags": bool, "top_k": int, "num_resamples": int}
        mock_distribution_adapter = MagicMock()
        with patch('builtins.issubclass', return_value=True):
            distribution_registry.register("mock_distribution_adapter", "mock_io_adapter", config_spec, mock_distribution_adapter)

        # Missing 'num_resamples' in config should raise ValueError
        with self.assertRaises(TypeError) as context:
            distribution_registry.setup("mock_distribution_adapter", {
                "use_hard_sample": True,
                "use_embedding_bags": False,
                "top_k": 10
            })

    def test_setup_with_wrong_type(self):
        distribution_registry = self.create_clean_registry()
        config_spec = {"use_hard_sample": bool, "use_embedding_bags": bool, "top_k": int, "num_resamples": int}
        mock_distribution_adapter = MagicMock()
        with patch('builtins.issubclass', return_value=True):
            distribution_registry.register("mock_distribution_adapter", "mock_io_adapter", config_spec, mock_distribution_adapter)

        # Incorrect type for 'top_k' should raise ValueError
        with self.assertRaises(TypeError) as context:
            distribution_registry.setup("mock_distribution_adapter", {
                "use_hard_sample": True,
                "use_embedding_bags": False,
                "top_k": "10",  # Incorrect type
                "num_resamples": 5
            })

    def test_get_structure(self):
        distribution_registry = self.create_clean_registry()
        config_spec = {"use_hard_sample": bool, "use_embedding_bags": bool, "top_k": int, "num_resamples": int}
        mock_distribution_adapter = MagicMock()
        with patch('builtins.issubclass', return_value=True):
            distribution_registry.register("mock_distribution_adapter", "mock_io_adapter", config_spec, mock_distribution_adapter)

        # Retrieve and validate the config structure
        structure = distribution_registry.get_config_spec("mock_distribution_adapter")
        self.assertEqual(structure, config_spec)

    def test_get_structure_invalid_name(self):
        distribution_registry = self.create_clean_registry()
        # Attempt to retrieve a non-existent adapter's structure should raise ValueError
        with self.assertRaises(ValueError) as context:
            distribution_registry.get_config_spec("non_existent_adapter")

    def test_get_documentation(self):
        distribution_registry = self.create_clean_registry()
        mock_distribution_adapter = MockDistributionAdapter
        with patch('builtins.issubclass', return_value=True):
            distribution_registry.register("mock_distribution_adapter",
                                           "mock_io_adapter",
                                           {},
                                           mock_distribution_adapter)


        # Retrieve and check the documentation for the registered adapter
        class_docstring, setup_docstring = distribution_registry.get_documentation("mock_distribution_adapter")
        self.assertIsNotNone(class_docstring)
        self.assertIsNotNone(setup_docstring)

    def test_get_documentation_invalid_name(self):
        distribution_registry = self.create_clean_registry()
        # Attempt to retrieve documentation for a non-existent adapter should raise ValueError
        with self.assertRaises(ValueError) as context:
            distribution_registry.get_documentation("non_existent_adapter")

class TestVocabularyDistributionAdapter(unittest.TestCase):

    def instantiate_adapter(self):
        """
        Helper method to create a new instance of VocabularyDistributionAdapter with hardcoded configurations.
        """
        return VocabularyDistributionAdapter(
            smoothing_rates=torch.tensor([0.0, 0.1, 0.2])
        )

    def test_initialization(self):
        # Test if the adapter is initialized correctly
        adapter = self.instantiate_adapter()
        self.assertTrue(torch.equal(adapter.smoothing_rates, torch.tensor([0.0, 0.1, 0.2])))

    def test_sample(self):
        # Test the sample function
        adapter = self.instantiate_adapter()
        distribution = torch.rand(3, 10)
        mask = torch.ones(3, 10, dtype=torch.bool)  # No masking
        sampled_indices = adapter.sample(distribution, mask=mask, temperature=1.0)

        # Check that the sampled indices have the correct shape
        self.assertEqual(sampled_indices.shape, torch.Size([3]))
        self.assertTrue((sampled_indices >= 0).all() and (sampled_indices < 10).all())

    def test_sample_with_masking(self):
        # Test the sample function with masking
        adapter = self.instantiate_adapter()
        distribution = torch.rand(3, 10)

        # Create a mask that allows only one valid option per row
        mask = torch.zeros(3, 10, dtype=torch.bool)
        mask[0, 3] = True  # Only allow sampling from index 3 in the first row
        mask[1, 7] = True  # Only allow sampling from index 7 in the second row
        mask[2, 2] = True  # Only allow sampling from index 2 in the third row

        sampled_indices = adapter.sample(distribution, mask=mask, temperature=1.0)

        # Check that the sampled indices match the only allowed indices
        self.assertEqual(sampled_indices[0], 3)
        self.assertEqual(sampled_indices[1], 7)
        self.assertEqual(sampled_indices[2], 2)

    def test_sample_temperature_assertion(self):
        # Test that sample raises an assertion for non-positive temperature
        adapter = self.instantiate_adapter()
        distribution = torch.rand(3, 10)
        mask = torch.ones(3, 10, dtype=torch.bool)

        with self.assertRaises(AssertionError):
            adapter.sample(distribution, mask=mask, temperature=-0.1)

    def test_loss(self):
        # Test the loss function
        adapter = self.instantiate_adapter()
        distribution = torch.randn(2, 5, 10)  # Example logit distribution
        targets = torch.randint(0, 10, (2, 5)).long()  # Example targets
        mask = torch.ones_like(targets, dtype=torch.bool)  # No masking for simplicity
        smoothing_association = torch.randint(0, 3, (2, 5))  # Random smoothing associations

        # Compute loss
        loss = adapter.loss(distribution, targets, mask, smoothing_association=smoothing_association)

        # Check that loss has the correct shape
        self.assertEqual(loss.shape, torch.Size([2]))

    def test_loss_shape_assertions(self):
        # Test the loss function with incorrect shapes to ensure assertions work
        adapter = self.instantiate_adapter()
        distribution = torch.randn(2, 5, 10)
        targets = torch.randint(0, 10, (2, 4)).long()  # Incorrect target shape
        mask = torch.ones_like(targets, dtype=torch.bool)
        smoothing_association = torch.randint(0, 3, (2, 5))

        with self.assertRaises(AssertionError):
            adapter.loss(distribution, targets, mask, smoothing_association=smoothing_association)

    def test_loss_with_masking(self):
        # Test the loss function with masking
        adapter = self.instantiate_adapter()
        distribution = torch.randn(2, 5, 10)
        targets = torch.randint(0, 10, (2, 5)).long()
        mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]], dtype=torch.bool)
        smoothing_association = torch.randint(0, 3, (2, 5))

        loss = adapter.loss(distribution, targets, mask, smoothing_association=smoothing_association)

        # Check the loss shape
        self.assertEqual(loss.shape, torch.Size([2]))

    #TODO: Finish up manual loss check.
    @unittest.skip("Finish later")
    def test_manual_loss(self):
        # Test a manually computed loss.
        #
        # There will be three entries to compute loss on, with three catagories for each
        # The first catagory will target a class of 1, with a 0.0 label smoothing rate.
        # 2, class 2, 0.1 label smoothing rate. 3. class 3, 0.2 label smoothing rate.
        # The third case, however, will be masked.
        # If everything goes right, the following SHOULD have a loss of zero.

        target_classes = torch.tensor([[0, 1, 2]])
        target_smoothing_selections = torch.tensor([[0, 1, 2]])
        target_probs = torch.tensor([[1.0, 0.0, 0.0], [0.9, 0.05, 0.05], [0.0, 0.8, 0.2]])
        solving_mask = torch.tensor([[True, True, False]])

        logits = torch.log(target_classes) + torch.logsumexp(target_probs, dim=-1)

        # Instance and test
        instance = self.instantiate_adapter()
        loss = instance.loss(logits, target_classes, solving_mask, target_smoothing_selections)
        print(loss)

    def test_vocabulary_distribution_adapter_setup_via_registry(self):
        # Define the configuration for setting up the adapter
        setup_config = {
            "label_smoothing_rates": [0.0, 0.1, 0.2]
        }

        # Instantiate the adapter using the registry's setup method
        adapter_instance = registry.setup("vocab_distribution", setup_config)

        # Check that the adapter was instantiated correctly
        self.assertIsInstance(adapter_instance, VocabularyDistributionAdapter)
        self.assertEqual(adapter_instance.smoothing_rates, [0.0, 0.1, 0.2])