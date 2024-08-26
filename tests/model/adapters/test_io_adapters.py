import unittest
import torch
from torch import nn
from unittest.mock import MagicMock, patch
from typing import Dict, Type, Any
from src.model.adapters.io_adapters import IOAdapter, IORegistry, RMSImageIOAdapter, VocabularyIOAdapter, registry, \
    LogitSeparator, ControllerIOAdapter


# Mock classes to use in the tests

class MockAdapter(IOAdapter):
    """
    This is a mock adapter for testing.
    """

    def __init__(self, embedding_dim: int, vocabulary_size: int):
        super(MockAdapter, self).__init__()
        self.embedding_dim = embedding_dim
        self.vocabulary_size = vocabulary_size

    @classmethod
    def setup(cls, embedding_dim: int, vocabulary_size: int) -> 'MockAdapter':
        return cls(embedding_dim, vocabulary_size)

    def embed_input(self, inputdata: torch.Tensor) -> torch.Tensor:
        pass

    def create_distribution(self, embeddings: torch.Tensor) -> torch.Tensor:
        pass


class AnotherMockAdapter(IOAdapter):
    """
    Another mock adapter for testing.
    """

    def __init__(self, embedding_dim: int, vocabulary_size: int):
        super(AnotherMockAdapter, self).__init__()
        self.embedding_dim = embedding_dim
        self.vocabulary_size = vocabulary_size

    @classmethod
    def setup(cls, embedding_dim: int, vocabulary_size: int) -> 'AnotherMockAdapter':
        return cls(embedding_dim, vocabulary_size)

    def embed_input(self, inputdata: torch.Tensor) -> torch.Tensor:
        pass

    def create_distribution(self, embeddings: torch.Tensor) -> torch.Tensor:
        pass


class TestIORegistry(unittest.TestCase):

    def setUp(self):
        self.registry = IORegistry()

    def test_register_io_adapter(self):
        config_spec = {"embedding_dim": int, "vocabulary_size": int}

        # Register MockAdapter
        self.registry.register("mock_adapter", config_spec, MockAdapter)

        # Check that the adapter can be set up correctly
        adapter_instance = self.registry.setup("mock_adapter", {"embedding_dim": 128, "vocabulary_size": 10000})
        self.assertIsInstance(adapter_instance, MockAdapter)
        self.assertEqual(adapter_instance.embedding_dim, 128)
        self.assertEqual(adapter_instance.vocabulary_size, 10000)

        # Registering another adapter with the same name should warn
        with self.assertWarns(UserWarning):
            self.registry.register("mock_adapter", config_spec, AnotherMockAdapter)

    def test_register_decorator(self):
        config_spec = {"embedding_dim": int, "vocabulary_size": int}

        # Use the decorator to register the adapter
        decorator = self.registry.register_decorator("decorator_test_adapter", config_spec)
        decorator(MockAdapter)

        # Check that the adapter was registered and can be set up
        adapter_instance = self.registry.setup("decorator_test_adapter", {"embedding_dim": 64, "vocabulary_size": 5000})
        self.assertIsInstance(adapter_instance, MockAdapter)
        self.assertEqual(adapter_instance.embedding_dim, 64)
        self.assertEqual(adapter_instance.vocabulary_size, 5000)

    def test_setup_with_missing_config(self):
        config_spec = {"embedding_dim": int, "vocabulary_size": int}
        self.registry.register("mock_adapter", config_spec, MockAdapter)

        # Missing 'vocabulary_size' in config should raise ValueError
        with self.assertRaises(ValueError) as context:
            self.registry.setup("mock_adapter", {"embedding_dim": 128})
        self.assertEqual(str(context.exception), "Missing required config element 'vocabulary_size'")

    def test_setup_with_wrong_type(self):
        config_spec = {"embedding_dim": int, "vocabulary_size": int}
        self.registry.register("mock_adapter", config_spec, MockAdapter)

        # Incorrect type for 'embedding_dim' should raise ValueError
        with self.assertRaises(ValueError) as context:
            self.registry.setup("mock_adapter", {"embedding_dim": "128", "vocabulary_size": 10000})
        self.assertEqual(str(context.exception), "Config element of name 'embedding_dim' was wrong type")

    def test_get_structure(self):
        config_spec = {"embedding_dim": int, "vocabulary_size": int}
        self.registry.register("mock_adapter", config_spec, MockAdapter)

        # Retrieve and validate the config structure
        structure = self.registry.get_structure("mock_adapter")
        self.assertEqual(structure, config_spec)

    def test_get_structure_invalid_name(self):
        # Attempt to retrieve a non-existent adapter's structure should raise ValueError
        with self.assertRaises(ValueError) as context:
            self.registry.get_structure("non_existent_adapter")
        self.assertEqual(str(context.exception),
                         "No IO adapter of name 'non_existent_adapter' exists, cannot retrieve config structure")

    def test_get_documentation(self):
        config_spec = {"embedding_dim": int, "vocabulary_size": int}
        self.registry.register("mock_adapter", config_spec, MockAdapter)

        # Retrieve and check the documentation for the registered adapter
        class_docstring, setup_docstring = self.registry.get_documentation("mock_adapter")
        self.assertIn("This is a mock adapter for testing.", class_docstring)

    def test_get_documentation_invalid_name(self):
        # Attempt to retrieve documentation for a non-existent adapter should raise ValueError
        with self.assertRaises(ValueError) as context:
            self.registry.get_documentation("non_existent_adapter")
        self.assertEqual(str(context.exception),
                         "No IO adapter of name 'non_existent_adapter' exists, cannot retrieve adapter documentation")

# Unit and Integration Tests

class TestVocabularyIOAdapter(unittest.TestCase):

    def setUp(self):
        # Mocking the registry for testing
        self.registry = IORegistry()

    def test_setup(self):
        # Test the setup function
        embedding_dim = 128
        vocabulary_size = 10000

        adapter_instance = VocabularyIOAdapter.setup(embedding_dim, vocabulary_size)

        # Ensure the adapter instance is properly initialized
        self.assertIsInstance(adapter_instance, VocabularyIOAdapter)
        self.assertIsInstance(adapter_instance.embeddings, nn.Embedding)
        self.assertIsInstance(adapter_instance.logits, nn.Linear)
        self.assertEqual(adapter_instance.embeddings.embedding_dim, embedding_dim)
        self.assertEqual(adapter_instance.embeddings.num_embeddings, vocabulary_size)
        self.assertEqual(adapter_instance.logits.in_features, embedding_dim)
        self.assertEqual(adapter_instance.logits.out_features, vocabulary_size)

    def test_embed_input(self):
        # Test the embed_input function
        embedding_dim = 128
        vocabulary_size = 10000

        adapter_instance = VocabularyIOAdapter.setup(embedding_dim, vocabulary_size)

        # Create a mock input tensor
        input_tensor = torch.randint(0, vocabulary_size, (2, 10))

        # Call embed_input
        embeddings = adapter_instance.embed_input(input_tensor)

        # Check if embeddings are correct
        self.assertEqual(embeddings.shape, (2, 10, embedding_dim))

    def test_create_distribution(self):
        # Test the create_distribution function
        embedding_dim = 128
        vocabulary_size = 10000

        adapter_instance = VocabularyIOAdapter.setup(embedding_dim, vocabulary_size)

        # Create a mock embeddings tensor
        embeddings = torch.randn(2, 10, embedding_dim)

        # Call create_distribution
        logits = adapter_instance.create_distribution(embeddings)

        # Check if logits are correct
        self.assertEqual(logits.shape, (2, 10, vocabulary_size))

    def test_registry_integration(self):

        # Setup the adapter through the registry
        embedding_dim = 128
        vocabulary_size = 10000
        config = {"embedding_dim": embedding_dim, "vocabulary_size": vocabulary_size}
        adapter_instance = registry.setup("vocabulary_adapter", config)

        # Ensure the adapter is properly initialized and works as expected
        self.assertIsInstance(adapter_instance, VocabularyIOAdapter)

        # Test the embed_input function via the registry
        input_tensor = torch.randint(0, vocabulary_size, (2, 10))
        embeddings = adapter_instance.embed_input(input_tensor)
        self.assertEqual(embeddings.shape, (2, 10, embedding_dim))

        # Test the create_distribution function via the registry
        logits = adapter_instance.create_distribution(embeddings)
        self.assertEqual(logits.shape, (2, 10, vocabulary_size))

class TestRMSImageIOAdapter(unittest.TestCase):

    def setUp(self):
        # Setup parameters for the adapter
        self.input_channels = 3
        self.embedding_dim = 128
        self.normalize = True
        self.max_image_value = 255.0

    def test_setup(self):
        # Test the setup function
        adapter_instance = RMSImageIOAdapter.setup(
            input_channels=self.input_channels,
            embedding_dim=self.embedding_dim,
            normalize=self.normalize,
            max_image_value=self.max_image_value
        )

        # Ensure the adapter instance is properly initialized
        self.assertIsInstance(adapter_instance, RMSImageIOAdapter)
        self.assertIsInstance(adapter_instance.embedding_proj, nn.Linear)
        self.assertIsInstance(adapter_instance.distribution_proj, nn.Linear)
        self.assertEqual(adapter_instance.embedding_proj.in_features, self.input_channels)
        self.assertEqual(adapter_instance.embedding_proj.out_features, self.embedding_dim)
        self.assertEqual(adapter_instance.distribution_proj.in_features, self.embedding_dim)
        self.assertEqual(adapter_instance.distribution_proj.out_features, self.input_channels)
        self.assertEqual(adapter_instance.normalize, self.normalize)
        self.assertEqual(adapter_instance.max_image_value, self.max_image_value)

    def test_embed_input(self):
        # Test the embed_input function
        adapter_instance = RMSImageIOAdapter.setup(
            input_channels=self.input_channels,
            embedding_dim=self.embedding_dim,
            normalize=self.normalize,
            max_image_value=self.max_image_value
        )

        # Create a mock input tensor with values in range [0, max_image_value]
        input_tensor = torch.zeros(2, 64, 64, self.input_channels).float()

        # Call embed_input
        embeddings = adapter_instance.embed_input(input_tensor)

        # Check if embeddings are correct
        self.assertEqual(embeddings.shape, (2, 64, 64, self.embedding_dim))

        # If normalize is True, the input should be divided by max_image_value
        expected_tensor = input_tensor / self.max_image_value
        self.assertTrue(torch.allclose(adapter_instance.embedding_proj(expected_tensor), embeddings))

    def test_create_distribution(self):
        # Test the create_distribution function
        adapter_instance = RMSImageIOAdapter.setup(
            input_channels=self.input_channels,
            embedding_dim=self.embedding_dim,
            normalize=self.normalize,
            max_image_value=self.max_image_value
        )

        # Create a mock embeddings tensor
        embeddings = torch.randn(2, 64, 64, self.embedding_dim)

        # Call create_distribution
        output_tensor = adapter_instance.create_distribution(embeddings)

        # Check if the output tensor has the correct shape
        self.assertEqual(output_tensor.shape, (2, 64, 64, self.input_channels))

        # If normalize is True, the output should be scaled by max_image_value
        expected_output = adapter_instance.distribution_proj(embeddings) * self.max_image_value
        self.assertTrue(torch.allclose(expected_output, output_tensor))

    def test_registry_integration(self):

        # Setup the adapter through the registry
        config = {
            "input_channels": self.input_channels,
            "embedding_dim": self.embedding_dim,
            "normalize": self.normalize,
            "max_image_value": self.max_image_value
        }
        adapter_instance = registry.setup("rms_image_adapter", config)

        # Ensure the adapter is properly initialized and works as expected
        self.assertIsInstance(adapter_instance, RMSImageIOAdapter)

        # Test the embed_input function via the registry
        input_tensor = torch.zeros(2, 64, 64, self.input_channels).float()
        embeddings = adapter_instance.embed_input(input_tensor)
        self.assertEqual(embeddings.shape, (2, 64, 64, self.embedding_dim))

        # Test the create_distribution function via the registry
        output_tensor = adapter_instance.create_distribution(embeddings)
        self.assertEqual(output_tensor.shape, (2, 64, 64, self.input_channels))


class TestLogitSeparator(unittest.TestCase):

    def setUp(self):
        self.logit_separator = LogitSeparator()

    def test_compute_zone_edges_basic(self):
        schemas = torch.tensor([[2, 3, 1], [1, 1, 2]])
        expected_start = torch.tensor([[0, 2, 5], [0, 1, 2]])
        expected_end = torch.tensor([[2, 5, 6], [1, 2, 4]])
        start, end = self.logit_separator.compute_zone_edges(schemas)
        self.assertTrue(torch.equal(start, expected_start), f"Expected start {expected_start}, but got {start}")
        self.assertTrue(torch.equal(end, expected_end), f"Expected end {expected_end}, but got {end}")

    def test_create_separation_mask(self):
        schemas = torch.tensor([[2, 3, 1], [1, 1, 1]])
        logits = torch.randn(2, 6)  # Random logits tensor just to match the shape
        expected_mask = torch.tensor([
            [[True, True, False, False, False, False], [False, False, True, True, True, False], [False, False, False, False, False, True]],
            [[True, False, False, False, False, False], [False, True, False, False, False, False], [False, False, True, False, False, False]]
        ])
        separation_mask = self.logit_separator.create_separation_mask(schemas, logits)
        self.assertTrue(torch.equal(separation_mask, expected_mask), f"Expected {expected_mask}, but got {separation_mask}")

    def test_forward(self):
        schemas = torch.tensor([[2, 3, 1], [1, 1, 2]])
        logits = torch.tensor([
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        ])
        expected_output = torch.tensor([
            [[0.1, 0.2, 0.0, 0.0, 0.0, 0.0], [0.3, 0.4, 0.5, 0.0, 0.0, 0.0], [0.6, 0.0, 0.0, 0.0, 0.0, 0.0]],
            [[0.1, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3, 0.4, 0.0, 0.0, 0.0, 0.0]]
        ])
        output = self.logit_separator.forward(schemas, logits)
        self.assertTrue(torch.equal(output[0], expected_output), f"Expected {expected_output}, but got {output}")


class TestControllerIOAdapter(unittest.TestCase):

    def setUp(self):
        # Define the schema for the tests
        self.schemas = {
            "control": torch.tensor([2, 0]),  # Control schema with 3 modes
            "mode1": torch.tensor([5, 0]),  # Mode 1 schema
            "mode2": torch.tensor([4, 2]),  # Mode 2 schema
        }

        self.embedding_dim = 16
        self.logit_size = 12  # Logit size should be sufficient for the schemas
        self.adapter_instance = ControllerIOAdapter.setup(
            embedding_dim=self.embedding_dim,
            logit_size=self.logit_size,
            schemas=self.schemas
        )

    def test_setup(self):
        # Test the setup function
        self.assertIsInstance(self.adapter_instance, ControllerIOAdapter)
        self.assertIsInstance(self.adapter_instance.embedding_layer, nn.Embedding)
        self.assertIsInstance(self.adapter_instance.logit_projector, nn.Linear)
        self.assertIsInstance(self.adapter_instance.logit_separator, LogitSeparator)
        self.assertEqual(self.adapter_instance.schema_reference.shape, torch.Size([3, 2]))

    def test_embed_input(self):
        # Test the embed_input function
        control_tensor = torch.tensor([0, 1, 2])  # Indices for control modes

        embeddings = self.adapter_instance.embed_input(control_tensor)

        # Verify the embeddings shape
        self.assertEqual(embeddings.shape, (3, self.embedding_dim))

    def test_create_distribution(self):
        # Test the create_distribution function
        control_tensor = torch.tensor([0, 1, 2])  # Indices for control modes

        embeddings = self.adapter_instance.embed_input(control_tensor)

        logits, mask = self.adapter_instance.create_distribution(embeddings, control_tensor)

        # Verify the output shapes
        self.assertEqual(logits.shape, (3, 2, self.logit_size))
        self.assertEqual(mask.shape, (3, 2, self.logit_size))

    def test_invalid_schema(self):
        # Test that invalid schema raises appropriate errors
        invalid_schemas = {
            "control": torch.tensor([3]),  # Control schema with 3 modes
            "mode1": torch.tensor([4]),  # Mode 1 schema with a different length
        }

        with self.assertRaises(AssertionError):
            ControllerIOAdapter.setup(
                embedding_dim=self.embedding_dim,
                logit_size=self.logit_size,
                schemas=invalid_schemas
            )

    def test_invalid_logit_size(self):
        # Test that an invalid logit size raises an error
        invalid_logit_size = 2  # Too small for the schema requirements

        with self.assertRaises(ValueError):
            ControllerIOAdapter.setup(
                embedding_dim=self.embedding_dim,
                logit_size=invalid_logit_size,
                schemas=self.schemas
            )