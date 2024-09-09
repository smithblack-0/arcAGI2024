import unittest
import torch
from src.old.model.schema import SchemaRegistry


class TestSchemaTracker(unittest.TestCase):

    def create_fresh_schema_tracker(self):
        """
        Helper method to create a fresh, uncontaminated SchemaTracker instance
        for each test case.

        Returns:
        --------
        SchemaTracker : A new instance of SchemaTracker with predefined dimensions and logit slots.
        """
        return SchemaRegistry(num_dimensions=3, logit_slots=10)

    def test_register_schema_success(self):
        # Create a fresh SchemaTracker instance
        schema_tracker = self.create_fresh_schema_tracker()

        # Register a valid schema
        schema_tracker.register_schema("text", [3, 2, 4])
        schema_id = schema_tracker.get_schema_id("text")

        # Verify that the schema ID is correct
        self.assertEqual(schema_id, 0)

        # Fetch the schema and check its correctness
        schemas = schema_tracker.fetch_schemas(torch.tensor([schema_id]))
        expected_schema = torch.tensor([[3, 2, 4]], dtype=torch.int64)
        self.assertTrue(torch.equal(schemas, expected_schema))

    def test_register_schema_padding(self):
        # Create a fresh SchemaTracker instance
        schema_tracker = self.create_fresh_schema_tracker()

        # Register a schema that needs padding
        schema_tracker.register_schema("image", [3])
        schema_id = schema_tracker.get_schema_id("image")

        # Verify that the schema ID is correct
        self.assertEqual(schema_id, 0)

        # Fetch the schema and check if padding was correctly applied
        schemas = schema_tracker.fetch_schemas(torch.tensor([schema_id]))
        expected_schema = torch.tensor([[3, 0, 0]], dtype=torch.int64)
        self.assertTrue(torch.equal(schemas, expected_schema))

    def test_register_schema_too_long(self):
        # Create a fresh SchemaTracker instance
        schema_tracker = self.create_fresh_schema_tracker()

        # Register a schema that is too long
        with self.assertRaises(ValueError):
            schema_tracker.register_schema("invalid_schema", [1, 2, 3, 4])

    def test_register_schema_sum_exceeds_logit_slots(self):
        # Create a fresh SchemaTracker instance
        schema_tracker = self.create_fresh_schema_tracker()

        # Register a schema that exceeds the logit slots
        with self.assertRaises(ValueError):
            schema_tracker.register_schema("invalid_schema", [5, 4, 3])

    def test_register_schema_duplicate_name(self):
        # Create a fresh SchemaTracker instance
        schema_tracker = self.create_fresh_schema_tracker()

        # Register a schema with a duplicate name
        schema_tracker.register_schema("text", [3, 2, 4])
        with self.assertRaises(ValueError):
            schema_tracker.register_schema("text", [1, 1, 1])

    def test_get_schema_id_invalid_name(self):
        # Create a fresh SchemaTracker instance
        schema_tracker = self.create_fresh_schema_tracker()

        # Attempt to get the ID of an unregistered schema name
        with self.assertRaises(KeyError):
            schema_tracker.get_schema_id("non_existent")

    def test_fetch_schemas_invalid_tensor_type(self):
        # Create a fresh SchemaTracker instance
        schema_tracker = self.create_fresh_schema_tracker()

        # Fetch schemas using an invalid tensor type
        with self.assertRaises(TypeError):
            schema_tracker.fetch_schemas(torch.tensor([0], dtype=torch.float32))

    def test_multiple_schemas_registration(self):
        # Create a fresh SchemaTracker instance
        schema_tracker = self.create_fresh_schema_tracker()

        # Register multiple schemas and verify their IDs and retrieval
        schema_tracker.register_schema("text", [3, 2, 4])
        schema_tracker.register_schema("image", [5, 2])
        schema_tracker.register_schema("audio", [1, 1, 1])

        text_id = schema_tracker.get_schema_id("text")
        image_id = schema_tracker.get_schema_id("image")
        audio_id = schema_tracker.get_schema_id("audio")

        # Fetch and verify all schemas
        schemas = schema_tracker.fetch_schemas(torch.tensor([text_id, image_id, audio_id]))
        expected_schemas = torch.tensor([[3, 2, 4], [5, 2, 0], [1, 1, 1]], dtype=torch.int64)
        self.assertTrue(torch.equal(schemas, expected_schemas))

