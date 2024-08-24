import unittest
import torch
from src.model.core_transformer import SchemaRegistry, LogitSeparator


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
        self.assertTrue(torch.equal(output, expected_output), f"Expected {expected_output}, but got {output}")
