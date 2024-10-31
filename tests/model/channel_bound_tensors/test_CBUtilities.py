import unittest
import torch
from src.old.CBTensors import CBTensor
from src.old.CBTensors import CBTensorSpec
from src.old.CBTensors import CBIndirectionLookup, CBReplaceOnMatch


class TestCBIndirectionLookup(unittest.TestCase):

    def setUp(self):
        """
        Set up the input and output specs, and initialize the necessary parameters.
        """
        self.input_spec = CBTensorSpec({'state': 1, 'mode': 1})
        self.output_spec = CBTensorSpec({'vocabulary_size': 1})
        self.device = torch.device('cpu')

    def create_fresh_lookup(self):
        """
        This function creates and returns a new CBIndirectionLookup instance
        to ensure no cross-test contamination.
        """
        return CBIndirectionLookup(self.input_spec, self.output_spec, self.device)

    def test_register_and_lookup_single_pattern(self):
        """
        Test registering a single pattern and checking if it returns the expected output.
        """
        lookup = self.create_fresh_lookup()

        input_pattern = {'state': 0, 'mode': 0}
        output_pattern = {'vocabulary_size': 10}

        # Register the pattern
        lookup.register(input_pattern, output_pattern)

        # Test lookup
        input_tensor = CBTensor.create_from_channels({'state': torch.tensor([0]), 'mode': torch.tensor([0])})
        output = lookup(input_tensor)

        # Check the output
        expected_output = CBTensor.create_from_channels({'vocabulary_size': torch.tensor([10])})
        self.assertTrue(torch.equal(output.get_tensor(), expected_output.get_tensor()))

    def test_register_and_lookup_multiple_patterns(self):
        """
        Test registering multiple patterns and ensuring the correct output is returned for each.
        """
        lookup = self.create_fresh_lookup()

        # Register multiple patterns
        lookup.register({'state': 0, 'mode': 0}, {'vocabulary_size': 10})
        lookup.register({'state': 1, 'mode': 0}, {'vocabulary_size': 100})
        lookup.register({'state': 1, 'mode': 1}, {'vocabulary_size': 40})

        # Check first pattern
        input_tensor = CBTensor.create_from_channels({'state': torch.tensor([0]), 'mode': torch.tensor([0])})
        output = lookup(input_tensor)
        expected_output = CBTensor.create_from_channels({'vocabulary_size': torch.tensor([10])})
        self.assertTrue(torch.equal(output.get_tensor(), expected_output.get_tensor()))

        # Check second pattern
        input_tensor = CBTensor.create_from_channels({'state': torch.tensor([1]), 'mode': torch.tensor([0])})
        output = lookup(input_tensor)
        expected_output = CBTensor.create_from_channels({'vocabulary_size': torch.tensor([100])})
        self.assertTrue(torch.equal(output.get_tensor(), expected_output.get_tensor()))

        # Check third pattern
        input_tensor = CBTensor.create_from_channels({'state': torch.tensor([1]), 'mode': torch.tensor([1])})
        output = lookup(input_tensor)
        expected_output = CBTensor.create_from_channels({'vocabulary_size': torch.tensor([40])})
        self.assertTrue(torch.equal(output.get_tensor(), expected_output.get_tensor()))

    def test_many_dimensional_lookup(self):
        """
        Test registering patterns and performing lookups in a batch scenario.
        """
        lookup = self.create_fresh_lookup()

        # Register multiple patterns
        lookup.register({'state': 0, 'mode': 0}, {'vocabulary_size': 10})
        lookup.register({'state': 1, 'mode': 1}, {'vocabulary_size': 50})
        lookup.register({"state" : 1, "mode" : 0}, {"vocabulary_size": 20})
        lookup.register({"state" : 0, "mode" : 1}, {"vocabulary_size": 10})

        # Create an input with lots of dimensions to test

        input_tensor = torch.randint(0, 1, [20, 10, 30, 2])
        input_tensor = CBTensor(self.input_spec, input_tensor)

        # see if we run
        lookup = lookup(input_tensor)
    def test_lookup_no_match(self):
        """
        Test attempting to lookup a pattern that hasn't been registered. It should raise an error.
        """
        lookup = self.create_fresh_lookup()

        lookup.register({'state': 0, 'mode': 0}, {'vocabulary_size': 10})

        # Input that doesn't match any registered patterns
        input_tensor = CBTensor.create_from_channels({'state': torch.tensor([1]), 'mode': torch.tensor([1])})

        with self.assertRaises(ValueError):
            lookup(input_tensor)

    def test_lookup_multiple_matches(self):
        """
        Test the behavior when multiple patterns match, which should raise an error.
        """
        lookup = self.create_fresh_lookup()

        lookup.register({'state': 0, 'mode': 0}, {'vocabulary_size': 10})
        lookup.register({'state': 0, 'mode': 0}, {'vocabulary_size': 100})

        # Input tensor that could match both patterns (ambiguous)
        input_tensor = CBTensor.create_from_channels({'state': torch.tensor([0]), 'mode': torch.tensor([0])})

        with self.assertRaises(ValueError):
            lookup(input_tensor)

    def test_register_invalid_spec(self):
        """
        Test trying to register an input or output pattern that doesn't conform to the expected spec.
        """
        lookup = self.create_fresh_lookup()

        # Input pattern has a missing 'mode' key
        with self.assertRaises(ValueError):
            lookup.register({'state': 0}, {'vocabulary_size': 10})

        # Output pattern has an invalid 'vocabulary_size' shape
        with self.assertRaises(ValueError):
            lookup.register({'state': 0, 'mode': 0}, {'vocabulary_size': [10, 20]})

class TestCBReplaceOnMatch(unittest.TestCase):

    def setUp(self):
        """
        Set up the input and output specs, and initialize the necessary parameters.
        """
        self.input_spec = CBTensorSpec({'state': 1, 'mode': 1})
        self.output_spec = CBTensorSpec({'state': 1})  # We are only replacing 'state' in this example.
        self.device = torch.device('cpu')
        self.dtype = torch.long

    def create_fresh_replace_on_match(self):
        """
        This function creates and returns a new CBReplaceOnMatch instance
        to ensure no cross-test contamination.
        """
        return CBReplaceOnMatch(self.input_spec, self.output_spec, self.device, self.dtype)

    def test_register_and_replace_single_pattern(self):
        """
        Test registering a single pattern and checking if it replaces the expected output.
        """
        replace_on_match = self.create_fresh_replace_on_match()

        input_pattern = {'state': 0, 'mode': 0}
        output_pattern = {'state': 1}

        # Register the pattern
        replace_on_match.register(input_pattern, output_pattern)

        # Create an input tensor with 'state' and 'mode'
        input_tensor = CBTensor.create_from_channels({'state': torch.tensor([0]), 'mode': torch.tensor([0])})
        output_tensor = replace_on_match(input_tensor)

        # The expected output should now have 'state' replaced with 1
        expected_tensor = CBTensor.create_from_channels({'state': torch.tensor([1]), 'mode': torch.tensor([0])})
        self.assertTrue(torch.equal(output_tensor.get_tensor(), expected_tensor.get_tensor()))

    def test_register_and_replace_multiple_patterns(self):
        """
        Test registering multiple patterns and ensuring the correct replacements happen.
        """
        replace_on_match = self.create_fresh_replace_on_match()

        # Register multiple patterns
        replace_on_match.register({'state': 0, 'mode': 0}, {'state': 1})
        replace_on_match.register({'state': 1, 'mode': 0}, {'state': 2})
        replace_on_match.register({'state': 1, 'mode': 1}, {'state': 3})

        # Create pattern. This will
        # 1: Match the first case
        # 2: Match the second case
        # 3: Match the third case
        # 4: Match no case

        state_pattern = torch.tensor([[0], [1], [1], [2]])
        mode_pattern = torch.tensor([[0], [0], [1], [0]])
        input_tensor = CBTensor.create_from_channels({"state" : state_pattern, "mode" : mode_pattern})

        # Define the expected output pattern
        state_pattern = torch.tensor([[1], [2], [3], [2]])
        mode_pattern = torch.tensor([[0], [0], [1], [0]])
        expected_output = CBTensor.create_from_channels({"state" : state_pattern, "mode" : mode_pattern})

        # Run test

        actual_output= replace_on_match(input_tensor)
        self.assertTrue(torch.equal(actual_output.get_tensor(), expected_output.get_tensor()))

    def test_no_match_no_replacement(self):
        """
        Test attempting to replace with no matching pattern. No replacement should occur.
        """
        replace_on_match = self.create_fresh_replace_on_match()

        # Register a pattern
        replace_on_match.register({'state': 0, 'mode': 0}, {'state': 1})

        # Input that doesn't match any registered pattern
        input_tensor = CBTensor.create_from_channels({'state': torch.tensor([2]), 'mode': torch.tensor([2])})

        # No replacement should happen; input should remain unchanged
        output_tensor = replace_on_match(input_tensor)
        self.assertTrue(torch.equal(output_tensor.get_tensor(), input_tensor.get_tensor()))

    def test_multiple_matches_error(self):
        """
        Test the behavior when multiple patterns match, which should raise an error.
        """
        replace_on_match = self.create_fresh_replace_on_match()

        replace_on_match.register({'state': 0, 'mode': 0}, {'state': 1})
        replace_on_match.register({'state': 0, 'mode': 0}, {'state': 2})

        # Input tensor that could match both patterns (ambiguous)
        input_tensor = CBTensor.create_from_channels({'state': torch.tensor([0]), 'mode': torch.tensor([0])})

        with self.assertRaises(ValueError):
            replace_on_match(input_tensor)

    def test_batch_replacement(self):
        """
        Test the replacement behavior across a batch of inputs.
        """
        replace_on_match = self.create_fresh_replace_on_match()

        # Register multiple patterns
        replace_on_match.register({'state': 0, 'mode': 0}, {'state': 1})
        replace_on_match.register({'state': 1, 'mode': 0}, {'state': 3})

        # Batch of input tensors
        state = torch.randint(0, 1, [20, 30, 10, 12, 1])
        mode = torch.randint(0, 1, [20, 30, 10, 12, 1])
        input_tensor = CBTensor.create_from_channels({"mode" : mode, "state" : state})
        output_tensor = replace_on_match(input_tensor)
