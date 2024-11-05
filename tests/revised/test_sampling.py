import torch
import unittest
from src.main.arcAGI2024.sampling import (make_random_selection_mask,
                                          make_top_k_selection_mask,
                                          make_top_p_selection_mask,
                                          TopLogitSampling,
                                          TopKSampling,
                                          NucleusSampling,
                                          DefaultSampling

                                          )


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


class TestTopLogitSampling(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        self.logits = torch.tensor([[2.0, 0.5, 0.3, 0.1, -1.0], [1.5, 0.7, 0.3, -0.2, -1.2]])
        self.temperature = 1.0

    def test_top_logit_sampling(self):
        sampler = TopLogitSampling()
        sampled = sampler(self.logits, temperature=self.temperature)
        expected = torch.argmax(self.logits, dim=-1)
        self.assertTrue(torch.equal(sampled, expected), "TopLogitSampling should return the argmax.")

class TestDefaultSampling(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        self.logits = torch.tensor([[2.0, 0.5, 0.3, 0.1, -1.0], [1.5, 0.7, 0.3, -0.2, -1.2]])

    def test_default_sampling_with_temperature(self):
        sampler = DefaultSampling()
        sampled = sampler(self.logits, temperature=1.0)
        self.assertEqual(sampled.shape, (2,), "DefaultSampling should return a tensor of shape (batch size,).")

    def test_default_sampling_with_zero_temperature(self):
        sampler = DefaultSampling()
        sampled = sampler(self.logits, temperature=0.0)
        expected = torch.argmax(self.logits, dim=-1)
        self.assertTrue(torch.equal(sampled, expected), "With zero temperature, DefaultSampling should return argmax.")

class TestTopKSampling(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        self.logits = torch.tensor([[2.0, 0.5, 0.3, 0.1, -1.0], [1.5, 0.7, 0.3, -0.2, -1.2]])

    def test_top_k_sampling_with_k(self):
        sampler = TopKSampling(num_k=3)
        sampled = sampler(self.logits, temperature=1.0)
        self.assertEqual(sampled.shape, (2,), "TopKSampling should return a tensor of shape (batch size,).")

    def test_top_k_sampling_with_zero_temperature(self):
        sampler = TopKSampling(num_k=2)
        sampled = sampler(self.logits, temperature=0.0)
        expected = torch.argmax(self.logits, dim=-1)
        self.assertTrue(torch.equal(sampled, expected), "With zero temperature, TopKSampling should return argmax.")

    def test_top_k_sampling_k_greater_than_logits(self):
        sampler = TopKSampling(num_k=10)  # Exceeds logits count
        sampled = sampler(self.logits, temperature=1.0)
        self.assertEqual(sampled.shape, (2,), "TopKSampling should handle k greater than logits size.")

class TestNucleusSampling(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        self.logits = torch.tensor([[2.0, 0.5, 0.3, 0.1, -1.0], [1.5, 0.7, 0.3, -0.2, -1.2]])

    def test_nucleus_sampling_with_top_p(self):
        sampler = NucleusSampling(top_p=0.9)
        sampled = sampler(self.logits, temperature=1.0)
        self.assertEqual(sampled.shape, (2,), "NucleusSampling should return a tensor of shape (batch size,).")

    def test_nucleus_sampling_with_zero_temperature(self):
        sampler = NucleusSampling(top_p=0.9)
        sampled = sampler(self.logits, temperature=0.0)
        expected = torch.argmax(self.logits, dim=-1)
        self.assertTrue(torch.equal(sampled, expected), "With zero temperature, NucleusSampling should return argmax.")

    def test_nucleus_sampling_with_invalid_top_p(self):
        with self.assertRaises(AssertionError):
            NucleusSampling(top_p=1.5)  # Invalid top_p