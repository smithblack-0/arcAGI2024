import unittest
import torch
import numpy as np

from src.model.helper_functions import sinusoidal_positional_encoding, eval_legendre, pope_positional_encoding, \
    PositionalEncodings


class TestPositionalEncodings(unittest.TestCase):

    def test_sinusoidal_positional_encoding_shape(self):
        seq_len = 50
        model_dim = 512
        encoding = sinusoidal_positional_encoding(seq_len, model_dim, gen_term=10000.0)
        self.assertEqual(encoding.shape, (seq_len, model_dim))

    def test_sinusoidal_positional_encoding_values(self):
        seq_len = 2
        model_dim = 4
        gen_term = 10000.0

        expected_first_term = np.array([0.0, 1.0, 0.0, 1.0])
        encoding = sinusoidal_positional_encoding(seq_len, model_dim, gen_term)
        np.testing.assert_almost_equal(encoding[0], expected_first_term, decimal=5)

    def test_pope_positional_encoding_shape(self):
        seq_len = 50
        model_dim = 512
        encoding = pope_positional_encoding(seq_len, model_dim, gen_term=10)
        self.assertEqual(encoding.shape, (seq_len, model_dim))

    def test_pope_positional_encoding_values(self):
        seq_len = 2
        model_dim = 4
        order = 10
        position = np.linspace(-1, 1, seq_len)
        expected = np.zeros((seq_len, model_dim))
        for i in range(model_dim):
            expected[:, i] = eval_legendre(order, position) * ((i + 1) / model_dim)
        encoding = pope_positional_encoding(seq_len, model_dim, order)
        np.testing.assert_almost_equal(encoding, expected, decimal=5)

    def test_pope_positional_encoding_non_zero(self):
        seq_len = 50
        model_dim = 512
        encoding = pope_positional_encoding(seq_len, model_dim, gen_term=10)
        self.assertTrue(np.any(encoding != 0), "Positional encoding should not be all zeros.")

    def test_eval_legendre(self):
        # Test known values of Legendre polynomials
        x_values = np.array([-1, 0, 1])
        # P_0(x) = 1
        expected_p0 = np.array([1, 1, 1])
        np.testing.assert_almost_equal(eval_legendre(0, x_values), expected_p0, decimal=5)

        # P_1(x) = x
        expected_p1 = np.array([-1, 0, 1])
        np.testing.assert_almost_equal(eval_legendre(1, x_values), expected_p1, decimal=5)

        # P_2(x) = 0.5 * (3x^2 - 1)
        expected_p2 = np.array([1, -0.5, 1])
        np.testing.assert_almost_equal(eval_legendre(2, x_values), expected_p2, decimal=5)

        # P_3(x) = 0.5 * (5x^3 - 3x)
        expected_p3 = np.array([-1, 0, 1])
        np.testing.assert_almost_equal(eval_legendre(3, x_values), expected_p3, decimal=5)


class TestComputeSizes(unittest.TestCase):
    def test_compute_sizes(self):
        # Define mask
        mask = torch.zeros(2, 5, 5, dtype=torch.bool)
        mask[0, :3, :4] = True
        mask[1, :2, :3] = True

        # Define shapes
        expected_shapes = [(3, 4), (2, 3)]

        # Compute and compare
        outcome = PositionalEncodings.compute_shape_size(mask)
        for actual, expected in zip(outcome.unbind(0), expected_shapes):
            actual = (actual[0], actual[1])
            self.assertEqual(actual, expected)


class TestPositionalEncoding(unittest.TestCase):
    def setUp(self):
        self.embedding_dim = 4
        self.gen_term = 100
        self.gen_function = sinusoidal_positional_encoding

    def test_1d_pos_encoding_constructor(self):
        layer = PositionalEncodings(self.embedding_dim, 20, self.gen_function, self.gen_term)

    def test_2d_pos_encoding_constructor(self):
        layer = PositionalEncodings(self.embedding_dim, [5, 20], self.gen_function, self.gen_term)

    def test_1d_encoding(self):
        # Make layer
        max_sequence_length = 14
        layer = PositionalEncodings(self.embedding_dim, max_sequence_length, self.gen_function, self.gen_term)

        # Define mask
        mask = torch.zeros(2, 5, dtype=torch.bool)
        mask[0, :3] = True
        mask[1, :2] = True

        # Apply encoding
        output = layer(mask)

    def test_2d_encoding(self):
        # make layer
        max_seq_length = [20, 15]
        layer = PositionalEncodings(self.embedding_dim, max_seq_length, self.gen_function, self.gen_term)

        # Define mask

        mask = torch.zeros(2, 5, 10, dtype=torch.bool)
        mask[0, :3, :4] = True
        mask[1, :2, :7] = True

        # Make encoding
        output = layer(mask)


