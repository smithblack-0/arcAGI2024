import unittest
from src.old.CBTensors import CBTensorSpec


### Behavioral test suite
class TestCBTensorSpecBehavioral(unittest.TestCase):

    def test_spec_immutability(self):
        spec = CBTensorSpec({'channel_1': 5, 'channel_2': 3})
        with self.assertRaises(TypeError):
            spec.spec['channel_1'] = 10

    def test_channels_property(self):
        spec = CBTensorSpec({'channel_1': 5, 'channel_2': 3})
        self.assertEqual(spec.channels, ['channel_1', 'channel_2'])

    def test_channel_widths_property(self):
        spec = CBTensorSpec({'channel_1': 5, 'channel_2': 3})
        self.assertEqual(spec.channel_widths, {'channel_1': 5, 'channel_2': 3})

    def test_total_width_property(self):
        spec = CBTensorSpec({'channel_1': 5, 'channel_2': 3})
        self.assertEqual(spec.total_width, 8)

    def test_contains_method(self):
        spec = CBTensorSpec({'channel_1': 5, 'channel_2': 3})
        self.assertIn('channel_1', spec)
        self.assertNotIn('channel_3', spec)


class TestCBTensorSpecPrecomputed(unittest.TestCase):

    def setUp(self):
        self.precomputed_spec = {
            'channel_a': 4,
            'channel_b': 6,
            'channel_c': 2
        }
        self.spec = CBTensorSpec(self.precomputed_spec)

    def test_precomputed_channels(self):
        self.assertEqual(self.spec.channels, ['channel_a', 'channel_b', 'channel_c'])

    def test_precomputed_channel_widths(self):
        self.assertEqual(self.spec.channel_widths, {'channel_a': 4, 'channel_b': 6, 'channel_c': 2})

    def test_precomputed_total_width(self):
        self.assertEqual(self.spec.total_width, 12)  # 4 + 6 + 2

    def test_precomputed_start_index(self):
        self.assertEqual(self.spec.start_index, {'channel_a': 0, 'channel_b': 4, 'channel_c': 10})

    def test_precomputed_end_index(self):
        self.assertEqual(self.spec.end_index, {'channel_a': 4, 'channel_b': 10, 'channel_c': 12})

    def test_precomputed_slices(self):
        expected_slices = {
            'channel_a': slice(0, 4),
            'channel_b': slice(4, 10),
            'channel_c': slice(10, 12)
        }
        self.assertEqual(self.spec.slices, expected_slices)


if __name__ == '__main__':
    unittest.main()
