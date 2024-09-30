import unittest
import torch
from src.main.CBTensors.channel_bound_tensors import CBTensor  # Replace 'your_module' with your actual module
from src.main.CBTensors.channel_bound_spec import CBTensorSpec


class TestCBTensorValidation(unittest.TestCase):

    def setUp(self):
        # Sample CBTensorSpec and CBTensor for use in validation tests
        self.spec = CBTensorSpec({'channel_1': 5, 'channel_2': 3})
        self.tensor = CBTensor(self.spec, torch.randn([2, 8]))

    def test_validate_cb_tensor(self):
        # Should not raise error when provided with CBTensor
        self.tensor.validate_cb_tensor(self.tensor)

        # Should raise ValueError when not a CBTensor
        with self.assertRaises(ValueError) as context:
            self.tensor.validate_cb_tensor("not a tensor")
        self.assertIn("Expected a CBTensor", str(context.exception))

    def test_validate_channels_exist(self):
        # Valid case: source channels are in destination channels
        source_channels = ['channel_1']
        destination_channels = self.spec.channels
        self.tensor.validate_channels_exist(source_channels, destination_channels)

        # Invalid case: source channels are not in destination
        invalid_source_channels = ['channel_3']
        with self.assertRaises(ValueError) as context:
            self.tensor.validate_channels_exist(invalid_source_channels, destination_channels)
        self.assertIn("channel_3", str(context.exception))

    def test_validate_common_channel_widths(self):
        # Valid case: source and destination have the same channel widths
        source_widths = {'channel_1': 5, 'channel_2': 3}
        destination_widths = self.spec.channel_widths
        self.tensor.validate_common_channel_widths(source_widths, destination_widths)

        # Invalid case: mismatched channel widths
        mismatched_source_widths = {'channel_1': 4, 'channel_2': 3}
        with self.assertRaises(ValueError) as context:
            self.tensor.validate_common_channel_widths(mismatched_source_widths, destination_widths)
        self.assertIn("Channel widths are mismatched", str(context.exception))

    def test_validate_broadcastable(self):
        # Valid case: source and destination shapes are broadcastable
        source_shape = (1, 3)
        destination_shape = (2, 3)
        self.tensor.validate_broadcastable(destination_shape, source_shape)

        # Invalid case: non-broadcastable shapes
        non_broadcastable_shapes = [(2, 3), (4, 4)]
        with self.assertRaises(ValueError) as context:
            self.tensor.validate_broadcastable(*non_broadcastable_shapes)
        self.assertIn("broadcast", str(context.exception))
class TestCBTensorSpecPassthrough(unittest.TestCase):

    def setUp(self):
        # Sample CBTensorSpec and CBTensor for use in passthrough tests
        self.spec = CBTensorSpec({'channel_1': 5, 'channel_2': 3})
        # The total channel width is 8, and the batch size is 2
        self.tensor = CBTensor(self.spec, torch.randn([2, 8]))

    def test_channels_passthrough(self):
        # Test that the channels passthrough correctly
        self.assertEqual(self.tensor.channels, ['channel_1', 'channel_2'])

    def test_channel_widths_passthrough(self):
        # Test that channel_widths are passed through correctly
        self.assertEqual(self.tensor.channel_widths, {'channel_1': 5, 'channel_2': 3})

    def test_total_channel_width_passthrough(self):
        # Test that total_channel_width is passed through correctly
        self.assertEqual(self.tensor.total_channel_width, 8)  # 5 + 3

    def test_channel_start_index_passthrough(self):
        # Test that start_index is passed through correctly
        self.assertEqual(self.tensor.channel_start_index, {'channel_1': 0, 'channel_2': 5})

    def test_channel_end_index_passthrough(self):
        # Test that end_index is passed through correctly
        self.assertEqual(self.tensor.channel_end_index, {'channel_1': 5, 'channel_2': 8})

    def test_slices_passthrough(self):
        # Test that slices are passed through correctly
        expected_slices = {
            'channel_1': slice(0, 5),
            'channel_2': slice(5, 8)
        }
        self.assertEqual(self.tensor.slices, expected_slices)

    def test_shape_passthrough(self):
        # Test that the shape property correctly excludes the channel dimension
        self.assertEqual(self.tensor.shape, (2,))  # Excluding the channel dimension

    def test_dim_passthrough(self):
        # Test that the dim property returns the correct number of dimensions excluding the channel dimension
        self.assertEqual(self.tensor.dim(), 1)  # Excluding the channel dimension



class TestCBTensorCreateAndSeparate(unittest.TestCase):

    def setUp(self):
        # Setting up sample tensors for testing
        self.tensor_channels = {
            'channel_1': torch.randn([2, 5]),  # Batch size 2, 5 elements in the channel
            'channel_2': torch.randn([2, 3])   # Batch size 2, 3 elements in the channel
        }

    def test_create_from_channels(self):
        # Create a CBTensor from separate channels
        cb_tensor = CBTensor.create_from_channels(self.tensor_channels)

        # Check that the tensor has the expected shape (batch_size, total_channel_width)
        expected_shape = (2, 8)  # 5 + 3 = 8 (total width)
        self.assertEqual(cb_tensor.get_tensor().shape, expected_shape)

    def test_separate_into_channels(self):
        # Create a CBTensor and then separate it into channels
        cb_tensor = CBTensor.create_from_channels(self.tensor_channels)
        separated_channels = cb_tensor.separate_into_channels()

        # Ensure that the separated channels match the original tensor channels
        for channel_name, original_tensor in self.tensor_channels.items():
            separated_tensor = separated_channels[channel_name]
            self.assertTrue(torch.equal(original_tensor, separated_tensor),
                            f"Channel {channel_name} does not match the original tensor")

    def test_create_and_separate_consistency(self):
        # Test the consistency of creating and separating a CBTensor
        cb_tensor = CBTensor.create_from_channels(self.tensor_channels)
        separated_channels = cb_tensor.separate_into_channels()

        # Check that creating from separated channels recreates the original tensor
        recreated_tensor = CBTensor.create_from_channels(separated_channels)
        self.assertTrue(torch.equal(cb_tensor.get_tensor(), recreated_tensor.get_tensor()),
                        "Recreated tensor does not match the original tensor after separating and creating")

    def test_incompatible_shapes(self):
        # Test with incompatible tensor shapes to check error handling
        invalid_channels = {
            'channel_1': torch.randn([3, 5]),  # Batch size 3
            'channel_2': torch.randn([2, 3])   # Batch size 2
        }
        with self.assertRaises(ValueError):
            CBTensor.create_from_channels(invalid_channels)

class TestCBTensorSetAndGatherChannels(unittest.TestCase):

    def setUp(self):
        # Set up some sample tensors and specs for testing
        self.tensor_channels = {
            'channel_1': torch.randn([2, 5]),
            'channel_2': torch.randn([2, 3]),
            'channel_3': torch.randn([2, 4])
        }

    def test_gather_channels_single(self):
        # Gather a single channel and ensure it matches the original
        tensor = CBTensor.create_from_channels(self.tensor_channels)
        gathered = tensor.gather_channels('channel_1')
        self.assertEqual(gathered.get_tensor().shape[-1], 5)  # Only channel_1 is 5 wide
        self.assertTrue(torch.equal(gathered.get_tensor(), self.tensor_channels['channel_1']))

    def test_gather_channels_multiple(self):
        # Gather multiple channels and ensure they match the originals
        tensor = CBTensor.create_from_channels(self.tensor_channels)
        gathered = tensor.gather_channels(['channel_1', 'channel_3'])
        expected_tensor = torch.cat([self.tensor_channels['channel_1'], self.tensor_channels['channel_3']], dim=-1)
        self.assertEqual(gathered.get_tensor().shape[-1], 9)  # 5 + 4 = 9
        self.assertTrue(torch.equal(gathered.get_tensor(), expected_tensor))

    def test_gather_channels_reorder(self):
        # Test that gathering channels in a different order works correctly
        tensor = CBTensor.create_from_channels(self.tensor_channels)
        gathered = tensor.gather_channels(['channel_3', 'channel_1'])
        expected_tensor = torch.cat([self.tensor_channels['channel_3'], self.tensor_channels['channel_1']], dim=-1)
        self.assertTrue(torch.equal(gathered.get_tensor(), expected_tensor))

    def test_set_channels(self):
        # Modify one of the channels and set it back
        tensor = CBTensor.create_from_channels(self.tensor_channels)
        modified_channel_1 = torch.zeros_like(self.tensor_channels['channel_1'])
        new_tensor = CBTensor.create_from_channels({
            'channel_1': modified_channel_1,
            'channel_2': self.tensor_channels['channel_2'],
            'channel_3': self.tensor_channels['channel_3']
        })

        # Set the modified channel back into the original tensor
        updated_tensor = tensor.set_channels(new_tensor)

        # Check that the channel was set properly, and others were unchanged
        separated_channels = updated_tensor.separate_into_channels()
        self.assertTrue(torch.equal(separated_channels['channel_1'], modified_channel_1))
        self.assertTrue(torch.equal(separated_channels['channel_2'], self.tensor_channels['channel_2']))
        self.assertTrue(torch.equal(separated_channels['channel_3'], self.tensor_channels['channel_3']))

    def test_gather_and_set_channels(self):
        # Gather two channels, zero them out, and set them back into the original tensor
        tensor = CBTensor.create_from_channels(self.tensor_channels)
        gathered = tensor.gather_channels(['channel_1', 'channel_2'])
        zeroed_tensor = torch.zeros_like(gathered.get_tensor())
        updated_gathered = CBTensor(gathered.spec, zeroed_tensor)

        # Set the zeroed channels back into the original tensor
        updated_tensor = tensor.set_channels(updated_gathered)

        # Verify that the zeroed channels were updated
        separated_channels = updated_tensor.separate_into_channels()
        self.assertTrue(torch.equal(separated_channels['channel_1'], torch.zeros_like(self.tensor_channels['channel_1'])))
        self.assertTrue(torch.equal(separated_channels['channel_2'], torch.zeros_like(self.tensor_channels['channel_2'])))
        # Ensure the untouched channel remains the same
        self.assertTrue(torch.equal(separated_channels['channel_3'], self.tensor_channels['channel_3']))

    def test_set_channels_broadcast(self):
        # Test broadcasting: Set a channel with a single batch dimension and broadcast it
        tensor = CBTensor.create_from_channels(self.tensor_channels)
        broadcasted_channel_1 = torch.zeros([1, 5])  # Only one batch, should broadcast to two batches
        new_tensor = CBTensor.create_from_channels({
            'channel_1': broadcasted_channel_1,
            'channel_2': self.tensor_channels['channel_2'],
            'channel_3': self.tensor_channels['channel_3']
        })

        # Set and broadcast the modified channel back into the original tensor
        updated_tensor = tensor.set_channels(new_tensor)

        # Ensure broadcasting was handled correctly
        separated_channels = updated_tensor.separate_into_channels()
        expected_broadcast = broadcasted_channel_1.expand([2, 5])
        self.assertTrue(torch.equal(separated_channels['channel_1'], expected_broadcast))

    def test_invalid_channel_gather(self):
        # Test gathering a non-existent channel
        with self.assertRaises(ValueError):
            tensor = CBTensor.create_from_channels(self.tensor_channels)
            tensor.gather_channels('non_existent_channel')

    def test_invalid_channel_set(self):
        # Test setting a tensor with a non-existent channel
        invalid_tensor = CBTensor(CBTensorSpec({'channel_4': 2}), torch.randn([2, 2]))
        with self.assertRaises(ValueError):
            tensor = CBTensor.create_from_channels(self.tensor_channels)
            tensor.set_channels(invalid_tensor)
class TestCBTensorRebindToSpec(unittest.TestCase):

    def setUp(self):
        # Set up some sample tensor channels for testing
        self.tensor_channels = {
            'channel_1': torch.randn([2, 5]),  # Batch size 2, 5 elements in the channel
            'channel_2': torch.randn([2, 3]),  # Batch size 2, 3 elements in the channel
            'channel_3': torch.randn([2, 4])   # Batch size 2, 4 elements in the channel
        }

    def test_rebind_prune_channels(self):
        # Create a CBTensor and rebind it with a spec that removes some channels (prunes them)
        original_tensor = CBTensor.create_from_channels(self.tensor_channels)

        # New spec that removes 'channel_2'
        new_spec = CBTensorSpec({
            'channel_1': 5,
            'channel_3': 4  # Pruned 'channel_2'
        })

        # Rebind the tensor with pruning allowed
        pruned_tensor = original_tensor.rebind_to_spec(new_spec, allow_channel_pruning=True)

        # Check that 'channel_2' was pruned
        separated_channels = pruned_tensor.separate_into_channels()
        self.assertEqual(list(separated_channels.keys()), ['channel_1', 'channel_3'])
        self.assertNotIn('channel_2', separated_channels)

        # Ensure that channel_1 and channel_3 have the correct data from the original tensor
        self.assertTrue(torch.equal(separated_channels['channel_1'], self.tensor_channels['channel_1']))
        self.assertTrue(torch.equal(separated_channels['channel_3'], self.tensor_channels['channel_3']))

    def test_rebind_prune_not_allowed(self):
        # Create a CBTensor and attempt to rebind it without allowing pruning
        original_tensor = CBTensor.create_from_channels(self.tensor_channels)

        # New spec that is missing 'channel_2'
        invalid_spec = CBTensorSpec({
            'channel_1': 5,
            'channel_3': 4  # Missing 'channel_2'
        })

        # Rebinding should raise a ValueError because 'channel_2' is missing and pruning is not allowed
        with self.assertRaises(ValueError):
            original_tensor.rebind_to_spec(invalid_spec, allow_channel_pruning=False)

    def test_rebind_reorder_and_add_channels(self):
        # Create a CBTensor and rebind it with a spec that reorders existing channels and adds a new one
        original_tensor = CBTensor.create_from_channels(self.tensor_channels)

        # New spec that reorders the channels and adds a new one
        new_spec = CBTensorSpec({
            'channel_3': 4,
            'channel_1': 5,
            'channel_2': 3,
            'channel_4': 2  # New channel added
        })

        # Rebind the tensor
        new_tensor = original_tensor.rebind_to_spec(new_spec, allow_channel_expansion=True)

        # Step 1: Ensure channels were reordered properly using separate_into_channels
        separated_channels = new_tensor.separate_into_channels()
        self.assertEqual(list(separated_channels.keys()), ['channel_3', 'channel_1', 'channel_2', 'channel_4'])
        self.assertTrue(torch.equal(separated_channels['channel_4'], torch.zeros([2, 2])))

        # Step 2: Direct tensor slice access to check that data was actually moved
        new_tensor_data = new_tensor.get_tensor()

        # Get the slices for each channel from the new spec
        channel_3_slice = new_tensor.slices['channel_3']
        channel_1_slice = new_tensor.slices['channel_1']
        channel_2_slice = new_tensor.slices['channel_2']

        # Ensure the slices contain the correct data from the original tensor
        self.assertTrue(torch.equal(new_tensor_data[..., channel_3_slice], self.tensor_channels['channel_3']))
        self.assertTrue(torch.equal(new_tensor_data[..., channel_1_slice], self.tensor_channels['channel_1']))
        self.assertTrue(torch.equal(new_tensor_data[..., channel_2_slice], self.tensor_channels['channel_2']))

    def test_rebind_missing_channel_error(self):
        # Create a CBTensor and attempt to rebind it with a spec that omits an original channel
        original_tensor = CBTensor.create_from_channels(self.tensor_channels)

        # New spec that is missing 'channel_2'
        invalid_spec = CBTensorSpec({
            'channel_1': 5,
            'channel_3': 4
        })

        # Rebinding should raise a ValueError because 'channel_2' is missing
        with self.assertRaises(ValueError):
            original_tensor.rebind_to_spec(invalid_spec)

    def test_rebind_add_channels(self):
        # Create a CBTensor and rebind it with a new spec that adds a new channel
        original_tensor = CBTensor.create_from_channels(self.tensor_channels)

        # New spec that includes an additional channel
        new_spec = CBTensorSpec({
            'channel_1': 5,
            'channel_2': 3,
            'channel_3': 4,
            'channel_4': 2  # New channel added
        })

        # Rebind the tensor
        new_tensor = original_tensor.rebind_to_spec(new_spec, allow_channel_expansion=True)

        # Check that the new channel was added and initialized with zeros
        separated_channels = new_tensor.separate_into_channels()
        self.assertIn('channel_4', separated_channels)
        self.assertTrue(torch.equal(separated_channels['channel_4'], torch.zeros([2, 2])))

class TestCBTensorGetItem(unittest.TestCase):

    def setUp(self):
        # Set up tensor channels for each test to use
        # Tensor shape is now (3, 2, 5, 6, 4, channel)
        self.tensor_channels = {
            'channel_1': torch.rand([3, 2, 5, 6, 4, 5]),  # Shape (3, 2, 5, 6, 4, 5) for channel_1
            'channel_2': torch.rand([3, 2, 5, 6, 4, 3])   # Shape (3, 2, 5, 6, 4, 3) for channel_2
        }

    def test_getitem_ellipsis_position_end(self):
        # Initialize CBTensor within the method
        cb_tensor = CBTensor.create_from_channels(self.tensor_channels)

        # Test ellipsis with slice at the beginning [3, ...]
        result = cb_tensor[2, ...]
        expected_result = cb_tensor.get_tensor()[2, ...]  # Selects first element in the first dimension
        self.assertTrue(torch.equal(result.get_tensor(), expected_result),
                        "Ellipsis with slice [3, ...] failed")

    def test_getitem_ellipsis_position_middle(self):
        # Initialize CBTensor within the method
        cb_tensor = CBTensor.create_from_channels(self.tensor_channels)

        # Test ellipsis in the middle [:, ..., :]
        result = cb_tensor[:, ..., :]
        expected_result = torch.cat([self.tensor_channels['channel_1'], self.tensor_channels['channel_2']], dim=-1)
        self.assertTrue(torch.equal(result.get_tensor(), expected_result),
                        "Ellipsis in the middle failed")

    def test_getitem_ellipsis_position_start(self):
        # Initialize CBTensor within the method
        cb_tensor = CBTensor.create_from_channels(self.tensor_channels)

        # Test ellipsis at the start [..., :, :]
        result = cb_tensor[..., :, :]
        expected_result = torch.cat([self.tensor_channels['channel_1'], self.tensor_channels['channel_2']], dim=-1)
        self.assertTrue(torch.equal(result.get_tensor(), expected_result),
                        "Ellipsis at the start failed")

    def test_getitem_integer_indexing(self):
        # Initialize CBTensor within the method
        cb_tensor = CBTensor.create_from_channels(self.tensor_channels)

        # Test integer indexing [0, 0, 0, 0, 0] (single element)
        result = cb_tensor[0, 0, 0, 0, 0]
        expected_result = cb_tensor.get_tensor()[0, 0, 0, 0, 0]
        self.assertTrue(torch.equal(result.get_tensor(), expected_result),
                        "Integer indexing failed")

    def test_getitem_slice_indexing(self):
        # Initialize CBTensor within the method
        cb_tensor = CBTensor.create_from_channels(self.tensor_channels)

        # Test slice indexing [0:2, 0:1, 0:3, 1:4] (range)
        result = cb_tensor[0:2, 0:1, 0:3, 1:4]
        expected_result = cb_tensor.get_tensor()[0:2, 0:1, 0:3, 1:4, :, :]
        self.assertTrue(torch.equal(result.get_tensor(), expected_result),
                        "Slice indexing failed")

    def test_invalid_channel_indexing(self):
        # Initialize CBTensor within the method
        cb_tensor = CBTensor.create_from_channels(self.tensor_channels)

        # Attempt to index the channel dimension directly, which should raise an error
        with self.assertRaises(IndexError):
            cb_tensor[0, 0, 0, 0, 0, 0]

class TestCBTensorSetItem(unittest.TestCase):

    def setUp(self):
        # Set up tensor channels for each test to use
        # Tensor shape is now (3, 2, 5, 6, 4, channel)
        self.tensor_channels = {
            'channel_1': torch.rand([3, 2, 5, 6, 4, 5]),  # Shape (3, 2, 5, 6, 4, 5) for channel_1
            'channel_2': torch.rand([3, 2, 5, 6, 4, 3])   # Shape (3, 2, 5, 6, 4, 3) for channel_2
        }

    def test_setitem_ellipsis_position_end(self):
        # Initialize CBTensor within the method
        cb_tensor = CBTensor.create_from_channels(self.tensor_channels)

        # Test ellipsis with slice at the beginning [2, ...]
        new_data = torch.zeros_like(cb_tensor.get_tensor()[2, ...])
        cb_tensor[2, ...] = CBTensor(cb_tensor.spec, new_data)
        self.assertTrue(torch.equal(cb_tensor.get_tensor()[2, ...], new_data),
                        "Setting with ellipsis [2, ...] failed")

    def test_setitem_ellipsis_position_middle(self):
        # Initialize CBTensor within the method
        cb_tensor = CBTensor.create_from_channels(self.tensor_channels)

        # Test ellipsis in the middle [:, ..., :]
        new_data = torch.zeros_like(cb_tensor.get_tensor()[:, ..., :])
        cb_tensor[:, ..., :] = CBTensor(cb_tensor.spec, new_data)
        self.assertTrue(torch.equal(cb_tensor.get_tensor()[:, ..., :], new_data),
                        "Setting with ellipsis [:, ..., :] failed")

    def test_setitem_ellipsis_position_start(self):
        # Initialize CBTensor within the method
        cb_tensor = CBTensor.create_from_channels(self.tensor_channels)

        # Test ellipsis at the start [..., :, :]
        new_data = torch.zeros_like(cb_tensor.get_tensor()[..., :, :])
        cb_tensor[..., :, :] = CBTensor(cb_tensor.spec, new_data)
        self.assertTrue(torch.equal(cb_tensor.get_tensor()[..., :, :], new_data),
                        "Setting with ellipsis [..., :, :] failed")

    def test_setitem_integer_indexing(self):
        # Initialize CBTensor within the method
        cb_tensor = CBTensor.create_from_channels(self.tensor_channels)

        # Test integer indexing [0, 0, 0, 0, 0]
        new_data = torch.zeros_like(cb_tensor.get_tensor()[0, 0, 0, 0, 0, :])
        cb_tensor[0, 0, 0, 0, 0] = CBTensor(cb_tensor.spec, new_data.unsqueeze(0))
        self.assertTrue(torch.equal(cb_tensor.get_tensor()[0, 0, 0, 0, 0, :], new_data),
                        "Integer indexing failed for __setitem__")

    def test_setitem_slice_indexing(self):
        # Initialize CBTensor within the method
        cb_tensor = CBTensor.create_from_channels(self.tensor_channels)

        # Test slice indexing [0:2, 0:1, 0:3, 1:4]
        new_data = torch.zeros_like(cb_tensor.get_tensor()[0:2, 0:1, 0:3, 1:4, :, :])
        cb_tensor[0:2, 0:1, 0:3, 1:4] = CBTensor(cb_tensor.spec, new_data)
        self.assertTrue(torch.equal(cb_tensor.get_tensor()[0:2, 0:1, 0:3, 1:4, :, :], new_data),
                        "Slice indexing failed for __setitem__")

    def test_invalid_channel_indexing(self):
        # Initialize CBTensor within the method
        cb_tensor = CBTensor.create_from_channels(self.tensor_channels)

        # Attempt to set data in the channel dimension directly, which should raise an error
        invalid_data = torch.rand([3, 2, 5, 6, 4, 1])
        with self.assertRaises(AssertionError):
            cb_tensor[0, 0, 0, 0, 0, 0] = CBTensor(cb_tensor.spec, invalid_data)


class TestCBTensorOperatorSupportRegistry(unittest.TestCase):

    def setUp(self):
        # Save the original supported_operators to restore after the test
        self.original_supported_operators = CBTensor.supported_operators.copy()

    def tearDown(self):
        # Restore the original supported_operators after each test to prevent contamination
        CBTensor.supported_operators = self.original_supported_operators.copy()

    def test_operator_override_mechanism(self):
        # Define a custom "foo" function to simulate a torch function
        def foo_func() -> str:
            return "foo was called"

        # Define a custom "buzz" function that should override "foo"
        @CBTensor.register_operator(foo_func)
        def buzz_func() -> str:
            return "buzz was called"

        # Register "buzz" to override "foo"
        CBTensor.register_operator(foo_func)(buzz_func)

        # Call __torch_func__ and verify that "buzz_func" is invoked instead of "foo_func"
        result = CBTensor.__torch_func__(foo_func, types=(), args=())
        self.assertEqual(result, "buzz was called", "Operator override mechanism failed")

    def test_operator_override_with_arguments(self):
        # Define a custom "foo" function to simulate a torch function with arguments
        def foo_func(arg1: int, arg2: str) -> str:
            return f"foo was called with {arg1} and {arg2}"

        # Define a custom "buzz" function that should override "foo" with the same arguments
        @CBTensor.register_operator(foo_func)
        def buzz_func(arg1: int, arg2: str) -> str:
            return f"buzz was called with {arg1} and {arg2}"

        # Call __torch_func__ with arguments and verify that "buzz_func" is invoked
        result = CBTensor.__torch_func__(foo_func, types=(), args=(42, 'test'))
        self.assertEqual(result, "buzz was called with 42 and test", "Operator override mechanism with arguments failed")

    def test_operator_unregistered(self):
        # Define an unregistered function
        def unregistered_func() -> str:
            return "unregistered function was called"

        # Ensure a ValueError is raised when an unregistered function is called
        with self.assertRaises(ValueError) as context:
            CBTensor.__torch_func__(unregistered_func, types=(), args=())
        self.assertIn("Torch function was not supported", str(context.exception))

    def test_operator_restore_after_override(self):
        # Define a custom "foo" function
        def foo_func() -> str:
            return "foo was called"

        # Define a custom "buzz" function to override "foo"
        @CBTensor.register_operator(foo_func)
        def buzz_func() -> str:
            return "buzz was called"

        # Ensure "buzz" function was registered successfully
        self.assertIn(foo_func, CBTensor.supported_operators)

        # Restore original operators and check that the override was removed
        CBTensor.supported_operators = self.original_supported_operators.copy()
        self.assertNotIn(foo_func, CBTensor.supported_operators, "Supported operators were not restored correctly")


