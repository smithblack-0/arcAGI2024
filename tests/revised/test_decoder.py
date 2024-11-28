import unittest
import torch
from src.main.arcAGI2024 import (RecurrentDecoder,
                                 FeedforwardConfig,
                                 LinearMemoryConfig,
                                 GradientTimeLossConfig,
                                 MemRegularizationLossConfig,
                                 RecurrentDecoderLayerConfig,
                                 RecurrentDecoderConfig,
                                 RecurrentDecoderLayer,
                                 parallel_pytree_map, get_rng_state, set_rng_state)
class TestDecoderLayer(unittest.TestCase):
    def setUp(self):
        """
        Set up defaults for all the various config entries.
        """
        feedforward_config = FeedforwardConfig(
            d_hidden=256,
            num_internal_layers=2,
            feedforward_dropout = 0.01
        )

        memory_gradient_loss_config = GradientTimeLossConfig(
            num_bins=4,
            target_distribution=[0.25, 0.25, 0.25, 0.25],
            target_thresholds=[0.1, 0.1, 0.1, 0.1],
            loss_weight=1000.0,
            loss_type="quadratic_threshold",
            z_score=1.0
        )

        memory_regularization_loss_config = MemRegularizationLossConfig(
            magnitude_loss_type= 'l2',
            magnitude_loss_weight = 10.0
        )

        memory_config = LinearMemoryConfig(
            num_heads=10,
            d_address=32,
            d_memory=32,
            gradient_loss=memory_gradient_loss_config,
            mem_regularization_loss=memory_regularization_loss_config
        )

        layer_config = RecurrentDecoderLayerConfig(
            d_bottleneck=128,
            bottlenecked_dropout_rate=0.1,
            feedforward_config=feedforward_config,
            memory_config=memory_config,
        )

        self.layer_config = layer_config
        self.d_model = 768
        self.device= torch.device('cpu')
        self.dtype = torch.float32

    def make_layer(self)->RecurrentDecoderLayer:
        return RecurrentDecoderLayer(self.d_model, self.dtype, self.device, self.layer_config)
    def test_forward_backward_syncronous_restored_random(self):
        # Create the mock test data
        batch_size = 10
        mock_data = torch.randn([batch_size, self.d_model], device=self.device, dtype=self.dtype)
        mock_mask = torch.rand([batch_size]) > 0.5

        # Create and setup the model
        model = self.make_layer()
        original_state = model.create_state([batch_size])

        # run a forward step
        rng = get_rng_state(mock_data.device)
        forward_outputs, forward_next_state = model(mock_data, mock_mask, original_state)

        # Second forward ouptuts
        set_rng_state(rng, mock_data.device)
        second_forward_outputs, _ = model(mock_data, mock_mask, original_state)
        self.assertTrue(torch.allclose(forward_outputs, second_forward_outputs))

        # Run the reverse step
        set_rng_state(rng, mock_data.device)
        (reverse_outputs, reverse_next_state), reverse_original_state = model.reverse(mock_data, mock_mask, forward_next_state)

        # See if they are similar
        self.assertTrue(torch.allclose(forward_outputs, reverse_outputs))
        def check_memories(tensor: torch.Tensor, other_tensor: torch.Tensor):
            self.assertTrue(torch.allclose(tensor, other_tensor))

        parallel_pytree_map(check_memories, forward_next_state, reverse_next_state)
        parallel_pytree_map(check_memories, original_state, reverse_original_state)

class TestDecoder(unittest.TestCase):
    def setUp(self):
        feedforward_config = FeedforwardConfig(
            d_hidden=256,
            num_internal_layers=2,
            feedforward_dropout=0.01
        )

        memory_gradient_loss_config = GradientTimeLossConfig(
            num_bins=4,
            target_distribution=[0.25, 0.25, 0.25, 0.25],
            target_thresholds=[0.1, 0.1, 0.1, 0.1],
            loss_weight=1000.0,
            loss_type="quadratic_threshold",
            z_score=1.0
        )

        memory_regularization_loss_config = MemRegularizationLossConfig(
            magnitude_loss_type='l2',
            magnitude_loss_weight=10.0
        )

        memory_config = LinearMemoryConfig(
            num_heads=10,
            d_address=32,
            d_memory=32,
            gradient_loss=memory_gradient_loss_config,
            mem_regularization_loss=memory_regularization_loss_config
        )

        layer_config = RecurrentDecoderLayerConfig(
            d_bottleneck=128,
            bottlenecked_dropout_rate=0.1,
            feedforward_config=feedforward_config,
            memory_config=memory_config,
        )

        self.layer_config = layer_config
        self.d_model = 768
        self.device = torch.device('cpu')
        self.dtype = torch.float32

    def make_decoder_layer(self, dropout_rate: float)->DeepDecoderLayer:
        return DeepDecoderLayer(
            self.d_model,
            self.d_hidden,
            self.d_address,
            self.d_memory,
            self.num_read_heads,
            self.num_write_heads,
            self.num_memories,
            self.numeric_write_factor,
            dropout_rate,
            self.device,
            self.dtype
        )

    def make_model(self, dropout_rate: float, main_dropout_rate: float)->RecurrentDecoder:
        layers = [self.make_decoder_layer(dropout_rate) for _ in range(3)]
        return RecurrentDecoder(self.d_main, dropout_rate=main_dropout_rate, decoder_layers=layers,
                                dtype=self.dtype, device=self.device)
    def test_forward_backward_syncronous(self):
        # Create the mock test data
        batch_size = 10
        mock_data = torch.randn([batch_size, self.d_main], device=self.device, dtype=self.dtype)
        mock_mask = torch.rand([batch_size]) > 0.5

        # Create and setup the model
        model = self.make_model(dropout_rate=0.0, main_dropout_rate=0.0)
        original_state = model.create_state([batch_size])

        # run a forward step
        forward_outputs, forward_next_state = model(mock_data, mock_mask, original_state)

        # Run the reverse step
        (reverse_outputs, reverse_next_state), reverse_original_state = model.reverse(mock_data, mock_mask, forward_next_state)

        # See if they are similar
        self.assertTrue(torch.allclose(forward_outputs, reverse_outputs))
        def check_memories(tensor: torch.Tensor, other_tensor: torch.Tensor):
            self.assertTrue(torch.allclose(tensor, other_tensor))

        parallel_pytree_map(check_memories, forward_next_state, reverse_next_state)
        parallel_pytree_map(check_memories, original_state, reverse_original_state)
    def test_forward_backward_syncronous_dropout_restored(self):
        # Create the mock test data
        batch_size = 10
        mock_data = torch.randn([batch_size, self.d_main], device=self.device, dtype=self.dtype)
        mock_mask = torch.rand([batch_size]) > 0.5

        # Create and setup the model
        model = self.make_model(dropout_rate=0.2, main_dropout_rate=0.2)
        original_state = model.create_state([batch_size])

        # run a forward step
        rng = get_rng_state(mock_data.device)
        forward_outputs, forward_next_state = model(mock_data, mock_mask, original_state)

        # Second forward ouptuts
        set_rng_state(rng, mock_data.device)
        second_forward_outputs, _ = model(mock_data, mock_mask, original_state)
        self.assertTrue(torch.allclose(forward_outputs, second_forward_outputs))

        # Run the reverse step
        set_rng_state(rng, mock_data.device)
        (reverse_outputs, reverse_next_state), reverse_original_state = model.reverse(mock_data, mock_mask, forward_next_state)

        # See if they are similar
        self.assertTrue(torch.allclose(forward_outputs, reverse_outputs))
        def check_memories(tensor: torch.Tensor, other_tensor: torch.Tensor):
            self.assertTrue(torch.allclose(tensor, other_tensor))

        parallel_pytree_map(check_memories, forward_next_state, reverse_next_state)
        parallel_pytree_map(check_memories, original_state, reverse_original_state)

