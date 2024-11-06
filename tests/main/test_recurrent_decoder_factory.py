import unittest
import torch
from torch import nn
from torch.profiler import profile, record_function, ProfilerActivity

from src.old.arcAGI2024 import build_recurrent_decoder_v1
from src.old.arcAGI2024 import parallel_pytree_map

class TestRecurrentDecoderBuilder(unittest.TestCase):
    """
    Unit tests for the build_recurrent_decoder_v1 builder function.
    """

    def setUp(self):
        # Basic parameters for constructing a RecurrentDecoder
        self.d_embedding = 1024
        self.d_model = 128
        self.bank_size = 40
        self.chunk_size = 512
        self.direct_dropout = 0.1
        self.submodule_dropout = 0.05
        self.control_dropout = 0.1
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.stack_depth = 8

        # Variants for the registry submodules
        self.deep_memory_variant = "FastLinearMemory"
        self.act_variant = "Default"
        self.support_stack_variant = "Default"
        self.virtual_layer_controller_variant = "LinearBankSelector"

        # Deep memory details
        self.mem_d_value = self.d_model
        self.num_memories = 100

        # Extra parameters in case they’re required for certain variants
        self.variant_context = {
            "d_address": self.d_model // 8,
            "d_memory": self.d_model,
            "num_read_heads": 20,
            "num_write_heads": 8,
            "num_memories": self.num_memories,
            "threshold": 0.99,
        }

    def test_successful_decoder_creation(self):
        """
        Test that a RecurrentDecoder is successfully created with standard parameters.
        """
        decoder = build_recurrent_decoder_v1(
            d_embedding=self.d_embedding,
            d_core=self.d_model,
            bank_size=self.bank_size,
            primary_dropout=self.direct_dropout,
            core_dropout=self.submodule_dropout,
            control_dropout=self.control_dropout,
            dtype=self.dtype,
            device=self.device,
            stack_depth=self.stack_depth,
            chunk_size=self.chunk_size,

            deep_memory_variant=self.deep_memory_variant,
            deep_memory_details= {"d_address" : self.d_model//8,
                                  "d_memory": self.d_model,
                                  "num_read_heads" : 10,
                                  "num_write_heads" : 10,
                                  "num_memories" : self.num_memories
                                  },

            layer_controller_variant="LinearBankSelector",
            layer_controller_details={}
        )


        def module_numel(module: nn.Module) -> int:
            return sum(param.numel() for param in module.parameters())

        print("num elements: %s" % module_numel(decoder))


class TestRecurrentDecoder(unittest.TestCase):
    def setUp(self):
        # Basic parameters for constructing a RecurrentDecoder
        self.d_embedding = 1024
        self.d_model = 128
        self.bank_size = 100
        self.direct_dropout = 0.1
        self.submodule_dropout = 0.05
        self.control_dropout = 0.1
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.stack_depth = 8
        self.chunk_size = 1

        # Variants for the registry submodules
        self.deep_memory_variant = "FastLinearMemory"
        self.act_variant = "Default"
        self.support_stack_variant = "Default"
        self.virtual_layer_controller_variant = "LinearBankSelector"

        # Deep memory details
        self.num_memories = 100

        # Extra parameters in case they’re required for certain variants
        self.variant_context = {
            "d_address": self.d_model // 16,
            "d_memory": self.d_model,
            "num_read_heads": 16,
            "num_write_heads": 16,
            "num_memories": self.num_memories,
            "threshold": 0.99,
        }

    def test_decode_step(self):
        """
        Test a decode step with a randomized input embedding.
        :return:
        """
        batch_size = 10
        num_items = 4
        embeddings = torch.randn([batch_size, num_items, self.d_embedding], requires_grad=True)
        decoder = build_recurrent_decoder_v1(
            d_embedding=self.d_embedding,
            d_core=self.d_model,
            bank_size=self.bank_size,
            primary_dropout=self.direct_dropout,
            core_dropout=self.submodule_dropout,
            control_dropout=self.control_dropout,
            dtype=self.dtype,
            device=self.device,
            stack_depth=self.stack_depth,
            chunk_size=self.chunk_size,
            dense_mode=True,

            deep_memory_variant=self.deep_memory_variant,
            deep_memory_details= {"d_address" : self.d_model//8,
                                  "d_memory": self.d_model,
                                  "num_read_heads" : 10,
                                  "num_write_heads" : 10,
                                  "num_memories" : self.num_memories
                                  },

            layer_controller_variant="LinearBankSelector",
            layer_controller_details={"top_k" : 3}
        )
        # First step creates state
        output, state, statistics = decoder(embeddings, None)

        print("begin backwards pass")
        output.sum().backward()

        # Print out the statistics
        print(statistics)

        # Track down num used parameters
        num_parameters = 0
        for parameter in decoder.parameters():
            num_parameters += parameter.numel()
        print("num parameters %s" % num_parameters)

        # Track down num memory elements
        accumulator = []
        def store(tensor: torch.Tensor):
            accumulator.append(tensor.numel())
            return tensor
        parallel_pytree_map(store, state)
        print("num_mem_elements %s" % sum(accumulator))
class TestRecurrentDecoderWithProfiling(unittest.TestCase):
    def setUp(self):
        # Basic parameters for constructing a RecurrentDecoder
        self.d_embedding = 1024
        self.d_model = 128
        self.bank_size = 100
        self.direct_dropout = 0.1
        self.submodule_dropout = 0.05
        self.control_dropout = 0.1
        self.device = torch.device("cpu")  # Use "cuda" if profiling on GPU
        self.dtype = torch.float32
        self.stack_depth = 8
        self.chunk_size = 512

        # Variants for the registry submodules
        self.deep_memory_variant = "FastLinearMemory"
        self.act_variant = "Default"
        self.support_stack_variant = "Default"
        self.virtual_layer_controller_variant = "LinearBankSelector"

        # Deep memory details
        self.num_memories = 100

        # Extra parameters in case they’re required for certain variants
        self.variant_context = {
            "d_address": self.d_model // 16,
            "d_memory": self.d_model,
            "num_read_heads": 16,
            "num_write_heads": 16,
            "num_memories": self.num_memories,
            "threshold": 0.99,
        }

    def test_decode_step_with_layer_profiling(self):
        """
        Profile a decode step with a randomized input embedding, recording time for each layer.
        """
        batch_size = 10
        num_items = 4
        embeddings = torch.randn([batch_size, num_items, self.d_embedding])

        # Build the decoder
        decoder = build_recurrent_decoder_v1(
            d_embedding=self.d_embedding,
            d_core=self.d_model,
            bank_size=self.bank_size,
            primary_dropout=self.direct_dropout,
            core_dropout=self.submodule_dropout,
            control_dropout=self.control_dropout,
            dtype=self.dtype,
            device=self.device,
            stack_depth=self.stack_depth,
            chunk_size=self.chunk_size,
            deep_memory_variant=self.deep_memory_variant,
            deep_memory_details={
                "d_address": self.d_model // 8,
                "d_memory": self.d_model,
                "num_read_heads": 10,
                "num_write_heads": 10,
                "num_memories": self.num_memories,
            },
            layer_controller_variant="LinearBankSelector",
            layer_controller_details={}
        )

        # Function to patch each layer with record_function timing
        def patch_forward_with_record_function(layer, layer_name):
            original_forward = layer.step

            def timed_forward(*args, **kwargs):
                with record_function(f"{layer_name}.forward"):
                    return original_forward(*args, **kwargs)

            layer.step = timed_forward

        # Patch each layer with record_function for profiling
        patched_layers = {}
        for name, layer in decoder.named_modules():
            patched_layers[name] = layer
            patch_forward_with_record_function(layer, name)

        # Profile the forward pass of the decoder
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
            output, state, statistics = decoder(embeddings, None)
            output, state, statistics = decoder(embeddings, state)  # Second step with initialized state

        # Unpatch layers to restore original forward methods
        for name, layer in patched_layers.items():
            layer.step = layer.__class__.step

        # Print profiling results with layer-specific timings
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

        # Additional statistics printout
        print(statistics)

        # Track down num used parameters
        num_parameters = sum(param.numel() for param in decoder.parameters())
        print("Number of parameters:", num_parameters)

        # Track down num memory elements
        accumulator = []
        def store(tensor: torch.Tensor):
            accumulator.append(tensor.numel())
            return tensor
        parallel_pytree_map(store, state)
        print("Number of memory elements:", sum(accumulator))

class TestRecurrentDecoderWithProfilingCUDA(unittest.TestCase):
    def setUp(self):
        # Set device to CUDA for GPU profiling
        self.d_embedding = 1024
        self.d_model = 128
        self.bank_size = 30
        self.direct_dropout = 0.0
        self.submodule_dropout = 0.0
        self.control_dropout = 0.0
        self.device = torch.device("cuda")  # Use "cuda" for GPU profiling
        self.dtype = torch.float32
        self.stack_depth = 8
        self.chunk_size = 2

        # Variants for the registry submodules
        self.deep_memory_variant = "FastLinearMemory"
        self.act_variant = "Default"
        self.support_stack_variant = "Default"
        self.virtual_layer_controller_variant = "LinearBankSelector"

        # Deep memory details
        self.num_memories = 100

        # Extra parameters for variant configurations
        self.variant_context = {
            "d_address": self.d_model // 16,
            "d_memory": self.d_model,
            "num_read_heads": 16,
            "num_write_heads": 16,
            "num_memories": self.num_memories,
            "threshold": 0.99,
        }

    def test_decode_step_with_layer_profiling(self):
        """
        Profile a decode step with a randomized input embedding, recording time for each layer.
        """
        batch_size = 10
        num_items = 5000
        embeddings = torch.randn([batch_size, num_items, self.d_embedding], device=self.device)

        # Build the decoder
        decoder = build_recurrent_decoder_v1(
            d_embedding=self.d_embedding,
            d_core=self.d_model,
            bank_size=self.bank_size,
            primary_dropout=self.direct_dropout,
            core_dropout=self.submodule_dropout,
            control_dropout=self.control_dropout,
            dtype=self.dtype,
            device=self.device,
            stack_depth=self.stack_depth,
            chunk_size=self.chunk_size,
            deep_memory_variant=self.deep_memory_variant,
            deep_memory_details={
                "d_address": self.d_model // 8,
                "d_memory": self.d_model,
                "num_read_heads": 10,
                "num_write_heads": 10,
                "num_memories": self.num_memories,
            },
            layer_controller_variant="LinearBankSelector",
            layer_controller_details={}
        )

        # Function to patch each layer with record_function timing
        def patch_forward_with_record_function(layer, layer_name):
            original_forward = layer.step

            def timed_forward(*args, **kwargs):
                with torch.autograd.profiler.record_function(f"{layer_name}.forward"):
                    return original_forward(*args, **kwargs)

            layer.step = timed_forward

        # Patch each layer for profiling
        patched_layers = {}
        for name, layer in decoder.named_modules():
            patched_layers[name] = layer
            patch_forward_with_record_function(layer, name)

        # Profile the forward pass of the decoder
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                    record_shapes=True, profile_memory=True) as prof:
            output, state, statistics = decoder(embeddings, None)
            output, state, statistics = decoder(embeddings, state)  # Second step with initialized state

        # Unpatch layers to restore original forward methods
        for name, layer in patched_layers.items():
            layer.step = layer.__class__.step

        # Print profiling results with layer-specific timings
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))

        # Additional statistics printout
        # Track down num used parameters
        num_parameters = sum(param.numel() for param in decoder.parameters())
        print("Number of parameters:", num_parameters)

        # Track down num memory elements
        accumulator = []
        def store(tensor: torch.Tensor):
            accumulator.append(tensor.numel())
            return tensor
        parallel_pytree_map(store, state)
        print("Number of memory elements:", sum(accumulator))
    def test_decode_step_with_layer_profiling_sparse(self):
        """
        Profile a decode step with a randomized input embedding, recording time for each layer.
        """
        batch_size = 10
        num_items = 30
        embeddings = torch.randn([batch_size, num_items, self.d_embedding],
                                 requires_grad=True,
                                 device=self.device)

        # Build the decoder
        decoder = build_recurrent_decoder_v1(
            d_embedding=self.d_embedding,
            d_core=self.d_model,
            bank_size=self.bank_size,
            dense_mode=False,
            primary_dropout=self.direct_dropout,
            core_dropout=self.submodule_dropout,
            control_dropout=self.control_dropout,
            dtype=self.dtype,
            device=self.device,
            stack_depth=self.stack_depth,
            chunk_size=self.chunk_size,
            deep_memory_variant=self.deep_memory_variant,
            deep_memory_details={
                "d_address": self.d_model // 8,
                "d_memory": self.d_model,
                "num_read_heads": 10,
                "num_write_heads": 10,
                "num_memories": self.num_memories,
            },
            layer_controller_variant="LinearBankSelector",
            layer_controller_details={"top_k" : 3}
        )

        # Profile the forward pass of the decoder
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                    record_shapes=True, profile_memory=True) as prof:
            output, state, statistics = decoder(embeddings, None)
            print("beginning backward pass")
            output.sum().backward()
            print("tests done")


        # Print profiling results with layer-specific timings
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
        print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))

        # Additional statistics printout

        # Track down num used parameters
        num_parameters = sum(param.numel() for param in decoder.parameters())
        print("Number of parameters:", num_parameters)

        # Track down num memory elements
        accumulator = []
        def store(tensor: torch.Tensor):
            accumulator.append(tensor.numel())
            return tensor
        parallel_pytree_map(store, state)
        print("Number of memory elements:", sum(accumulator))
        print(statistics)