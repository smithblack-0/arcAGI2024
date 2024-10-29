import unittest
import torch
from torch import nn

from src.main.model import build_recurrent_decoder_v1, DeepRecurrentDecoderV1


class TestRecurrentDecoderBuilder(unittest.TestCase):
    """
    Unit tests for the build_recurrent_decoder_v1 builder function.
    """

    def setUp(self):
        # Basic parameters for constructing a RecurrentDecoder
        self.d_embedding = 1024
        self.d_model = 128
        self.bank_size = 40
        self.direct_dropout = 0.1
        self.submodule_dropout = 0.05
        self.control_dropout = 0.1
        self.device = torch.device("cpu")
        self.dtype = torch.float32

        # Variants for the registry submodules
        self.deep_memory_variant = "LinearKernelMemoryBank"
        self.act_variant = "Default"
        self.support_stack_variant = "Default"
        self.virtual_layer_controller_variant = "LinearBankSelector"

        # Deep memory details
        self.mem_d_value = self.d_model
        self.num_memories = 100

        # Extra parameters in case they’re required for certain variants
        self.variant_context = {
            "mem_d_value": self.mem_d_value,  # Example placeholder if needed by a variant
            "num_memories" : self.num_memories,
            "threshold": 0.99,
        }

    def test_successful_decoder_creation(self):
        """
        Test that a RecurrentDecoder is successfully created with standard parameters.
        """
        decoder = build_recurrent_decoder_v1(
            d_embedding=self.d_embedding,
            d_model=self.d_model,
            bank_size=self.bank_size,
            direct_dropout=self.direct_dropout,
            submodule_dropout=self.submodule_dropout,
            control_dropout=self.control_dropout,
            dtype=self.dtype,
            device=self.device,
            deep_memory_variant=self.deep_memory_variant,
            act_variant=self.act_variant,
            support_stack_variant=self.support_stack_variant,
            virtual_layer_controller_variant=self.virtual_layer_controller_variant,
            **self.variant_context
        )

        # Validate instance type and check registry
        self.assertIsInstance(decoder, DeepRecurrentDecoderV1)

        def module_numel(module: nn.Module) -> int:
            return sum(param.numel() for param in module.parameters())
        print("num elements: %s" % module_numel(decoder))

    def test_missing_variant_context(self):
        """
        Test that the builder raises an appropriate error if a required parameter in variant_context is missing.
        """
        incomplete_context = self.variant_context.copy()
        incomplete_context.pop("extra_memory_param", None)  # Remove a parameter to simulate missing context

        with self.assertRaises(RuntimeError) as context:
            build_recurrent_decoder_v1(
                d_embedding=self.d_embedding,
                d_model=self.d_model,
                bank_size=self.bank_size,
                direct_dropout=self.direct_dropout,
                submodule_dropout=self.submodule_dropout,
                control_dropout=self.control_dropout,
                dtype=self.dtype,
                device=self.device,
                deep_memory_variant="SomeVariantNeedingExtraParam",
                act_variant=self.act_variant,
                support_stack_variant=self.support_stack_variant,
                virtual_layer_controller_variant=self.virtual_layer_controller_variant,
                **incomplete_context
            )

        self.assertIn("provide it in **variant_context", str(context.exception))


    def test_missing_required_parameters(self):
        """
        Test that the builder raises a ValueError if essential parameters are missing.
        """
        with self.assertRaises(TypeError) as context:
            build_recurrent_decoder_v1(
                d_embedding=None,  # Missing required d_embedding parameter
                d_model=self.d_model,
                bank_size=self.bank_size,
                direct_dropout=self.direct_dropout,
                submodule_dropout=self.submodule_dropout,
                control_dropout=self.control_dropout,
                dtype=self.dtype,
                device=self.device,
                deep_memory_variant=self.deep_memory_variant,
                act_variant=self.act_variant,
                support_stack_variant=self.support_stack_variant,
                virtual_layer_controller_variant=self.virtual_layer_controller_variant,
                **self.variant_context
            )

    def test_registry_failure_handling(self):
        """
        Test that the builder raises a RuntimeError with appropriate messaging if a registry component fails to build.
        """
        with self.assertRaises(RuntimeError) as context:
            build_recurrent_decoder_v1(
                d_embedding=self.d_embedding,
                d_model=self.d_model,
                bank_size=self.bank_size,
                direct_dropout=self.direct_dropout,
                submodule_dropout=self.submodule_dropout,
                control_dropout=self.control_dropout,
                dtype=self.dtype,
                device=self.device,
                deep_memory_variant="NonExistentMemoryVariant",  # Intentionally incorrect variant name
                act_variant=self.act_variant,
                support_stack_variant=self.support_stack_variant,
                virtual_layer_controller_variant=self.virtual_layer_controller_variant,
                **self.variant_context
            )

        self.assertIn("Failed to build the deep memory unit", str(context.exception))
