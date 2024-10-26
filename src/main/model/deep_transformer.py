"""
The actual decoder implementation is specified here.
"""
import torch
import virtual_layers
import registry
from torch import nn

# Registry imports
from typing import Tuple, Any, Optional
from transformer_primitives import (stack_registry, deep_memory_registry,
                                    DeepMemoryUnit, AbstractComputationStack, MemState)
from virtual_layers import VirtualFeedforward, AbstractBankSelector

class RecurrentDecoder(nn.Module):
    """
    The recurrent decoder mechanism. Sets up and
    manages recurrent state in order to decode
    next sequence prediction embeddings.
    """

    def __init__(self,
                 d_model: int,
                 stack_depth: int,
                 memory_unit: DeepMemoryUnit,
                 feedforward: VirtualFeedforward,
                 layer_selector: AbstractBankSelector
                 ):
        super().__init__()

        self.d_model = d_model
        self.stack_depth = stack_depth

        # Create ACT controller.

        # Store layers
        self.memory_unit = memory_unit
        self.feedforward = feedforward
        self.layer_selector = layer_selector

        # Create layernorm features
        self.memory_layernorm = nn.LayerNorm(d_model)
        self.feedforward_layernorm = nn.LayerNorm(d_model)
        self.stack_layernorm = nn.LayerNorm(d_model)

    def forward(self,
                embedding: torch.Tensor,
                recurrent_state: Optional[Any] = None
                ) -> Tuple[torch.Tensor, Any]:
        """
        Forward implementation of the recurrent decoder.

        :param embedding: The embedding to process. Must be recurrently fed. Shape (..., d_model)
        :param recurrent_state: The previous recurrent state, if available.
        :return:
            - The response embedding. Shape (..., d_model)
            - The recurrent state.
        """

        # Build the stack.

        stack = stack_registry.build("PointerSuperpositionStack",
                                     stack_depth=self.stack_depth,
                                     dtype=embedding.dtype,
                                     device=embedding.device,
                                     embedding_shape=embedding.shape
                                     )

        # Setup for adaptive computation time.
        accumulator = torch.zeros_like(embedding)
