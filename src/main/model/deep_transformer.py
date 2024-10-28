"""
The actual decoder implementation is specified here.
"""
import torch
from torch import nn

# Registry imports
from typing import Tuple, Any, Optional
from transformer_primitives import (DeepMemoryUnit)
from virtual_layers import VirtualFeedforward, AbstractBankSelector
from src.main.model.computation_support_stack.pointer_superposition_stack import StackFactory
from .adaptive_computation_time import AdaptiveComputationTime
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
                 layer_selector: AbstractBankSelector,
                 stack_factory: StackFactory
                 ):
        super().__init__()

        self.d_model = d_model
        self.stack_depth = stack_depth

        # Store layers
        self.memory_unit = memory_unit
        self.feedforward = feedforward
        self.layer_selector = layer_selector
        self.stack_factory = stack_factory

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
        if recurrent_state is None:
            self.memory_unit.

        # Build the stack. Build the ACT
        stack = self.stack_factory(embedding, embedding)
        act = AdaptiveComputationTime(embedding.shape[:-1],
                                      dtype=embedding.dtype,
                                      device=embedding.device)

        while act.should_continue():


        # Setup for adaptive computation time.
        accumulator = torch.zeros_like(embedding)
