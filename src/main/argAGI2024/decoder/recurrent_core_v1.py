from .abstract import AbstractComputationalCore, CoreAdapter, Model
import textwrap

import torch
from torch import nn

# Registry imports
from typing import Tuple, Any, Optional, Dict
from abc import ABC, abstractmethod
from ..virtual_layers import VirtualFeedforward, AbstractBankSelector, selector_registry, VirtualLinear, \
    SelectionSpec
from ..computation_support_stack import (stack_controller_registry,
                                                      AbstractStackController,
                                                      AbstractSupportStack,)
from ..deep_memories import deep_memory_registry, DeepMemoryUnit, AbstractMemoryState
from ..adaptive_computation_time import act_controller_registry

class ComputationalCore(AbstractComputationalCore):
    """
    An implementation of the recurrent core,
    and the first version of it.

    We internally manage state consisting
    of a stack, a deep memory unit, and a
    virtual feedforward.
    """

    def __init__(self,
                 stack_depth: int,
                 stack_controller: AbstractStackController,
                 deep_memory_unit: DeepMemoryUnit,
                 feedforward: VirtualFeedforward,
                 core_dropout: float = 0.05
                 ):
        super().__init__(d_core=deep_memory_unit.d_model,
                         device=deep_memory_unit.device,
                         dtype=deep_memory_unit.dtype)

        # Store the layers
        self.stack_controller = stack_controller
        self.deep_memory_unit = deep_memory_unit
        self.feedforward = feedforward

        # Store the layernorms.
        self.stack_layernorm = nn.LayerNorm(self.d_core, device=self.device, dtype=self.dtype)
        self.memory_layernorm = nn.LayerNorm(self.d_core, device=self.device, dtype=self.dtype)
        self.ff_layernorm = nn.LayerNorm(self.d_core, device=self.device, dtype=self.dtype)

        # Store stack depth and core dropout
        self.stack_depth = stack_depth
        self.dropout = nn.Dropout(core_dropout)

    def create_state(self, batch_shape: torch.Size) ->Tuple[AbstractSupportStack, AbstractMemoryState]:

        template_core_embedding = torch.zeros([*batch_shape, self.d_core], dtype=self.dtype, device=self.device)
        stack_state = self.stack_controller.create_state(batch_shape,
                                                         self.stack_depth,
                                                         core_embedding=template_core_embedding
                                                         )
        memory_state = self.deep_memory_unit.create_state(batch_shape)
        return stack_state, memory_state

    def get_statistics(self, state: Tuple[AbstractSupportStack, AbstractMemoryState]) ->Dict[str, Any]:
        stack_state, memory_state = state
        statistics = {}
        statistics["stack"] = stack_state.get_statistics()
        statistics["memory"] = memory_state.get_statistics()
        return statistics

    def forward(self,
                core_embedding: torch.Tensor,
                layer_selection: SelectionSpec,
                recurrent_state: Tuple[AbstractSupportStack, AbstractMemoryState]
                ) -> Tuple[torch.Tensor, Tuple[AbstractSupportStack, AbstractMemoryState]]:
        """
        The forward mechanism for the computation core
        :param core_embedding: The embeddigns in the core dimensionality. (... d_core)
        :param layer_selection: The layer selection spec
        :param recurrent_state: The recurrent state
        :return:
        - The response embedding. Shape (..., d_core)
        - The new recurrent state.
        """
        # Unpack the state
        stack_state, memory_state = recurrent_state

        # Run the stack interaction.
        # state is updated indirectly, but does not
        # overwrite tensors on object.
        stack_embedding, = self.stack_controller(core_embedding, stack_state,
                                                 core_embedding=core_embedding).values()
        core_embedding = self.stack_layernorm(core_embedding+self.dropout(stack_embedding))

        # Run the memory interaction
        # state is updated indirectly, but
        # does not overwrite state on objects.
        memory_embedding = self.deep_memory_unit(core_embedding, layer_selection, memory_state)
        core_embedding = self.memory_layernorm(core_embedding + self.dropout(memory_embedding))

        # Run the feedforward interaction
        ff_embedding = self.feedforward(core_embedding, layer_selection)
        core_embedding = self.ff_layernorm(core_embedding + self.dropout(ff_embedding))
        return core_embedding, (stack_state, memory_state)

def build_recurrent_decoder_v1(
        d_embedding: int,
        d_core: int,
        bank_size: int,
        stack_depth: int,
        chunk_size: int,

        primary_dropout: float,
        core_dropout: float,
        control_dropout: float,

        # More specific details. Selects the variants
        deep_memory_variant: str,
        deep_memory_details: Dict[str, Any],

        layer_controller_variant: str,
        layer_controller_details: Dict[str, Any],

        # Default dtype, device
        dense_mode: bool = True,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device('cpu')
        )->Model:
    """
    Builds a recurrent decoder.

    Universal:
    :param d_embedding: The width of incoming embeddings
    :param d_core: The width of the argAGI2024 core. usually much less
    :param bank_size: How many virtual layers to use
    :param stack_depth: How deep the stack should be
    :param chunk_size: Size of the recurrent chunks. Influences how much memory
        is used. If you are running out of memory, lower it!
    :param primary_dropout: How aggressive the dropout should be during the main act loop
    :param core_dropout: Inside the core, how aggressive the dropout should be.
    :param control_dropout: When making decisions, how aggressive dropout should act

    Variants. Specific features are likely needed:
    :param deep_memory_variant: The deep memory variant to use
    :param deep_memory_details:
        - The additional parameters needed to support the variant.
        - Suggest just trying to build a varient, and seeing what happens
    :param layer_controller_variant: The virtual layer selection variant
    :param layer_controller_details: The additional details to use.

    Final
    :param dtype: The dtype
    :param device: The device
    :return: A setup argAGI2024
    """

    # Create the stack controller
    stack_controller = stack_controller_registry.build("Default",
                                                       d_model=d_core,
                                                       control_dropout=control_dropout,
                                                       device=device,
                                                       dtype=dtype,
                                                       )

    # Create the deep memory unit
    memory_unit = deep_memory_registry.build(deep_memory_variant,
                                             bank_size=bank_size,
                                             d_model=d_core,
                                             device=device,
                                             dtype=dtype,
                                             **deep_memory_details
                                             )

    # Create the feedforward unit
    feedforward = VirtualFeedforward(d_core, d_core*4, bank_size,core_dropout, device=device, dtype=dtype)

    # Create the computational core
    comp_core = ComputationalCore(stack_depth,
                                  stack_controller,
                                  memory_unit,
                                  feedforward,
                                  core_dropout
                                  )

    # Create the core adapter
    adapter = CoreAdapter.build_from_parameters(d_embedding,
                                                d_core,
                                                bank_size,
                                                device=device,
                                                dtype=dtype,
                                                control_dropout=control_dropout,
                                                selector_varient=layer_controller_variant,
                                                dense_mode=dense_mode,
                                                **layer_controller_details)

    # Create the act controller
    act_controller = act_controller_registry.build("Default",
                                                   d_model=d_embedding,
                                                   device=device,
                                                   dtype=dtype,
                                                   threshold=0.99,
                                                   )
    # Create the argAGI2024
    return Model(act_controller,
                 adapter,
                 comp_core,
                 primary_dropout,
                 chunk_size)

