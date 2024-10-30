"""
The recurrent decoder mechanism. This module is designed for decoding tasks that process input embeddings
in a recurrent, adaptive manner. It integrates dynamic controllers to manage memory, computation stacks,
and adaptive computation time, making it suitable for sequence-based predictions, such as next-token
predictions in language models.

The RecurrentDecoder operates by repeatedly updating its state and output embedding until a halting
criterion is reached. Each forward pass processes a given embedding along with recurrent states, managing
the flow of information across memory and stack components. Additionally, a deep memory process is used
to manage gradients over extremely long context lengths.
"""
import textwrap

import torch
from torch import nn

# Registry imports
from typing import Tuple, Any, Optional, Dict
from abc import ABC, abstractmethod
from .virtual_layers import VirtualFeedforward, AbstractBankSelector, selector_registry, VirtualLinear
from src.main.model.computation_support_stack import (stack_controller_registry,
                                                      AbstractStackController,
                                                      AbstractSupportStack)
from src.main.model.adaptive_computation_time import (act_controller_registry,
                                                      AbstractACT,
                                                      ACTController)
from src.main.model.deep_memories import deep_memory_registry, DeepMemoryUnit
from src.main.model.registry import InterfaceRegistry


class RecurrentDecoderInterface(nn.Module, ABC):
    """
    Defines the interface of the recurrent decoder.
    This expects features to presented in terms
    of (..batch_shape, d_embedding).
    It is worth discussing a little bit the assumptions
    going into this decoder.

    One thing of note is it is expected we are using
    virtual layers in an adaptive computation structure.
    layers in terms of parameter banks that are
    selected from makes sense.

    It is also expected that this is a recurrent process,
    which accepts an embedding and, optionally, another
    state then proceeds to predict the next embedding.

    The recurrent state is not firmly pinned down, but
    connected to the various implementations. Finally, it
    should be noted this is designed to be made using
    the registry design, allowing easily swappable
    components.
    """
    def __init__(self):
        super().__init__()
    @abstractmethod
    def forward(self,
                embedding: torch.Tensor,
                recurrent_state: Optional[Any] = None
                ) -> Tuple[torch.Tensor, Any, Dict[str, Any]]:
        """
        Forward contract for the recurrent decoder.

        :param embedding: The embedding to process. Must be recurrently fed. Shape (..., d_model)
        :param recurrent_state: The previous recurrent state, if available.
        :return:
            - The response embedding. Shape (..., d_model)
            - The recurrent state.
            - The statistics banks for various processes, usable for metrics or training.
        """

registry_indirection = {"act_controller" : act_controller_registry,
                        "stack_controller" : stack_controller_registry,
                        "virtual_layer_controller" : selector_registry,
                        "deep_memory_unit" : deep_memory_registry,
                        }
recurrent_decoder_registry = InterfaceRegistry[RecurrentDecoderInterface]("RecurrentDecoder",
                                                                          RecurrentDecoderInterface,
                                                                          **registry_indirection)


@recurrent_decoder_registry.register("DeepRecurrentDecoderV1")
class DeepRecurrentDecoderV1(RecurrentDecoderInterface):
    """
    The recurrent decoder mechanism. Sets up and
    manages recurrent state in order to decode
    next sequence prediction embeddings.

    It operates according to the virtual layers system,
    where layer banks can dynamically be swapped
    out while processing. The number of banks is given
    in bank size

    The decoder is intended to be initialized as something out
    of the builder.

    A bottleneck is used to ensure that the model can run with
    relatively few parameters and maintain broader compatibility.
    """

    def __init__(self,
                 # Some final parameters
                 d_embedding: int,
                 d_model: int,
                 bank_size: int,
                 direct_dropout: float,
                 submodule_dropout: float,
                 device: torch.device,
                 dtype: torch.dtype,

                 # Controllers, generally, but also the memory unit.
                 act_controller: ACTController,
                 stack_controller: AbstractStackController,
                 virtual_layer_controller: AbstractBankSelector,
                 deep_memory_unit: DeepMemoryUnit,
                 ):
        """
        :param d_embedding: The size of the incoming embeddings
        :param d_model: The bottlenecked model dimension. May be different than embeddings
        :param bank_size: Size of the virtual layer bank
        :param direct_dropout: Dropout applied directly to the computation
        :param submodule_dropout: Dropout applied in submodules.
        :param act_controller: The act controller to use
        :param stack_controller: The stack controller to use
        :param virtual_layer_controller: The virtual layer controller to use
        :param deep_memory_unit: the deep memory unit to use
        """
        super().__init__()

        # Store controllers
        self.act_controller = act_controller
        self.stack_controller = stack_controller
        self.virtual_layer_controller = virtual_layer_controller

        # Store computational layers
        self.deep_memory_unit = deep_memory_unit
        self.feedforward = VirtualFeedforward(d_model, d_model*2,
                                              bank_size, submodule_dropout,
                                              device=device,
                                              dtype=dtype)

        # Create embedding layernorm features and dropout

        self.memory_layernorm = nn.LayerNorm(d_model)
        self.feedforward_layernorm = nn.LayerNorm(d_model)
        self.stack_layernorm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(direct_dropout)

        # Create the bottleneck projector.
        self.bottlneck_projector = VirtualLinear(d_embedding, d_model, bank_size, device=device, dtype=dtype)
        self.unbottleneck_projector = VirtualLinear(d_model, d_embedding, bank_size, device=device, dtype=dtype)
        self.main_layernorm = nn.LayerNorm(d_model)

    def forward(self,
                embedding: torch.Tensor,
                recurrent_state: Optional[Dict[str, Any]] = None
                ) -> Tuple[torch.Tensor, Dict[str, Any], Dict[str, Any]]:
        """
        Forward implementation of the recurrent decoder.

        :param embedding: The embedding to process. Must be recurrently fed. Shape (..., d_model)
        :param recurrent_state: The previous recurrent state, if available.
        :return:
            - The response embedding. Shape (..., d_model)
            - The recurrent state.
            - The statistics banks for various processes, usable for metrics or training. These are
                - act_statistics: Whatever the act statistic were. Implementation dependent
                - memory_statistics: Whatever the memory statistics were. Implementation dependent
                - stack_statistics: Whatever the stack statistics were. Implementation dependent
        """

        # Handle setup if the recurrent state was not passed in.


        if recurrent_state is None:
            # Get the batch shape
            batch_shape = embedding.shape[:-1]
            bottleneck_embedding = torch.zeros([*embedding.shape[:-1], self.d_model],
                                               device=embedding.device, dtype=embedding.dtype)

            # Setup the important recurrent states. These are principally
            # the memory and stack states. Both are stored in the
            # bottlenecked state.

            memory_state, mem_access_state = self.deep_memory_unit.create_state(batch_shape)
            stack_state = self.stack_controller.create_state(batch_shape,
                                                             self.stack_depth,
                                                             embedding=bottleneck_embedding,
                                                             mem_access_state=mem_access_state)

            # Store
            recurrent_state = {"memory_state": memory_state,
                               "stack_state": stack_state,
                               }

        # Create act state, including the embedding accumulator
        # and the internal state accumulator.
        #
        # Also, create default selection state and bottleneck
        # state for the process to work with.



        act_state = self.act_controller.create_state(embedding.shape[:-1],
                                                     embedding=embedding,
                                                     recurrent_state=recurrent_state)
        selection_state = None  # This is temporary state that is implementation dependent.
        mem_access_state = recurrent_state["stack_state"].pop("mem_access_state") # Get the state to resume with.


        # Run core process.
        #
        # Repeat until done.
        while act_state.should_continue():
            # Configure virtual layer layout.
            # Then reduce the embedding down to its bottlenecked version
            # for fast computation
            layer_selection, selection_state = self.virtual_layer_controller(embedding, selection_state)
            bottleneck_embedding = self.bottlneck_projector(embedding, selection_state)

            # Get memory and stack state, then get results off
            # of stack
            memory_state, stack_state = recurrent_state.values()


            # Run deep memory access and update.
            # Remember, memory state is updated by side effect!
            mem_result, mem_access_state = self.deep_memory_unit(bottleneck_embedding,
                                                                 layer_selection,
                                                                 recurrent_state,
                                                                 mem_access_state)
            bottleneck_embedding = self.memory_layernorm(bottleneck_embedding + self.dropout(mem_result))


            # Run feedforward access
            ff_result = self.feedforward(bottleneck_embedding, layer_selection)
            bottleneck_embedding = self.feedforward_layernorm(bottleneck_embedding + self.dropout(ff_result))


            # Run computation stack update.
            # Remember that this updates stack_state by side effect!
            stack_outputs = self.stack_controller(bottleneck_embedding,
                                                 act_state.halted_batches,
                                                 stack_state,
                                                 embedding=bottleneck_embedding,
                                                 mem_access_state=mem_access_state
                                                 )
            stack_embedding, mem_access_state = stack_outputs.values()
            bottleneck_embedding = self.stack_layernorm(bottleneck_embedding + self.dropout(stack_embedding))

            # Unbottleneck the embedding and integrate the results

            embedding_update = self.unbottleneck_projector(bottleneck_embedding, selection_state)
            embedding = self.main_layernorm(embedding + self.dropout(embedding_update))

            # Run the act update. Since it was updated indirectly, I can
            # just hand recurrent state into the act controller. Along with
            # the embedding to accumulate, of course.
            self.act_controller(bottleneck_embedding,
                                act_state,
                                embedding=embedding,
                                recurrent_state=recurrent_state)

        # Act should be done. We get the results, and return them
        results = act_state.finalize()
        embedding, recurrent_state = results.values()

        # Setup statistics banks.
        statistics = {}
        statistics["act_statistics"] = act_state.get_statistics()
        statistics["memory_statistics"] = recurrent_state["memory_state"].get_statistics()
        statistics["stack_statistics"] = recurrent_state["stack_state"].get_statistics()

        # Return the result.
        return embedding, recurrent_state, statistics

def build_recurrent_decoder_v1(
            # Fairly universal parameters
            d_embedding: int,
            d_model: int,
            bank_size: int,


            direct_dropout: float,
            submodule_dropout: float,
            control_dropout: float,


            dtype: torch.dtype = torch.float32,
            device: torch.device = torch.device('cpu'),

            # More specific details. Selects the variation
            deep_memory_variant: str = "LinearKernelMemoryBank",
            act_variant: str = "Default",
            support_stack_variant: str = "Default",
            virtual_layer_controller_variant: str = "Linear",

            # Extra parameters may be required to support the
            # variation. If you are getting an error, you probably
            # need to provide the extra parameters here.

            **variant_context: Any
            )->DeepRecurrentDecoderV1:
    """
    Builds a deep recurrent decoder, the first variant. We basically need to
    specify a bunch of commonly used information, then variants on the various submodules
    that we will use. Depending on the exact submodule, you might also need to provide
    additional variant context.

    :param d_embedding: The size of the embeddings dimension for any embeddings we want to process
    :param d_model: The internal model width. This can be different the
    :param bank_size: The size of the virtual layer bank. This is where you get most of your parameters.

    Dropout variants. There are three variants. Direct dropout
    is intended to be dropout which is applied directly during computation,
    and can tend to be quite aggressive. Submodule dropout is dropout that
    is applied to subcomponents. Control dropout is applied when making
    decisions, and can force more exploration at higher levels.

    :param direct_dropout: Dropout applied directly during each computation step. This can be aggressive and is typically used for regularizing output embeddings.
    :param submodule_dropout: Dropout applied within submodules to reduce overfitting within the model’s inner layers.
    :param control_dropout: Dropout applied during decision-making processes, enhancing exploration within adaptive computation paths.


    dtype and device is fairly straightforward

    :param dtype: The torch dtype
    :param device: The torch device

    Now we get into implementation details. The variants allow the easy
    swapping of components to explore different architectures. You provide
    a string matching the name of the variant to use, and we automatically
    use it.

    Note that variants can have additional context requirements, and these requirements
    will have to be provided in variant context
    :param deep_memory_variant: The deep memory variant
    :param act_variant: The computation time variant
    :param support_stack_variant: The support stack variant
    :param virtual_layer_controller_variant: The virtual layer controller variant

    Finally, you have to provide any extra context here. Provide them as kwargs
    :param variant_context: The extra context

    :return: The setup recurrent decoder
    """

    # Create common context feature to interface
    # with builder.
    construction_context = {
        "d_embedding" : d_embedding,
        "d_model" : d_model,
        "bank_size" : bank_size,

        "direct_dropout" : direct_dropout,
        "submodule_dropout" : submodule_dropout,
        "control_dropout" : control_dropout,

        "dtype" : dtype,
        "device" : device,
    }
    construction_context.update(variant_context)


    # Go manually build the deep memory subunit.
    #
    # Provide a descriptive error message on failure.
    try:
        deep_memory_unit = deep_memory_registry.build(deep_memory_variant, **construction_context)
    except Exception as err:
        msg = f"""
        Failed to build the deep memory unit. If a parameter is indicated 
        as missing, provide it in **variant_context as a kwarg. Original:
        """
        msg = textwrap.dedent(msg)
        msg += str(err)
        raise RuntimeError(msg) from err

    # Go manually build the virtual layer controller.
    # Provide a descriptive error message on failure.
    try:
        virtual_layer_controller = selector_registry.build(virtual_layer_controller_variant, **construction_context)
    except Exception as err:
        msg = f"""
        Failed to build the virtual layer selector. If a parameter is indicated 
        as missing, provide it in **variant_context as a kwarg. Original:
        """
        msg = textwrap.dedent(msg)
        msg += str(err)
        raise RuntimeError(msg) from err

    # Go manually build the act controller
    try:
        act_controller = act_controller_registry.build(act_variant, **construction_context)
    except Exception as err:
        msg = f"""
        Failed to build the computation time controller. If a parameter is indicated 
        as missing, provide it in **variant_context as a kwarg. Original:
        """
        msg = textwrap.dedent(msg)
        msg += str(err)
        raise RuntimeError(msg) from err

    # Go manually build the stack controller
    try:
        stack_controller = stack_controller_registry.build(support_stack_variant, **construction_context)
    except Exception as err:
        msg = f"""
        Failed to build the stack controller. If a parameter is indicated 
        as missing, provide it in **variant_context as a kwarg. Original:
        """
        msg = textwrap.dedent(msg)
        msg += str(err)
        raise RuntimeError(msg) from err

    # Integrate the features
    construction_context["deep_memory_unit"] = deep_memory_unit
    construction_context["virtual_layer_controller"] = virtual_layer_controller
    construction_context["act_controller"] = act_controller
    construction_context["stack_controller"] = stack_controller

    # Build and return
    return recurrent_decoder_registry.build("DeepRecurrentDecoderV1", **construction_context)
