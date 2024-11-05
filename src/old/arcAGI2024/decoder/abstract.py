"""
The abstract descriptor for the decoder interface.

T
"""
import textwrap
import warnings
from typing import Any, Tuple, Optional, Dict, Callable, List
from abc import ABC, abstractmethod

import torch
from tokenizers import Tokenizer
from torch import nn
from torch.utils import checkpoint
from ..registry import InterfaceRegistry
from ..virtual_layers import AbstractBankSelector, VirtualLinear, SelectionSpec, selector_registry
from ..base import DeviceDtypeWatch, parallel_pytree_map, get_rng_state, set_rng_state
from ..adaptive_computation_time import act_controller_registry, ACTController

class AbstractComputationalCore(nn.Module, ABC):
    """
    The computational core of the decoder process,
    this processes information at the level of the
    bottleneck. It is an abstract interface that
    must be implemented.

    The computational core consists of a virtual
    layer bank obeying the interface specification,
    and manages a relevant internal state. It sits
    within an ACT loop, and will be invoked
    multiple times, hence the virtual layers.

    This contains a recurrent arcAGI2024, that is
    expected to
    """
    @property
    def device(self)->torch.device:
        return self.__metainfo.device
    @property
    def dtype(self)->torch.dtype:
        return self.__metainfo.dtype
    def __init__(self,
                 d_core: int,
                 device: torch.device,
                 dtype: torch.dtype,
                 ):
        super().__init__()
        self.d_core = d_core
        self.__metainfo = DeviceDtypeWatch(device=device, dtype=dtype)
    @abstractmethod
    def create_state(self, batch_shape: torch.Size)->Any:
        """
        Creates the required recurrent state when invoked
        :param batch_shape: The batch shape to bind ot
        :return: The created, and relevant, recurrent state.
        """

    @abstractmethod
    def get_statistics(self, state: Any)->Dict[str, Any]:
        """
        Gets a dictionary of statistics related to the recurrent
        state.
        :param state: The recurrent state
        :return: The statistics related to it
        """
    @abstractmethod
    def forward(self,
                core_embedding: torch.Tensor,
                layer_selection: SelectionSpec,
                recurrent_state: Any
                )->Tuple[torch.Tensor, Any]:
        """
        :param core_embedding: The core embedding to process. (..., d_core)
        :param layer_selection: The virtual layer configuration
        :return:
            - The response. Shape (..., d_core)
            - The new recurrent state
        """
class CoreAdapter:
    """
    The core adapter is responsible for getting
    us from embedding space into computational
    core space and back again. It also manages
    the bottleneck process.

    Generally, for reasons of computational efficiency,
    the core arcAGI2024 should have a considerably reduced
    computational width to the embedding size, and
    is convered between by a bottleneck process.
    """
    @property
    def device(self) -> torch.device:
        return self.layer_selector.device

    @property
    def dtype(self) -> torch.dtype:
        return self.layer_selector.dtype

    @classmethod
    def build_from_parameters(cls,
                            d_embedding: int,
                            d_core: int,
                            bank_size: int,
                            device: torch.device,
                            dtype: torch.dtype,
                            control_dropout: float,
                            selector_varient: str,
                            dense_mode: bool ,
                            **selector_details: Dict[str, Any]
                            )->'CoreAdapter':
        """
        Builds a working core adapter. This includes
        handling the different levels, and setting
        up the selector.

        :param d_embedding: The embedding dimension
        :param d_core: The core dimension
        :param bank_size: The bank size.
        :param control_dropout: The control dropout rate
        :param device: The device
        :param dtype: The dtype
        :param selector_varient: The version of abstract selector to use.
        :param selector_details: A dictionary consisting of selector details.
            - See selectors in virtual layers.
            - Can define at least:
                - [int] top_k
                - [float] top_p
                - [int] rand
            - Some implementations may require more.
        :return: A setup core adapter
        """

        bottleneck = VirtualLinear(d_embedding, d_core, bank_size, dtype=dtype, device=device)
        unbottleneck = VirtualLinear(d_core, d_embedding, bank_size, dtype=dtype, device=device)
        selector = selector_registry.build(selector_varient,
                                           d_model = d_embedding,
                                           control_dropout = control_dropout,
                                           bank_size = bank_size,
                                           dtype=dtype,
                                           device=device,
                                           dense_mode=dense_mode,
                                           **selector_details
                                           )
        return cls(bottleneck, unbottleneck, selector)

    def __init__(self,
                 bottleneck_projector: VirtualLinear,
                 unbottleneck_projector: VirtualLinear,
                 virtual_layer_selector: AbstractBankSelector
                 ):
        """
        Setup process for the adapter
        :param bottleneck_projector:
        - The bottleneck adapter converts us from embedding width to core width
        - It is a virtual layer and must satisfy bank size.
        :param unbottleneck_projector:
        - The unbottleneck projector reverses the bottleneck process to get back to the
          embedding width
        - It is a virtual layer and must satisfy bank size.
        :param virtual_layer_selector:
        - The virtual layer selector processes and produces selection specs for the
          virtual layer process. It operates at the unbottlenecked context so it
          sees the big picture.
        """
        if bottleneck_projector.in_features != unbottleneck_projector.out_features:
            msg = f"""
            Bottleneck and unbottleneck projectors do not interface with the same
            embedding width. 
            
            The bottleneck projector interfaced with an embedding width of: {bottleneck_projector.in_features}.
            However, the unbottleneck projector interfaced with: {unbottleneck_projector.out_features}.
            """
            msg = textwrap.dedent(msg)
            raise ValueError(msg)
        if bottleneck_projector.out_features != unbottleneck_projector.in_features:
            msg = f"""
            Bottleneck and unbottleneck projectors do not interface with the same d_core arcAGI2024
            width. 
            
            The bottleneck projector interfaced with a width of: {bottleneck_projector.out_features}.
            However, the unbottleneck projector interfaced with: {unbottleneck_projector.in_features}.
            """
            msg = textwrap.dedent(msg)
            raise ValueError(msg)
        if virtual_layer_selector.d_model != bottleneck_projector.in_features:
            msg = f"""
            Virtual layer projector was found to be incompatible with
            the detected d_embedding.
            
            d_embedding was detected as: {bottleneck_projector.in_features}.
            However, the selector wanted: {virtual_layer_selector.d_model}.
            """
            msg = textwrap.dedent(msg)
            raise ValueError(msg)

        assert virtual_layer_selector.dtype == bottleneck_projector.dtype
        assert virtual_layer_selector.device == bottleneck_projector.device
        assert virtual_layer_selector.dtype == unbottleneck_projector.dtype
        assert virtual_layer_selector.device == unbottleneck_projector.device


        # Store signals
        self.d_embedding = bottleneck_projector.in_features
        self.d_core = bottleneck_projector.out_features
        self.bank_size = virtual_layer_selector.bank_size

        # Store layers
        self.bottleneck_projector = bottleneck_projector
        self.unbottleneck_projector = unbottleneck_projector
        self.layer_selector = virtual_layer_selector

    def get_statistics(self, state: Any)->Dict[str, torch.Tensor]:
        """
        Gets a set of statistics based on the recurrent state related
        to how the arcAGI2024 is processing things
        :param state: The state to examine
        :return: The gathered statistics
        """
        return self.layer_selector.get_statistics(state)
    def create_state(self, batch_shape: torch.Size):
        """
        Sets up any required state for the virtual
        layer process
        :param batch_shape: The batch shape under consideration
        :return: The setup state feature.
        """
        return self.layer_selector.create_state(batch_shape)


    def select_virtual_layer(self,
                             embedding: torch.Tensor,
                             state: Any
                             )->Tuple[SelectionSpec, Any]:
        """
        Performs a virtual layer selection, presumably with
        :param embedding: The embedding to select with. (..., d_embedding)
        :param state: The recurrent state
        :return:
            - The selection spec
            - The recurrent state
        """
        return self.layer_selector(embedding, state)

    def bottleneck(self, embedding: torch.Tensor, selection: SelectionSpec)->torch.Tensor:
        """
        Reduces the embedding down to the arcAGI2024 core dimensions.
        :param embedding: The embedding to reduce with. (..., d_embedding)
        :param selection: The layer selection
        :return: The tensor in the d_core dimensionality. (..., d_core)
        """
        return self.bottleneck_projector(embedding, selection)

    def unbottleneck(self, embedding: torch.Tensor, selection: SelectionSpec)->torch.Tensor:
        """
        Returns the embedding from the d_core dim back to the embedding dim.
        :param embedding: The core embedding. Shape (..., d_core)
        :param selection: The selected virtual layer config
        :return: The restored embedding. Shape (..., d_embedding)
        """
        return self.unbottleneck_projector(embedding, selection)

class Model(nn.Module):
    """
    The recurrent portion of the arcAGI2024.
    """
    def create_state(self, batch_shape: torch.Size)->Tuple[Any, Any]:
        """
        Creates recurrent state when requested
        :param batch_shape: The batch shape to bind to
        :return: The setup recurrent state
        """
        adapter_state = self.core_adapter.create_state(batch_shape)
        core_state = self.computational_core.create_state(batch_shape)
        return adapter_state, core_state
    def __init__(self,
                 act_controller: ACTController,
                 core_adapter: CoreAdapter,
                 computational_core: AbstractComputationalCore,
                 primary_dropout: float = 0.2,
                 chunk_size: int = 512,
                 ):
        super().__init__()

        self.core_adapter = core_adapter
        self.computational_core = computational_core
        self.act_controller = act_controller

        self.primary_layernorm = nn.LayerNorm(core_adapter.d_embedding,
                                           device=core_adapter.device,
                                           dtype=core_adapter.dtype)
        self.primary_dropout = nn.Dropout(primary_dropout)
        self.chunk_size = chunk_size

    def recurrent_core(self,
                embedding: torch.Tensor,
                state: Tuple[Any, Any]
                )->Tuple[torch.Tensor, Any, Dict[str, torch.Tensor]]:
        """
        Core reccurrent step of the arcAGI2024. Accepts only recurrenly
        specified tensor collections. Will be invoked over and over again.
        :param embedding: The recurrent embedding to process. Shape (..., d_embedding)
        :param state: The recurrent state.
        :return:
        - The response. Shape (..., d_embedding)
        - The new recurrent state. Shape (..., d_embedding)
        - The act statistics.
        """
        # Unbind and setup the arcAGI2024 for processing. This includes
        # taking apart the state info, and setting up the act mechanism.
        batch_shape = embedding.shape[:-1]
        adapter_state, core_state = state
        act_state = self.act_controller.create_state(batch_shape,
                                                     embedding=embedding,
                                                     adapter_state=adapter_state,
                                                     core_state=core_state)

        # Perform the core computation process. We will
        # select the virtual layer configuration, bottleneck, compute
        # in the core, then unbottleneck until ready to move on.
        while act_state.should_continue():
            # Select the virtual layer configuration
            virtual_layer, adapter_state = self.core_adapter.select_virtual_layer(embedding, adapter_state)

            # Bottleneck, process, unbottleneck
            core_embedding = self.core_adapter.bottleneck(embedding, virtual_layer)
            core_embedding, core_state = self.computational_core(core_embedding, virtual_layer, core_state)
            embedding_update = self.core_adapter.unbottleneck(core_embedding, virtual_layer)

            # Integrate the update and apply dropout
            embedding = self.primary_layernorm(embedding + self.primary_dropout(embedding_update))

            # Act step
            act_state = self.act_controller(embedding,
                                            act_state,
                                            embedding=embedding,
                                            adapter_state=adapter_state,
                                            core_state=core_state
                                            )

        # Act is done, and the computation is presumably finished.
        # Return updates, and act statistics since they will vanish
        final_state = act_state.finalize()
        embedding, adapter_state, core_state = final_state.values()
        return embedding, (adapter_state, core_state), act_state.get_statistics()

    def run_chunk(self,
                  rng_state: Any,
                  chunk_num: int,
                  chunk: torch.Tensor,
                  state: Any,
                  )->Tuple[torch.Tensor, Any, Any]:
        """
        Runs the given chunk with the given state. Designed
        for checkpointing purposes
        :param rng_state: The random number state to initialize with
        :param chunk_num: Used for prettyprinting
        :param chunk: The chunk to run. Embeddings of shape (..., chunk_size, d_embedding)
        :param state: The state to run with.
        :return:
        - The response embeddings (..., chunk_size, d_embedding)
        - The response state.
        - The statistics for the chunk
        """
        # Run chunk
        print("running chunk %s" % chunk_num)
        set_rng_state(rng_state, chunk.device)
        response_embeddings = []
        act_accumulator = []
        for i, recurrent_embedding in enumerate(chunk.unbind(-2)):
            print("item in chunk: %s" % i )
            response, state, act_statistics = self.recurrent_core(recurrent_embedding, state)
            response_embeddings.append(response)
            act_accumulator.append(act_statistics)

        # Generate statistics for chunk

        def mean_of_tensors(*tensors):
            # Used to get the mean of any act statistics.
            tensor = torch.stack(tensors, dim=0)
            return tensor.mean(dim=0)

        selector_state, core_state = state
        chunk_statistics = {}
        chunk_statistics["core"] = self.computational_core.get_statistics(core_state)
        chunk_statistics["layer_select"] = self.core_adapter.get_statistics(selector_state)
        chunk_statistics["act_statistics"] = parallel_pytree_map(mean_of_tensors, *act_accumulator)

        # Return results
        return torch.stack(response_embeddings, dim=-2), state, chunk_statistics


    def forward(self,
                embeddings: torch.Tensor,
                state: Optional[Any] = None,
                )->Tuple[torch.Tensor, Any, Dict[str, Any]]:
        """
        Forward method for the arcAGI2024 core
        :param embeddings: The recurrent embedding to process. Shape (..., d_embedding)
        :param state: The recurrent state.
        :return:
        - The response. Shape (..., d_embedding)
        - The new recurrent state. Shape (..., d_embedding)
        - The process statistics, whatever they might be.
        """

        # Create the state, if needed.
        if state is None:
            batch_shape = embeddings.shape[:-2]
            state = self.create_state(batch_shape)

        # Run the recurrent mechanism with
        # chunking. This keeps the memory usage down, at the cost of doubling
        # the computation time.
        chunk_statistics = []
        embedding_responses = []
        chunks = torch.split(embeddings, self.chunk_size, dim=-2)
        for i, chunk in enumerate(chunks):
            embedding_response, state, statistics = checkpoint.checkpoint(self.run_chunk,
                                                                          get_rng_state(chunk.device),
                                                                          i,
                                                                          chunk,
                                                                          state,
                                                                          use_reentrant=False,
                                                                          preserve_rng_state=False,
                                                                          )
            chunk_statistics.append(statistics)
            embedding_responses.append(embedding_response)

        # Return the results
        response = torch.concat(embedding_responses, dim=-2)
        return response, state, chunk_statistics


