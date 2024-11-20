from typing import Tuple, List, Callable, Optional, Any, Dict

import torch
from torch import nn
from dataclasses import dataclass, Field
from .memory import make_memory_unit, AbstractMemoryConfig, MemoryState
from .base import DeviceDtypeWatch, SavableConfig, load_activation_from_torch

MemoryCollection = List[MemoryState]
@dataclass
class FeedforwardConfig(SavableConfig):
    """
    Configuration class for the feedforward mechanism.
    A fairly standard feedforward mechanism is utilized
    and the parameters we must define are as follows

    d_hidden: How wide the hidden state is
    num_internal_layers: The number of internal hidden layers in the feedforward unit.
                          - For standard feedforward, set to 0
    feedforward_dropout: dropout probability IN the hidden space.
                         Quite useful, but needs to be conservative,
                         as these layers are how the model makes decisions.
    activation_module: Activation function to use. Should reference something in torch.nn.
                       Custom activations are not supported at the moment due to issues with
                       saving. Default is ReLU
    activation_kwargs: Any kwargs to hand into the activation module when initializing.
    """
    d_hidden: int
    num_internal_layers: int
    feedforward_dropout: float
    activation_module: str = "ReLU"
    activation_kwargs: Optional[Dict[str, Any]] = None

@dataclass
class RecurrentDecoderLayerConfig(SavableConfig):
    """
    The savable config mechanism for the recurrent
    decoder layer. Specifies, principally, the
    feedforward and memory mechanism.

    d_bottleneck: The bottleneck width. Should typically be much less than the embedding width.
    bottlenecked_dropout_rate: The dropout rate within the bottleneck.
    feedforward_config: The configuration for how to make the feedforward mechanism. See it for details
    memory_config: The config for how to make the memory mechanism. See the various options in memory for details.
    """
    d_bottleneck: int
    bottlenecked_dropout_rate: float
    feedforward_config: FeedforwardConfig
    memory_config: AbstractMemoryConfig


@dataclass
class RecurrentDecoderConfig(SavableConfig):
    """
    The config object for the recurrent decoder.

    You need to either specify a number of layers
    and a layer config, in which case all layers will
    be setup in exactly the same way, or provide a list
    of layer configs, in which case each will from the
    start be initialized into a single transformer stack.

    num_layers: The number of layers to create
    main_dropout_rate: The primary training dropout rate. Operates at unbottlenecked level
    layer_config: One of the init options. Provide a decoder layer config, and it will be repeated over all
                  num_layers. One of layer_config or layers_config must be defined
    layers_config: One of the init options. You provide a list containing a config for each tranformer
                   unit individually. Your list must match num layers in length.
                   One of layer_config or layers_config must be defined
    """

    num_layers: int
    main_dropout_rate: float
    layer_config: Optional[RecurrentDecoderLayerConfig] = None
    layer_configs: Optional[List[RecurrentDecoderLayerConfig]] = None

    # We have to jump through a few additional hoops
    # to ensure we can serialize and deserialize
    # the lists, since support is not native at the moment.
    def _serialize_data(self, item: Any) -> Any:
        if isinstance(item, list):
            item = [config.serialize() for config in item]
        return item

    def _deserialize_data(self, item: Any) -> Any:
        if isinstance(item, list):
            item = [self.deserialize(config) for config in item]
        return item


class FeedForward(nn.Module):
    """
    A feedforward implementation. Fairly
    flexible. See the config for more
    details.
    """

    def __init__(self,
                 d_model: int,
                 device: torch.device,
                 dtype: torch.dtype,
                 config: FeedforwardConfig,
                 ):
        super().__init__()
        self.d_hidden = config.d_hidden
        self.d_model = d_model

        self.activation = load_activation_from_torch(config.activation_module, config.activation_kwargs)
        self.ff_intake = nn.Linear(d_model, config.d_hidden, device=device, dtype=dtype)
        self.ff_output = nn.Linear(config.d_hidden, d_model, device=device, dtype=dtype)
        self.ff_internal = nn.ModuleList(nn.Linear(config.d_hidden, config.d_hidden, device=device, dtype=dtype)
                                         for _ in range(config.num_internal_layers))
        self.dropout = nn.Dropout(config.feedforward_dropout)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Performs the actual feedforward computation.
        :param tensor: The tensor to perform feedforward with. Shape (..., d_model)
        :return: Another tensor of shape (..., d_model)
        """

        # Perform the intake step, including dropout and activation
        tensor = self.ff_intake(tensor)
        tensor = self.activation(self.dropout(tensor))

        # Run all the internal hidden states
        # Dropout occurs during this.
        for layer in self.ff_internal:
            tensor = layer(tensor)
            tensor = self.activation(self.dropout(tensor))

        # Run the final step. Do not activate
        tensor = self.ff_output(tensor)
        return tensor


class RecurrentDecoderLayer(nn.Module):
    """
    A recurrent transformer decoder layer, capable
    of operating in both forward and reverse modes.

    In forward mode, the memory is advanced. In
    reverse mode, meanwhile, we first reverse then
    advance the memory to build a graph.

    Designed to be entirely compatible with torchscript,
    the core logic will bottleneck us down to a reduced
    computation space, perform a memory access and
    feedforward, then return to the original computation
    width.

    Layernorm and feedforward works all throughout,
    and dropout keeps things stable.
    """

    def __init__(self,
                 d_model: int,
                 dtype: torch.dtype,
                 device: torch.device,
                 config: RecurrentDecoderLayerConfig,
                 ):
        """
        :param d_model: Size of information traveling into this layer.
        :param device: The device we are on
        :param dtype: The dtype we are on
        :param config: The recurrent decoder layer config. You should see it for details
        """
        super().__init__()

        # Define the bottleneck and unbottleneck layer.

        self.bottleneck_projector = nn.Linear(d_model, config.d_bottleneck, device=device, dtype=dtype)
        self.unbottleneck_projector = nn.Linear(config.d_bottleneck, d_model, device=device, dtype=dtype)

        # Define the memory and feedforwrd layer.
        #
        # Along with the layernorms, and the dropout
        #
        # This all operates at the bottlenecked width

        self.memory_unit = make_memory_unit(config.d_bottleneck, dtype, device, config.memory_config)
        self.feedforward = FeedForward(config.d_bottleneck, device, dtype, config.feedforward_config)

        self.ff_layernorm = nn.LayerNorm(config.d_bottleneck, device=device, dtype=dtype)
        self.memory_layernorm = nn.LayerNorm(config.d_bottleneck, device=device, dtype=dtype)

        self.bottleneck_dropout = nn.Dropout(config.bottlenecked_dropout_rate)

    def create_state(self, batch_shape: torch.Size) -> MemoryState:
        """
        Creates a blank memory state associated with the
        given batch size.
        :param batch_shape: The batch shape to match
        :return: The setup memory state
        """
        return self.deep_memories.create_state(batch_shape)

    def run_reverse(self,
                    tensor: torch.Tensor,
                    batch_mask: torch.Tensor,
                    next_memory: MemoryState
                    ) -> Tuple[Tuple[torch.Tensor, MemoryState], MemoryState]:
        """
        Runs the reverse process at the bottleneck width. The core logic
        is located here.

        :param tensor: The tensor input to use. Shape (..., d_bottleneck)
        :param batch_mask: Shape (...). Indicates whether memory can be updated. True means No.
        :param next_memory: The next memory state during the forward pass
        :return:
        -Tuple:
            - The original output, but with a graph on it.
            - The previous memory state .
        """

        # Perform the deep memory access pattern

        (update, next_memory), previous_memory = self.memory_unit.reverse(tensor, batch_mask, next_memory)
        tensor = self.memory_layernorm(tensor + self.bottleneck_dropout(update))

        # Perform the feedforward
        update = self.feedforward(tensor)
        tensor = self.ff_layernorm(tensor + self.bottleneck_dropout(update))

        # Return
        return (tensor, next_memory), previous_memory

    def reverse(self,
                tensor: torch.Tensor,
                batch_mask: torch.Tensor,
                next_memory: MemoryState
                ) -> Tuple[Tuple[torch.Tensor, MemoryState], MemoryState]:
        """
        Runs the reverse process.

        :param tensor: The tensor input to use. Shape (..., d_model)
        :param batch_mask: Shape (...). Indicates whether memory can be updated. True means No.
        :param next_memory: The next memory state during the forward pass
        :return:
        -Tuple:
            - The original output, but with a graph on it.
            - The previous memory state .
        """
        tensor = self.bottleneck_projector(tensor)
        (tensor, next_memory), previous_memory = self.run_reverse(tensor, batch_mask, next_memory)
        tensor = self.unbottleneck_projector(tensor)
        return (tensor, next_memory), previous_memory

    def run_forward(self,
                    tensor: torch.Tensor,
                    batch_mask: torch.Tensor,
                    previous_memory: MemoryState) -> Tuple[torch.Tensor, MemoryState]:
        """
        The core forward mechanism. Runs at the bottlneck width
        :param tensor: The input. Shape (..., d_bottleneck)
        :param batch_mask: Shape (...). Indicates whether memory can be updated. True means No.
        :param previous_memory: The current memory state.
        :return:
            - The output.
            - The next memory state.
        """
        # Perform the deep memory access pattern

        update, next_memory = self.memory_unit(tensor, batch_mask, previous_memory)
        tensor = self.memory_layernorm(tensor + self.bottleneck_dropout(update))

        # Perform the feedforward
        update = self.feedforward(tensor)
        tensor = self.ff_layernorm(tensor + self.bottleneck_dropout(update))

        # Return
        return tensor, next_memory

    def forward(self,
                tensor: torch.Tensor,
                batch_mask: torch.Tensor,
                previous_memory: MemoryState) -> Tuple[torch.Tensor, MemoryState]:
        """
        The forward mechanism. Performs the forward step.
        :param tensor: The input. Shape (..., d_bottleneck)
        :param batch_mask: Shape (...). Indicates whether memory can be updated. True means No.
        :param previous_memory: The current memory state.
        :return:
            - The output.
            - The next memory state.
        """
        tensor = self.bottleneck_projector(tensor)
        (tensor, next_memory) = self.run_forward(tensor, batch_mask, previous_memory)
        tensor = self.unbottleneck_projector(tensor)
        return (tensor, next_memory), previous_memory


class AddPlusLayernorm(nn.Module):
    """
    Fairly simple. Performs an add+layernormn with
    dropout when provided an injected layer
    """

    def __init__(self,
                 d_model: int,
                 main_dropout_rate: int,
                 dtype: torch.dtype,
                 device: torch.device,
                 decoder_layer: RecurrentDecoderLayer
                 ):
        super().__init__()
        self.dropout = nn.Dropout(main_dropout_rate)
        self.layer = decoder_layer
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self,
                tensor: torch.Tensor,
                batch_mask: torch.Tensor,
                previous_memory: MemoryState
                ) -> Tuple[torch.Tensor, MemoryState]:
        update, next_memory = self.layer(tensor, batch_mask, previous_memory)
        tensor = self.layernorm(tensor + self.dropout(update))
        return tensor, next_memory

    def reverse(self,
                tensor: torch.Tensor,
                batch_mask: torch.Tensor,
                next_memory: MemoryState
                ) -> Tuple[Tuple[torch.Tensor, MemoryState], MemoryState]:
        (update, next_memory), previous_memory = self.layer.reverse(tensor, batch_mask, next_memory)
        tensor = self.layernorm(tensor + self.dropout(update))
        return (tensor, next_memory), previous_memory


class RecurrentDecoder(nn.Module):
    """
    A recurrent decoder transformer mechanism, with
    forward and reverse modes.

    In forward mode it should be provided with an input embedding
    and the memory states from the previous layer. In
    reverse mode, it should be provided with an input embedding,
    and the memory states from the next layer. Either way, it
    computes a functioning graph.

    However, during the backwards pass it returns the prior
    memory states, rather than the next ones, and sets them
    up to accumulate gradients.
    """

    @property
    def device(self) -> torch.device:
        return self._metainfo.device

    @property
    def dtype(self) -> torch.dtype:
        return self._metainfo.dtype

    def __init__(self,
                 d_model: int,
                 dtype: torch.dtype,
                 device: torch.device,
                 config: RecurrentDecoderConfig
                 ):
        super().__init__()

        if config.layer_config is None and config.layer_configs is None:
            raise ValueError('Must provide either a layer config or a stack of layer configs')
        if config.layer_config is not None:
            configs = [config.layer_config] * config.num_layers
        else:
            assert config.num_layers == len(config.layer_configs), "Num layers and list of configs must be same length"
            configs = config.layer_configs

        layer_cores = (RecurrentDecoderLayer(d_model, dtype, device, config) for config in configs)
        self.d_model = d_model
        self._metainfo = DeviceDtypeWatch(device=device, dtype=dtype)
        self.decoder_layers = nn.ModuleList(AddPlusLayernorm(d_model, config.main_dropout_rate, dtype, device, core)
                                            for core in layer_cores)

    def create_state(self, batch_shape: torch.Size) -> List[MemoryState]:
        """
        Sets up the recurrent state bound
        to a particular batch shape

        :param batch_shape: The batch shape to match
        :return: A list of memory states. One for each layer.
        """
        states = []
        for decoder_layer in self.decoder_layers:
            states.append(decoder_layer.create_state(batch_shape))
        return states

    def reverse(self,
                embedding: torch.Tensor,
                batch_mask: torch.Tensor,
                next_memories: MemoryCollection
                ) -> Tuple[Tuple[torch.Tensor, MemoryCollection], MemoryCollection]:
        """
        Runs the reverse process. This means figuring out the
        previous memory states and setting them up for gradient
        accumulation. And of course returning the final output

        :param embedding: The input embedding. Whatever it might be
        :param batch_mask: Shape (...). Indicates whether memory can be updated. True means No.
        :param next_memories: The memories from the NEXT step
        :return:
        - Tuple:
            - The final embedding, ready for usage in logits.
            - The memory states for this timestep. It has a graph. We need to insert gradients here
        - The memory from the last timestep. Setup to accumulate gradients and continue the chain.
        """
        assert embedding.shape[-1] == self.d_model
        previous_memories = []
        graph_memories = []
        tensor = embedding
        for layer, memory in zip(self.decoder_layers, next_memories):
            (tensor, next_memory), previous_memory = layer.reverse(tensor, batch_mask, memory)
            graph_memories.append(next_memory)
            previous_memories.append(previous_memory)
        return (embedding, graph_memories), previous_memories

    def forward(self,
                embedding: torch.Tensor,
                batch_mask: torch.Tensor,
                previous_memories: MemoryState
                ) -> Tuple[torch.Tensor, MemoryState]:
        """
        The forward mechanism. Performs a forward pass through the model.
        This will usually occur without gradients.
        :param embedding: The embedding being processed
        :param batch_mask: Shape (...). Indicates whether memory can be updated. True means No.
        :param previous_memories: The memory states from the last timestep
        :return:
        - The output. Same whether forward or backwards
        - The memory states for the next timestep.
        """
        assert embedding.shape[-1] == self.d_model
        next_memories = []
        tensor = embedding
        for layer, memory in zip(self.decoder_layers, previous_memories):
            tensor, next_memory = layer(tensor, batch_mask, memory)
            next_memories.append(next_memory)
        return tensor, next_memories
