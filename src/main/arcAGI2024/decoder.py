from typing import Tuple, List

import torch
from torch import nn

from .deep_memory import MemoryState, FastLinearMemory


class Feedforward(nn.Module):
    """
    A classic feedforward implementation.

    d_model is the input and output dimension.
    d_hidden is how wide we get inside.
    """

    def __init__(self,
                 d_model: int,
                 d_hidden: int
                 ):
        super().__init__()
        self.d_hidden = d_hidden
        self.d_model = d_model

        self.ff1 = nn.Linear(d_model, d_hidden)
        self.ff2 = nn.Linear(d_hidden, d_model)
        self.relu = nn.ReLU()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = self.ff1(tensor)
        tensor = self.relu(tensor)
        tensor = self.ff2(tensor)
        return tensor


class DecoderLayer(nn.Module):
    """
    A deep transformer decoder layer

    Consists of the deep memory unit,
    and the feedforward block. Can
    operate in both forward and reverse
    modes.
    """

    def __init__(self,
                 d_model: int,
                 d_hidden: int,
                 d_address: int,
                 d_memory: int,
                 num_read_heads: int,
                 num_write_heads: int,
                 num_memories: int,
                 numeric_write_factor: float,
                 dropout: float,
                 device: torch.device,
                 dtype: torch.dtype
                 ):
        """
        :param d_model: The main model dimensions, as the embeddings come in as
        :param d_hidden: The dimensions of the feedforward process
        :param d_address: The dimensions used for memory addressing
        :param d_memory: The dimensions used for memory storage
        :param num_read_heads: The number of memory read heads
        :param num_write_heads: The numbery of memory write heads
        :param num_memories: The number of discrete memories
        :param dropout: The dropout probability
        :param device: The device we are on
        :param dtype: The dtype we are on
        """
        super().__init__()

        self.d_model = d_model
        self.d_hidden = d_hidden
        self.d_address = d_address
        self.d_memory = d_memory
        self.num_read_heads = num_read_heads
        self.num_write_heads = num_write_heads
        self.num_memories = num_memories
        self.dropout_rate = dropout

        # Define the deep memory layer

        self.deep_memories = FastLinearMemory(d_model,
                                              d_address,
                                              d_memory,
                                              num_read_heads,
                                              num_write_heads,
                                              num_memories,
                                              dropout,
                                              max_write_factor=numeric_write_factor,
                                              device=device,
                                              dtype=dtype)
        self.deep_layernorm = nn.LayerNorm(d_model)

        # Define the feedforward layer
        self.feedforward = Feedforward(d_model, d_hidden)
        self.ff_layernorm = nn.LayerNorm(d_model)

        # Define the dropout layer
        self.dropout = nn.Dropout(dropout)

    def create_state(self, batch_shape: torch.Size) -> MemoryState:
        """
        Creates a blank memory state associated with the
        given batch size.
        :param batch_shape: The batch shape to match
        :return: The setup memory state
        """
        return self.deep_memories.create_state(batch_shape)

    def reverse(self,
                tensor: torch.Tensor,
                batch_mask: torch.Tensor,
                next_memory: MemoryState
                ) -> Tuple[Tuple[torch.Tensor, MemoryState], MemoryState]:
        """
        The reverse mechanism. Able to perform the same computation,
        but with the notable complication of also returning the
        prior memory state, setup to accumulate gradients.

        :param tensor: The tensor input to use
        :param batch_mask: Shape (...). Indicates whether memory can be updated. True means No.
        :param next_memory: The next memory state during the forward pass
        :return:
        -Tuple:
            - The original output, but with a graph on it.
            - The previous memory state .
        """

        # Perform the deep memory access pattern
        (update, next_memory), previous_memory = self.deep_memories.reverse(tensor, batch_mask, next_memory)
        tensor = self.deep_layernorm(tensor + self.dropout(update))

        # Perform the feedforward
        update = self.feedforward(tensor)
        tensor = self.ff_layernorm(tensor + self.dropout(update))

        # Return
        return (tensor, next_memory), previous_memory

    def forward(self,
                tensor: torch.Tensor,
                batch_mask: torch.Tensor,
                previous_memory: MemoryState) -> Tuple[torch.Tensor, MemoryState]:
        """
        The forward mechanism. Performs a forward pass through the mode
        :param tensor: The input.
        :param batch_mask: Shape (...). Indicates whether memory can be updated. True means No.
        :param previous_memory: The current memory state.
        :return:
            - The output.
            - The next memory state.
        """
        # Perform the deep memory access pattern
        update, next_memory = self.deep_memories(tensor, batch_mask, previous_memory)
        tensor = self.deep_layernorm(tensor + self.dropout(update))

        # Perform the feedforward
        update = self.feedforward(tensor)
        tensor = self.ff_layernorm(tensor + self.dropout(update))

        # Return
        return tensor, next_memory


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

    Bottlenecks are built at the recurrent decoder level.
    This makes it quite easy to adjust and pretrain the model
    for different embeddings or different embedding widths.
    You can fetch the old decoder layers feature, and use
    it to initialize a new recurrent decoder.
    """

    def __init__(self,
                 d_model: int,
                 dropout_rate: float,
                 decoder_layers: List[DecoderLayer]
                 ):

        super().__init__()
        template = decoder_layers[0]
        self.d_model = d_model
        self.dropout_rate = dropout_rate

        # Setup layernorms, with a no op on the last for immediate consumption
        # by a logit system.
        layernorms = [nn.LayerNorm(d_model) for _ in range(len(decoder_layers))]
        layernorms.pop(-1)
        layernorms.append(nn.Identity())

        # Setup bottleneck, unbottleneck layers
        bottlenecks = [nn.Linear(d_model, template.d_model) for _ in range(len(decoder_layers))]
        unbottlenecks = [nn.Linear(template.d_model, d_model) for _ in range(len(decoder_layers))]

        # Setup layers and repositories
        self.layernorms = nn.ModuleList(layernorms)
        self.decoder_layers = nn.ModuleList(decoder_layers)
        self.dropout = nn.Dropout(dropout_rate)

        self.bottlenecks = nn.ModuleList(bottlenecks)
        self.unbottlenecks = nn.ModuleList(unbottlenecks)

    def rebuild_at_different_width(self, d_model: int) -> 'RecurrentDecoder':
        """
        Rebuilds an existing model to use a different d_model. Expect
        to do a moderate amount of fine tuning, but most of the core
        logic should still be useful.

        :param d_model: The model width to now set us up at
        :return: The new recurrent decoder
        """
        return RecurrentDecoder(d_model, self.dropout_rate, self.decoder_layers)

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
                next_memories: List[MemoryState]
                ) -> Tuple[Tuple[torch.Tensor, List[MemoryState]], List[MemoryState]]:
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
        gradprop_memories = []
        iterator = zip(self.layernorms, self.decoder_layers, self.bottlenecks, self.unbottlenecks, next_memories)
        for layernorm, decoder_layer, bottleneck, unbottleneck, next_memory in iterator:
            # Bottleneck, process, and unbottleneck
            bottlenecked_embedding = bottleneck(embedding)
            (bottlenecked_embedding, next_memory), previous_memory = decoder_layer.reverse(bottlenecked_embedding,
                                                                                              batch_mask,
                                                                                              next_memory)

            update = unbottleneck(bottlenecked_embedding)

            # Stash away memories
            gradprop_memories.append(next_memory)
            previous_memories.append(previous_memory)

            # Integrate update, and store memory.
            embedding = layernorm(embedding + self.dropout(update))
        return (embedding, gradprop_memories), previous_memories

    def forward(self,
                embedding: torch.Tensor,
                batch_mask: torch.Tensor,
                previous_memories: List[MemoryState]
                ) -> Tuple[torch.Tensor, List[MemoryState]]:
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
        iterator = zip(self.layernorms, self.decoder_layers, self.bottlenecks, self.unbottlenecks, previous_memories)
        for layernorm, decoder_layer, bottleneck, unbottleneck, previous_memory in iterator:
            # Bottleneck, process, then unbottleneck
            bottlenecked_embedding = bottleneck(embedding)
            bottlenecked_embedding, next_memory = decoder_layer(bottlenecked_embedding,
                                                                batch_mask,
                                                                previous_memory)
            update = unbottleneck(bottlenecked_embedding)

            # Integrate update, then append memory
            embedding = layernorm(embedding + self.dropout(update))
            next_memories.append(next_memory)
        return embedding, next_memories


def build_decoder(
        # Primary specifications
        d_model: int,
        num_layers: int,

        # Helper specifics
        d_core: int,
        d_hidden: int,
        d_address: int,
        d_memory: int,
        num_read_heads: int,
        num_write_heads: int,
        num_memories: int,
        dropout_rate: float,
        auxilary_dropout_rate: float,
        numeric_write_factor: float,

        # Dtype, device
        dtype: torch.dtype = None,
        device: torch.device = None
) -> RecurrentDecoder:
    """
    Creates a functioning recurrent decoder, with
    forward and reverse modes, ready for integration
    into a broader architecture.

    The returned model is designed to be entirely
    recurrent.

    --- Top level parameters ---
    :param d_model:
    - Dimensionality of the embeddings coming in, and being returned
    - Most of the computation does not occur at this width.
    :param num_layers:
    - Number of transformer layers.
    - Each is a DecoderLayer
    :param dropout_rate:
    - Dropout rate, within the primary recurrent model
    :param auxilary_dropout_rate:
    - Dropout rate, within the core decoder layers and computation models.

    --- DecoderLayer Parameters
    Everything here is occurring with respect to d_core.

    :param d_core:
    - Bottleneck rate. d_model is bottlenecked down to this dimension to save compute time
    - Much of the computation occurs at this rate
    :param d_hidden:
    - Size of the hidden layer.
    - Choose it with respect to d_core.
    :param d_address:
    - A smaller subset of d_core. Can be anything.
    - Represents the width of the attn memory addresses.
    :param d_memory:
    - Represents the width of the memory value addresses.
    - Can be different.
    :param num_read_heads:
    - Number of heads used for the memory read process
    - Larger means more of the memory can be read from per step
    :param num_write_heads:
    - Number of heads used for the memory write process
    - Larger means more of the memory can be written to per step
    :param num_memories:
    - Number of independent memory states

    ---- final ---
    :param dtype: The dtype
    :param device: The device
    :return: A setup RecurrentDecoder
    """

    # Create the decoder layer stack
    layers = [
        DecoderLayer(d_core,
                     d_hidden,
                     d_address,
                     d_memory,
                     num_read_heads,
                     num_write_heads,
                     num_memories,
                     numeric_write_factor,
                     auxilary_dropout_rate,
                     device=device,
                     dtype=dtype
                     )
        for _ in range(num_layers)
    ]

    # Create and return the model
    return RecurrentDecoder(d_model, dropout_rate, layers)
