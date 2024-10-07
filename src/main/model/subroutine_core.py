import torch
from torch import nn
from typing import Tuple
from abc import abstractmethod

from src.main.model.subroutine_driver import SubroutineCore
from src.main.model.base import TensorTree, StatefulCore
from src.main.model.banks import BankedLinear, BankSelector
# Interface

class TransformerMemoryCore(StatefulCore):
    """
    The abstract memory core implementation. It is assumed
    long term dependencies are being managed in this class.
    """
    @abstractmethod
    def setup_state(self, tensor: torch.Tensor)->TensorTree:
        """
        Sets up the memory core recurrent state, based on the
        provided input tensor.
        :param tensor: The embedding tensor to setup based on
        :return: The memory core state. Return a empty dictionary if
                 you have none.
        """

    @abstractmethod
    def forward(self,
                embeddings: torch.Tensor,
                state: TensorTree
                )->Tuple[torch.Tensor, TensorTree]:
        """
        This should implement the forward pass of the memory core
        :param embeddings: The embeddings to process
        :param state: The state we have to work with
        :return:
            - The new embeddings
            - The new state
        """
class CoreBottleneck(StatefulCore):
    """
    The core exists in a bottleneck layer. This
    layer is banked, and many different bottleneck
    projections and restorations are available.
    """

    def setup_state(self, tensor: torch.Tensor)->TensorTree:
        # Create example tensor in the bottlenecked size

        example = torch.zeros(list(tensor.shape[:-1]) + [self.d_model],
                              device=tensor.device,
                              dtype=tensor.dtype)
        state = self.core.setup_state(example)
        state = {"transformer_state" : state}
        return state

    def __init__(self,
                 d_latent: int,
                 d_model: int,
                 num_banks: int,
                 core: TransformerMemoryCore,
                 device: torch.device,
                 dtype: torch.dtype,
                 ):

        super().__init__()

        # Define the controller layers
        self.selector = BankSelector(d_latent, num_banks, device, dtype)
        self.core = core
        self.num_banks = num_banks
        self.d_latent = d_latent
        self.d_model = d_model

        # Define the projection layers
        self.bottleneck_projection = BankedLinear(d_latent, d_model, num_banks)
        self.restoration_projection = BankedLinear(d_model, d_latent, num_banks)
    def forward(self,
                embeddings: torch.Tensor,
                states: TensorTree) ->Tuple[torch.Tensor, TensorTree]:
        """
        Implements the forward pass. Note that not all state features are actually initialized
        here. In fact, the "state

        :param embeddings: The tensor to be reduced, in the latent space. Shape (..., d_latent)
        :param states: The state information. Should contain the "selection state" feature, and
                       a "transformer_state" feature.
        :return: The results of running the core bottleneck
        """
        # Basic asserts
        assert isinstance(states, dict)
        assert "selection_state" in states, "Needed to pass an externally created state select"

        #Get weights
        selected_weights, selected_indexes = states["selection_state"]

        # Perform bottleneck.
        embedding = self.bottleneck_projection(embeddings, selected_weights, selected_indexes)

        # Insert required downstream feature into substate. It can then be accessed by the
        # core model.
        substate = states["transformer_state"].copy()
        substate["selection_state"] = states["selection_state"]

        # Run core model
        embedding, substate = self.core(embedding, states)

        # Get and setup the updated state. The tranformer state feature will change.
        states = states.copy()
        states["transformer_state"] = substate

        # Undo the bottleneck.
        embedding = self.restoration_projection(embedding, selected_weights, selected_indexes)

        return embedding, states


class Subroutine(SubroutineCore):
    """
    The subroutine both contains the core bottleneck,
    and also contains the bank selection and update logic
    required for success.
    """
    def setup_state(self, tensor: torch.Tensor)->TensorTree:
        """
        Sets up the initial state. This mainly boils down to
        setting up any state the bank selector needs, and
        any state the core needs.

        :param tensor: The embeddings tensor to use as an example while initializing
        :return: The setup state. Output state had better look like this
        """

        state = {
            "selector_state" : self.selector.setup_state(tensor),
            "bottleneck_state" : self.selector.setup_state(tensor)

        }
        return state

    def __init__(self,
                 bank_selector: BankSelector,
                 core_bottleneck: CoreBottleneck,
                 ):
        self.selector = bank_selector
        self.core = core_bottleneck

