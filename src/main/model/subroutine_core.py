
import torch
from torch import nn
from typing import Tuple
from abc import abstractmethod

from src.main.model.subroutine_driver import SubroutineCore
from src.main.model.base import TensorTree, StatefulCore
from src.main.model.banks import BankedLinear, AbstractBankSelector
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
            "transformer_state" : self.core.setup_state(tensor)
        }
        return state

    def __init__(self,
                 bank_selector: AbstractBankSelector,
                 transformer_core: TransformerMemoryCore,
                 ):
        self.selector = bank_selector
        self.core = transformer_core


