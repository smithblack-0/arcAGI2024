import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple, List, Dict
from .base import StatefulCore, TensorTree


class BankedLinear(nn.Module):
    """
    A linear designed and expected to receive bank
    selection information.
    """
    def __init__(self,
                 in_features,
                 out_features,
                 num_banks: int,
                 dtype: torch.dtype = None,
                 device: torch.device = None
                 ):

        super(BankedLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.num_banks = num_banks

        # Create weights banks
        self.weights = nn.Parameter(torch.zeros([num_banks, in_features, out_features],
                                                dtype=dtype, device=device
                                                ))
        self.bias = nn.Parameter(torch.zeros([num_banks, out_features], dtype=dtype, device=device))
        nn.init.kaiming_uniform_(self.weights)


    def extract_weights(self, bank_selection: torch.Tensor)->torch.Tensor:
        """
        Extracts the weights needed to run the process with the given bank
        selections.
        :param bank_selection:
            - The given bank selections.
            - Shape (..., banks_selected)
            - Integer
        :return:
            - Weights
            - Shape (..., banks_selected, in_features, out_features
        """
        return self.weights[bank_selection, ...]

    def extract_bias(self, bank_selection: torch.Tensor)->torch.Tensor:
        """
        Extracts the bias needed to run the process with the given bank
        selections.
        :param bank_selection:
            - The given bank selections.
            - Shape (..., banks_selected)
            - Integer
        :return:
            - Weights
            - Shape (..., banks_selected, out_features
        """
        return self.bias[bank_selection, ...]

    def forward(self,
                tensor: torch.Tensor,
                bank_weights: torch.Tensor,
                bank_selections: torch.Tensor) -> torch.Tensor:
        """
        Runs the given tensor using the selected parameter banks. Note that ...only indicates
        dimensions that can be present on the incoming tensor, but not the banks, in case you
        wish to get clever.

        :param tensor: The tensor to process. Shape (..., ...only, in_features)
        :param bank_selections: The banks that were selected.  Shape (...,  banks_selected). Integers
        :param bank_weights: The weights to use when combining the banks. Shape (..., banks_selected)
        :return:
            - Shape (..., ...only, out_features)
            - Result of dense plus add plus combine
        """
        # Basic asserts
        assert bank_selections.shape == bank_weights.shape
        assert bank_selections.shape[:-1] == tensor.shape[:(bank_selections.dim() - 1)]

        # Add bank tensor position. Unsqueeze bank select to be ready to perform select

        tensor = tensor.unsqueeze(-2) #(...,...only, 1, in_features)
        while bank_selections.dim() < tensor.dim() - 1:
            bank_selections = bank_selections.unsqueeze(-2)
            bank_weights = bank_weights.unsqueeze(-2)

        # Perform matmul

        weights = self.extract_weights(bank_selections)  #(..., banks_selected, in_features, out_features)
        bias = self.extract_bias(bank_selections) #(..., banks_selected, out_features)

        # Matrix multiplication
        tensor = tensor.unsqueeze(-2)
        tensor = torch.matmul(tensor, weights) #(..., banks_selected, out_features)
        tensor = tensor.squeeze(-2)

        # Bias

        tensor = tensor + bias #(..., banks_selected, out_features)

        # Combine banks
        tensor = torch.sum(tensor*bank_weights.unsqueeze(-1), dim=-2)

        # Return
        return tensor


class AbstractBankSelector(StatefulCore):
    """
    Promises to select one of several banks.
    """
class BankSelector(nn.Module):
    """
    Selects one of N integers, where these integers
    would generally be corrolated with a bank of
    parameters of some kind.
    """
    def __init__(self,
                 d_model: int,
                 bank_size: int,
                 statistic_weight: float = 0.001,
                 device: torch.device = None,
                 dtype: torch.dtype = None
                 ):
        """
        Chooses the top k out of banks, and optionally normalizes
        the selection to encourage even use of all parameter banks.

        :param d_model: The size of the model that will be used to make projections
        :param bank_size: The size of the bak that can be chosen from
        :param statistic_weight: The weight to use when performing our statistical running average
        """

        super().__init__()
        self.d_model = d_model
        self.logit_projector = nn.Linear(self.d_model, bank_size, device=device, dtype=dtype)
        self.bank_statistics = torch.zeros([bank_size], device=device, dtype=dtype)
        self.statistics_weight = statistic_weight

    def forward(self,
                tensor: torch.Tensor,
                top_k: int,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select bank to use
        :param tensor: The tensor. Shape (..., d_model)
        :param top_k: The number of banks to choose from. int
        :return:
            - The chosen banks. (..., top_k)
            - The probabilistic weights. (..., top_k)
        """
        # Compute the output
        logits = self.logit_projector(tensor)
        top_values, top_indices = torch.topk(logits, top_k)
        top_probabilities = torch.softmax(top_values, dim=-1)

        # Compute, cache the access statistics
        bank_probabilities = torch.zeros_like(logits)
        bank_probabilities.scatter_(-1, top_indices, top_probabilities)

        access_statistics = bank_probabilities.flatten(0, -2).mean(dim=0)
        self.bank_statistics = self.bank_statistics*(1-self.statistics_weight) \
                               + access_statistics*self.statistics_weight

        # Return
        return top_probabilities, top_indices

class PseudoMarkovBankSelector(StatefulCore):
    """
    A finite state machine version of the bank selector, it keeps track
    of the probabilities which we last selected our banks with, and influenced
    what we can become next using transition probabilities.
    """
    def setup_state(self, tensor: torch.Tensor) -> TensorTree:
        """
        Sets up the state based on the incoming tensors. We will end up with
        a markov probability bank of similar shape, excluding data dims
        :param tensor: The tensor to setup the probability bank over. Shape (..., embeddings)
        :return: The setup states. Shape (num_banks, ....)
        """
        # Create banks
        state_probabilities = torch.zeros([self.bank_size] +list(tensor.shape[:-1]),
                                          device=tensor.device, dtype=tensor.dtype)

        # Initialize in first state.
        state_probabilities[0] = 1.0

        return state_probabilities

    def __init__(self,
                 d_model: int,
                 bank_size: int,
                 statistic_weight: float = 0.001,
                 top_k: int = 4,
                 device: torch.device = None,
                 dtype: torch.dtype = None
                 ):
        """
        Chooses the top k out of banks, and optionally normalizes
        the selection to encourage even use of all parameter banks.

        :param d_model: The size of the model that will be used to make projections
        :param bank_size: The size of the bak that can be chosen from
        :param statistic_weight: The weight to use when performing our statistical running average
        """

        super().__init__()
        self.d_model = d_model
        self.bank_size = bank_size
        self.top_k = top_k

        # Define the layers and parameters
        self.logit_projector = nn.Linear(self.d_model, bank_size, device=device, dtype=dtype)
        self.transitions = nn.Linear(self.bank_size, bank_size, device=device, dtype=dtype)

        # Define statistics behaviors
        self.bank_statistics = torch.zeros([bank_size], device=device, dtype=dtype)
        self.statistics_weight = statistic_weight

    def get_transition_logits(self, state_probabilities: torch.Tensor)->torch.Tensor:
        """
        Creates the relevant transition logits out of the current transition probabilities.
        These logits can then influence what can be transitioned to

        :param state_probabilities: The current probabilities. Shape (banks, ...)
        :return: The transition logits, for each transition. Shape (banks, ...)
        """
        return self.transitions(state_probabilities)
    def forward(self,
                tensor: torch.Tensor,
                states: TensorTree) ->Tuple[
                                                                  Tuple[torch.Tensor, TensorTree],
                                                                  TensorTree]:
        """
        Performs the pseudo markov bank selection, returing the probabilities then the banks
        :param tensor: The tensor of embeddings to use when making the decision
        :param states: The last state. Tensor. Shape (banks, ...). Probabilities
        :return:
            - The probabilities
            - The banks
            - The new state
        """

        # Compute bank probabilities
        logits = self.logit_projector(tensor)
        logits += self.get_transition_logits(states)

        # Compute probabilities and tops
        top_values, top_indices = torch.topk(logits, self.top_k)
        top_probabilities = torch.softmax(top_values, dim=-1)

        # Compute bank probabilities
        bank_probabilities = torch.zeros_like(logits)
        bank_probabilities.scatter_(-1, top_indices, top_probabilities)

        # Compute, cache the access statistics
        access_statistics = bank_probabilities.flatten(0, -2).mean(dim=0)
        self.bank_statistics = self.bank_statistics*(1-self.statistics_weight) \
                               + access_statistics*self.statistics_weight

        # Return
        return (top_probabilities, top_indices), bank_probabilities




