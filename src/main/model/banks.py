import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple, List, Dict, Any, Optional
from abc import abstractmethod, ABC
from src.main.model.base import StatefulCore, TensorTree

"""
The banks moduled is generally centered around selecting,
managing, and using parallel collections of probabilities
known as "banks".
"""




class DropoutLogits(nn.Module):
    """
    A dropout varient designed to operate on logits. It operates by
    masking to large negative values. This ensures that the softmax
    of these logits will be very small, as close to zero as is feasable.
    """
    def __init__(self,
                 probability: float = 0.2,
                 mask_value: float = -1e11
                 ):
        super().__init__()
        self.probability = probability
        self.mask_value = mask_value

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Performs the dropout process, if in eval mode.
        :param logits: The logits to dropout
        :return: The logits with the mask applied
        """
        if self.training:
            # Dropout happens only when training.
            mask = torch.rand_like(logits) < self.probability
            logits = logits.masked_fill(mask, self.mask_value)
        return logits

class BankedLinear(nn.Module):
    """
    "Banks" are sets of parallel kernels that essencially
    allow layers to be sideways, and dynamically selected
    as a computation proceeds. They exist to allow the same
    or similar layers to do many different processes.

    Banks are sparsely specified in terms of the index of the
    banks we want to use. We do, however, expect the bank
    dimension to already exist.
    """
    def __init__(self,
                 in_features,
                 out_features,
                 num_banks: int,
                 expand: bool = False,
                 squeeze: bool = False,
                 dtype: torch.dtype = None,
                 device: torch.device = None
                 ):
        """
        Init.

        :param in_features: Number of incoming features on the tensor
        :param out_features: Number of outgoing features on the tensor
        :param num_banks: The number of banks we will be dealing with.

        :param expand: Whether the incoming tensor already has a bank dimension in place, or we should add it
        :parem squeeze: Whether we want to get rid of the bank dimensions by the end of this.

        :param dtype: The dtype
        :param device: The device
        """

        super(BankedLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.num_banks = num_banks

        self.expand = expand
        self.contract = squeeze

        # Create weights banks
        self.weights = nn.Parameter(torch.zeros([num_banks, in_features, out_features],
                                                dtype=dtype, device=device
                                                ))
        self.bias = nn.Parameter(torch.zeros([num_banks, out_features], dtype=dtype, device=device))
        nn.init.kaiming_uniform_(self.weights)


    def extract_weights(self, bank_selection: torch.Tensor)->torch.Tensor:
        """
        Extracts the weights needed to run the process with the given bank
        selections. In particular, extracts a batched dense matrix kernel.
        Keep in mind, however, this kernel may have a multidimensional batch
        shape.

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
        selections. In particular, extracts a batched dense bias kernel.
        Keep in mind, however, this kernel may be attached to multidimensional
        batches.
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
                selection: Tuple[torch.Tensor, torch.Tensor]
                ) -> torch.Tensor:
        """
        Runs the given tensor using the selected parameter banks. Note that ...only indicates
        dimensions that can be present on the incoming tensor, but not the banks, in case you
        wish to get clever.

        :param tensor: The tensor to process.
            - Shape (..., ...only, in_features), when expand is true
            - Shape (..., ...only, bank_select, in_feature) when expand is false.
        :param selection: Two things, in this order.
            - sparse_bank_selections: The banks that were selected.  Shape (...,  banks_selected). Integers
            - sparse_bank_probabilities: The weights to use when combining the banks. Shape (..., banks_selected)
        :return:
            - Shape (..., ...only, out_features) when contract is true
            - Shape (..., ...only, bank_select, out_feature) when contract is false
            - Result of dense plus add plus combine
        """
        bank_selections, bank_probabilities = selection

        #Unsqueeze for extra dimension if dealing with expand condition
        if self.expand:
            tensor = tensor.unsqueeze(-2)  # (...,...only, 1, in_features)

        # Unsqueeze bank select to be ready to perform select
        while bank_selections.dim() < tensor.dim() - 1:
            bank_selections = bank_selections.unsqueeze(-2)
            bank_probabilities = bank_probabilities.unsqueeze(-2)

        # Get kernels

        weights = self.extract_weights(bank_selections)  #(..., banks_selected, in_features, out_features)
        bias = self.extract_bias(bank_selections) #(..., banks_selected, out_features)

        # Matrix multiplication
        tensor = tensor.unsqueeze(-2)
        tensor = torch.matmul(tensor, weights) #(..., banks_selected, out_features)
        tensor = tensor.squeeze(-2)

        # Bias

        tensor = tensor + bias #(..., banks_selected, out_features)

        # If contract, combine banks
        if self.contract:
            tensor = torch.sum(tensor * bank_probabilities.unsqueeze(-1), dim=-2)

        # Return
        return tensor

def banked_state_select(state: torch.Tensor,
                      selection: Tuple[torch.Tensor, torch.Tensor],
                      dim: int) -> torch.Tensor:
    """
    Performs a broadcasted banked select along the specified dimension.

    :param state: The state to select from. Should have shape (...batch, ..., options, ...)
    :param selection: The selection. Made up of:
        - index: The index to select. Shape (...batch, selected)
        - probabilities: Not used. Shape (...batch, selected)
    :param dim: The dimension to perform the selection along.
    :return: The selected features. Shape (...batch, ..., selected, ...)
    """

    # Unpack selection tuple
    indices, _ = selection  # Ignore probabilities for now

    # Move the selection dim to the end of the state tensor
    state = state.swapdims(dim, -1)  # (...batch, ..., options)

    # Ensure indices has the correct number of dimensions
    while indices.dim() < state.dim():
        indices = indices.unsqueeze(-2)

    # Expand indices to match the state tensor, excluding the last dimension
    indices = indices.expand(*state.shape[:-1], indices.shape[-1])  # (...batch, ..., selections)

    # Perform gather on the last dimension (which used to be dim)
    selected_state = state.gather(-1, indices)  # (...batch, ..., selections)

    # Restore the original dimension order by swapping back
    selected_state = selected_state.swapdims(-1, dim)

    return selected_state
def banked_state_scatter(state: torch.Tensor,
                          substate: torch.Tensor,
                          selection: Tuple[torch.Tensor, torch.Tensor],
                          dim: int) -> torch.Tensor:
    """
    Inserts substate into the state based on the selection along the specified dimension,
    using interpolation controlled by the selection probabilities.

    :param state: The original state tensor. Shape (...batch, ..., options, ...)
    :param substate: The substate tensor to insert. Shape (...batch, ..., selected, ...)
    :param selection: A tuple consisting of:
        - indices: The indices to select. Shape (...batch, selected)
        - probabilities: The interpolation weights. Shape (...batch, selected)
    :param dim: The dimension to perform the selection along.
    :return: The updated state tensor with the substate inserted using interpolation.
    """

    indices, probabilities = selection

    # Move the selection dim to the end of the state tensor for easier processing
    state = state.swapdims(dim, -1)  # (...batch, ..., options)
    substate = substate.swapdims(dim, -1) #(...batch, ..., selections

    # Ensure indices has the correct number of dimensions for broadcasting
    while indices.dim() < state.dim():
        indices = indices.unsqueeze(-2)
        probabilities = probabilities.unsqueeze(-2)

    # Expand indices and probabilities to match the state tensor
    indices = indices.expand(*state.shape[:-1], indices.shape[-1])  # (...batch, ..., selections)
    probabilities = probabilities.expand(*state.shape[:-1], probabilities.shape[-1])  # (...batch, ..., selections)

    # Gather the current state values at the specified indices
    gathered_state = state.gather(-1, indices)  # (...batch, ..., selections)

    # Perform interpolation between the gathered state and the substate using probabilities
    interpolated = (1 - probabilities) * gathered_state + probabilities * substate  # (...batch, ..., selections)

    # Scatter the interpolated values back into the original state tensor
    state = state.scatter(-1, indices, interpolated)

    # Restore the original dimension order by swapping back
    state = state.swapdims(-1, dim)

    return state


class AbstractBankSelector(nn.Module, ABC):
    """
    The abstract bank selector class. Capable of sparsely
    selecting banks of things to work with. Responsible for
    using a provided embeddings tensor for making such a
    determination.
    """
    @abstractmethod
    def setup_state(self, embeddings: torch.Tensor):
        """
        Sets up the state we will be using, if any
        :param embeddings: The example embeddings
        :return: The state
        """

    @abstractmethod
    def create_bank_probabilities(self,
                                  embeddings: torch.Tensor,
                                  state: TensorTree,
                                  *parameters: Any
                                  )->Tuple[torch.Tensor, TensorTree]:
        """
        Create and return bank probabilities, from the current situation.

        :param embeddings: The embeddings we can generate from.
        :param state: Any state that is needed
        :param parameters: Any passed parameters
        :return: The bank probabilities
        """
    def __init__(self,
                 dropout_probability: float = 0.0,
                 running_ave_weight: float = 0.001):

        self.dropout = nn.Dropout(dropout_probability)
        self.ave_weight = running_ave_weight


class AbstractBankSelector(StatefulCore):
    """
    Promises to consider the incoming embedding and select a subset
    of the N banks to consider. We will end up returing the banks
    selected, and the probabilities associated
    """
    def __init__(self,
                 d_model: int,
                 bank_size: int,
                 top_k: int,
                 dropout: float = 0.2,
                 statistics_weight: float = 0.001,
                 device: torch.device = None,
                 dtype: torch.dtype = None
                 ):

        assert top_k <= bank_size

        # Store
        self.d_model = d_model
        self.bank_size = bank_size
        self.statistics_weights = statistics_weight
        self.device = device
        self.dtype = dtype
        self.top_k = top_k

        # Setup

        self.dropout_logits = DropoutLogits(dropout)

        # Setup statistics bank.
        self.bank_statistics = torch.zeros([bank_size], device=device, dtype=dtype)


    @abstractmethod
    def create_bank_logits(self,
                           embeddings: torch.Tensor,
                           state: TensorTree) -> torch.Tensor:
        """
        Creates the bank logits that will be used in further selection processes.
        An abstract method that must be implemented. You must return both the
        logits and a state feature. The state feature may, however, be the same
        as you passed in

        :param embeddings: The embeddings. (..., d_model)
        :param state: Any state you wish to use
        :return:
            - The logits. (..., bank_size)
            - The state. Whatever you need
        """

    @abstractmethod
    def update_state(self, bank_probabilities: torch.Tensor, state: TensorTree)->TensorTree:
        """
        Some classes may use the computed bank probabilities to update their state.
        :param bank_probabilities: The probability of each bank being selected
        :param state: The last state.
        :return: The new state
        """

    def create_bank_probabilities(self,
                                  sparse_probabiliities: torch.Tensor,
                                  sparse_indices: torch.Tensor
                                  )->torch.Tensor:
        """
        :param sparse_probabiliities: The weights. Should sum up to 1. Shape (..., sparse))
        :param sparse_indices: The index of these weights in the banks. Shape (..,, sparse)
        :return: The full bank. Shape (..., bank_size)
        """
        bank_probabilities = torch.zeros(list(sparse_probabiliities.shape[:-1]) + [self.bank_size],
                                         device=sparse_probabiliities.device, dtype=sparse_probabiliities.dtype)
        bank_probabilities.scatter_(-1, sparse_indices, sparse_probabiliities)

        return bank_probabilities

    def update_bank_statistics(self, bank_probabilities):
        """
        Updates the bank statistics based on the provided bank probabilities
        :param bank_probabilities: The probabilities. Per bank. Shape (..., bank_size)
        """
        bank_probabilities = bank_probabilities.flatten(0, -2) #flatten until only two dimensions are left
        bank_probabilities = bank_probabilities.mean(dim=0) # Then take the mean over them all
        self.bank_statistics = self.bank_statistics*(1 - self.statistics_weights) \
                                + bank_probabilities*self.statistics_weights # And update the running average

    @abstractmethod
    def forward(self, embeddings: torch.Tensor, states: TensorTree
                )->Tuple[Tuple[torch.Tensor, torch.Tensor], TensorTree]:
        """
        Performs the forward pass. Handles accumulating statistics for monitoring or
        loss purposes.

        :param embeddings: The embeddings we need to work with to make the selection
        :param states: Any state we need to use
        :return:
            - Tuple
                - The selection weight
                - The selected bank indices
            - State:
                - Whatever you need to keep track of between selections. Can be minimal
        """

        # Get logits
        logits = self.create_bank_logits(embeddings, states)
        assert logits.shape[-1] == self.bank_size

        # Perform logit dropout.

        logits = self.dropout_logits(logits)

        # Find top candidates. Form them into probabilities.

        top_logits, top_index = logits.topk(self.top_k)
        top_probabilities = torch.softmax(top_logits, dim=-1)

        # Update statistics
        bank_probabilities = self.create_bank_probabilities(top_probabilities, top_index)
        self.update_bank_statistics(bank_probabilities)

        # Update state
        states = self.update_state(bank_probabilities, states)

        # return

        return (top_probabilities, top_index), states

class NaiveBankSelector(AbstractBankSelector):
    """
    A basic bank selector, that uses no state
    in it's decisions. Instead, it bases it's decision
    entirely off of the embeddings it sees.
    """
    def setup_state(self, tensor: torch.Tensor) ->TensorTree:
        # We just return a dictionary. This will result in no superposition.
        return {}

    def create_bank_logits(self,
                           embeddings: torch.Tensor,
                           state: TensorTree) -> torch.Tensor:
        """
        Computes the logits as a naive logit projection.

        :param embeddings: The embeddings to use
        :param state: The state.
        :return: The logits
        """

        logits = self.logits_projector(embeddings)
        return logits
    def update_state(self, bank_probabilities: torch.Tensor, state: TensorTree) ->TensorTree:
        """
        Naive state. No need to change.
        :param bank_probabilities:
        :param state:
        :return:
        """
        return state
    def __init__(self,
                 d_model: int,
                 bank_size: int,
                 top_k: int,
                 statistics_weight: float = 0.001,
                 dropout: float = 0.2,
                 gumbel: bool = False,
                 device: torch.device = None,
                 dtype: torch.dtype = None,
                 ):

        super().__init__(d_model, bank_size, top_k, dropout, gumbel,
                         statistics_weight, device, dtype)

        self.logits_projector = nn.Linear(d_model, bank_size)

class PseudoMarkovBankSelector(AbstractBankSelector):
    """
    The Pseudo Markov bank selector combines inputs based on the logits
    of the embeddings with an overall transition probability due to the
    markov state we are in.
    """
    def setup_state(self, tensor: torch.Tensor) ->TensorTree:
        """
        We setup a markov probability tensor tracking all the bank
        states. We update this later on.

        :param tensor: The tensor we are working with. Shape (..., embedding)
        :return: A tensor of shape (..., bank_size)
        """
        state = torch.zeros(list(tensor.shape[:-1]) + [self.bank_size], device=tensor.device, dtype=tensor.dtype)
        state[..., 0] = 1.0
        return state

    def create_bank_logits(self,
                           embeddings: torch.Tensor,
                           state: TensorTree) -> torch.Tensor:
        """
        Create the bank logits. We use the embeddings, and also
        :param embeddings:
        :param state:
        :return:
        """
        bank_probabilities = state
        logits = self.logits_projector(embeddings)
        logits += self.transitions_projector(bank_probabilities)
        return logits

    def update_state(self, bank_probabilities: torch.Tensor, state: TensorTree) -> TensorTree:
        """
        The new state is just the new bank probabilities

        :param bank_probabilities: The probabilities of each bank
        :param state: Not used
        :return: The new state
        """
        return bank_probabilities

    def __init__(self,
                 d_model: int,
                 bank_size: int,
                 top_k: int,
                 statistics_weight: float = 0.001,
                 dropout: float = 0.2,
                 gumbel: bool = False,
                 device: torch.device = None,
                 dtype: torch.dtype = None,
                 ):

        super().__init__(d_model, bank_size, top_k, dropout, gumbel,
                         statistics_weight, device, dtype)

        self.logits_projector = nn.Linear(d_model, bank_size)
        self.transitions_projector = nn.Linear(bank_size, bank_size)



