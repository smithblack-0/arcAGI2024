import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple, List, Dict
def extract_parameter_bank(layers: List[nn.Module])->Dict[str, nn.Parameter]:
    """
    Can be used to extract a parameter bank from a collection
    of layers. The layers should have the same structure, but do not
    need the same parameters.

    :param layers: The layers to extract from
    :return: A dictionary of parameters, where the last dimensions is from concatenating across
            list.
    """

    # Extract all parameters
    parameters = {}
    for layer in layers:
        for name, parameter in layer.named_parameters():
            if name not in parameters:
                parameters[name] = []
            parameters[name].append(parameter)

    # Stack. Create banks
    parameters = {key : torch.stack(item, dim=0) for key, item in parameters.items()}
    parameters = {key : nn.Parameter(item) for key, item in parameters.items()}
    return parameters
class BankedLinear(nn.Module):
    """
    A linear designed and expected to receive bank
    selection information.
    """

    def __init__(self,
                 in_features,
                 out_features,
                 num_banks: int,
                bias=True,
                 dtype: torch.dtype = None,
                 device: torch.device = None
                 ):

        super(BankedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_banks = num_banks
        self.bias = bias

        # Create and extract parameter banks
        layers = [nn.Linear(in_features, out_features, bias=bias, device=device, dtype=dtype) for _ in range(num_banks)]
        parameters = extract_parameter_bank(layers)

        # Store
        self.weight = parameters['weight']
        if self.bias:
            self.bias = parameters['bias']

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
        return self.weight[bank_selection, ...]

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
                bank_selections: torch.Tensor) -> torch.Tensor:
        """
        Runs the given tensor using the selected parameter banks
        :param tensor: The tensor to process
            - Shape (..., banks_selected, in_features)
        :param bank_selections: The banks that were selected
            - Shape (..., banks_selected)
            - Integers.
        :return:
            - Shape (..., banks_selected, out_features)
            - Result of dense
        """
        weights = self.extract_weights(bank_selections)
        if self.bias:
            bias = self.extract_bias(bank_selections)
            return F.linear(tensor, weights, bias)
        else:
            return F.linear(tensor, weights)

class BankSelector(nn.Module):
    """
    Selects one of N integers, where these integers
    would generally be corrolated with a bank of
    parameters of some kind.
    """
    def __init__(self,
                 d_model: int,
                 bank_size: int,
                 device: torch.device = None,
                 dtype: torch.dtype = None
                 ):
        """
        Chooses the top k out of banks, and optionally normalizes
        the selection to encourage even use of all parameter banks.

        :param d_model: The size of the model that will be used to make projections
        :param bank_size: The size of the bak that can be chosen from
        :param topk: The number of banks to choose from each time.
        :param encourage_normalization: If used, exceptionally inactive
               banks will see their probabilities increase, helping to ensure even usage of
               all bank states.
        :param normalization_factor:
                The normalization factor used to weight the normalization biases.
        """

        super().__init__()
        self.d_model = d_model
        self.selection_count = torch.zeros([bank_size])
        self.logit_projector = nn.Linear(self.d_model, bank_size, device=device, dtype=dtype)

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
        top_values, top_indices = torch.topk(tensor, top_k)
        top_probabilities = torch.softmax(top_values, dim=-1)
        return top_probabilities, top_indices



