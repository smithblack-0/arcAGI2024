import torch
from torch import nn
from typing import Optional
from src.main.model.banks import BankedLinear, BankSelector

class Feedforward(nn.Module):
    """
    A basic feedforward implementation
    """
    def __init__(self,
                 d_model: int,
                 d_hidden: Optional[int],
                 dropout: float = 0.1,
                 device: Optional[torch.device] = None,
                 dtype: Optional[torch.dtype] = torch.float32):

        super().__init__()

        d_hidden = 4*d_model if d_hidden is None else d_hidden

        self.activation = nn.ReLU()
        self.ff1 = nn.Linear(d_model, d_hidden, device=device, dtype=dtype)
        self.ff2 = nn.Linear(d_hidden, d_model, device=device, dtype=dtype)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor)->torch.Tensor:
        """
        Applies feedforward process. Dropout occurs on the hidden parameters for
        redundancy.

        :param x: The inputs
        :return: The outputs
        """

        x = self.ff1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.ff2(x)
        return x

class BankedFeedforward(nn.Module):
    """
    A specalized form of feedforward, it effectively
    contains a bunch of linear layers that are banked.
    Only a certain number of them are active each iteration,
    however, and the selected bank is based on a projection
    and probabilities.
    """
    def __init__(self,
                 d_model: int,
                 d_hidden: Optional[int],
                 num_banks: int = 100,
                 num_banks_selected: int = 3,
                 dropout: float = 0.1,
                 device: Optional[torch.device] = None,
                 dtype: Optional[torch.dtype] = torch.float32
                 ):
        """
        :param d_model:   Size of the input and output model parameters
        :param d_hidden:  Size of the hidden layer
        :param num_banks: The number of banks of layers.
        :param num_active: The number of banks selected when running. <= num banks
        :param dropout:  The dropout factor during training
        :param device: The device we will run on
        :param dtype: The dtype of our kernels
        """
        super().__init__()

        assert num_banks >= num_banks_selected


        d_hidden = 4*d_model if d_hidden is None else d_hidden

        self.activation = nn.ReLU()
        self.num_banks_selected = num_banks_selected

        # Define the linear layers. Then extract the weights
        self.bank_selector = BankSelector(d_model, num_banks, device=device, dtype=dtype)
        self.ff1 = BankedLinear(d_model, d_hidden, num_banks, device=device, dtype=dtype)
        self.ff1 = BankedLinear(d_hidden, d_hidden, num_banks, device=device, dtype=dtype)

        self.dropout = nn.Dropout(dropout)

    def forward(self,
                tensor: torch.Tensor,
                )->torch.Tensor:
        """
        Performs banked feedforward. Using only the indicated
        parameter banks.

        :param tensor: The input tensors
            - Shape (..., d_model)
        :return:
            - The processed attention pieces
            - Shape (..., bank_select, d_model)
        """
        # Create bank select and add dimensions
        bank_probabilities, bank_select = self.bank_selector(tensor)
        tensor = tensor.unsqueeze(-2)


        # Run parallel feedforwards.
        x = self.ff1(tensor, bank_select)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.ff2(x, bank_select) #(..., banks, d_model)

        # Combine banks by weights of probability.
        x = torch.matmul(x.T, bank_select.unsqueeze(-1)).squeeze(-1)

        return x
