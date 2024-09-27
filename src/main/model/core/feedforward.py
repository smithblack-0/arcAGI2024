import torch
from torch import nn
from typing import Optional


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
