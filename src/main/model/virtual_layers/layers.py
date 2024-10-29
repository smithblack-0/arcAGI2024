import textwrap
from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn

from src.main.model.virtual_layers import VirtualLayer, VirtualParameter, SelectionSpec

###
# Write code for a few different virtual layers, and their
# helping functions. Virtual layers support the virtual
# paradigm defined above, and always accept a selection
# parameter.
#
# Due to time constraints, they always operate in superposition.
# We are not going to support an ensemble mode. Ma
#
# TODO: Consider ensemble support.
##


class VirtualLinear(VirtualLayer):
    """
    A virtual linear layer. Implements the
    "linear" behavior for a particular
    selection spec, and handles batching elegantly
    to allow different batches to exist in different
    superpositions. See torch.nn.Linear for more details.
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bank_size: int,
                 bias: bool = True,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None,
                 ):
        super().__init__(bank_size)
        self.in_features = in_features
        self.out_features = out_features
        self.bank_size = bank_size
        self.has_bias = bias

        # Create the weights matrix, store it
        self.weights = VirtualParameter.create(bank_size,
                                               shape=[out_features, in_features],
                                               init=nn.init.kaiming_uniform_,
                                               dtype=dtype,
                                               device=device
                                               )

        if bias:
            self.bias = VirtualParameter.create(bank_size,
                                                shape=[out_features],
                                                init=nn.init.zeros_,
                                                dtype=dtype,
                                                device=device
                                                )

    def forward(self,
                tensor: torch.Tensor, selection: SelectionSpec) -> torch.Tensor:
        """
        Commences the forward process for the virtual linear layer.
        :param tensor: The input tensor to process. Shape: (batch_size, in_features)
        :param selection: The selection spec to use for choosing virtual parameters.
        :return: The resulting tensor after the linear transformation.
        """

        # Fetch the appropriate weight and bias based on the selection spec
        weights = self.weights(selection)  # Shape: (out_features, in_features)
        if self.has_bias:
            bias = self.bias(selection)  # Shape: (out_features,)


        # Perform the matrix multiplication: (batch_size, 1,  in_features) @ (batch_size, in_features, out_features).T
        tensor = tensor.unsqueeze(-1)
        tensor = torch.matmul(weights, tensor)  # Result: (batch_size, out_features)
        tensor = tensor.squeeze(-1)
        # Add the bias if required, making sure it broadcasts over the batch dimension
        if self.has_bias:
            tensor += bias

        return tensor


class VirtualMakeHeads(VirtualLayer):
    """
    A virtual layer to create attention heads from an input tensor.

    This layer applies a linear transformation to map the `d_model` dimension to
    `d_head * num_heads`, reshaping the result to generate `num_heads` attention heads
    each with dimensionality `d_head`. Supports multiple batch dimensions, allowing for
    flexible input shapes.

    Banked configuration allows different head configurations per batch.

    Attributes:
        d_model (int): Input dimensionality.
        d_head (int): Dimensionality of each attention head.
        num_heads (int): Number of attention heads.
        num_banks (int): Number of virtual banks for layer superposition.
    """

    def __init__(self,
                 d_model: int,
                 d_head: int,
                 num_heads: int,
                 num_banks: int
                 ):
        super().__init__(num_banks)

        self.d_model = d_model
        self.d_head = d_head
        self.num_heads = num_heads
        self.num_banks = num_banks

        # Define the projection to generate heads
        self.projection = VirtualLinear(d_model, d_head * num_heads, num_banks)

    def forward(self,
                tensor: torch.Tensor,
                bank_selection: SelectionSpec
                ) -> torch.Tensor:
        """
        Creates attention heads on the input tensor.

        Args:
            tensor (torch.Tensor): Input tensor with shape (..., d_model).
            bank_selection (SelectionSpec): Selection specification for bank superposition.

        Returns:
            torch.Tensor: Output tensor reshaped to (..., num_heads, d_head) after
            the transformation.
        """
        tensor = self.projection(tensor, bank_selection)  # (..., d_head*num_heads)
        tensor = tensor.reshape(tensor.shape[:-1] + (self.num_heads, self.d_head))
        return tensor


class VirtualMergeHeads(VirtualLayer):
    """
    A virtual layer to merge attention heads back to a single dimensionality.

    This layer applies a linear transformation to merge `num_heads` heads of
    dimensionality `d_head` into a `d_model` dimensional output. Supports
    flexible input shapes with multiple batch dimensions.

    Banked configuration allows different head configurations per batch.

    Attributes:
        d_model (int): Output dimensionality.
        d_head (int): Dimensionality of each attention head.
        num_heads (int): Number of attention heads.
        num_banks (int): Number of virtual banks for layer superposition.
    """

    def __init__(self,
                 d_model: int,
                 d_head: int,
                 num_heads: int,
                 num_banks: int
                 ):
        super().__init__(num_banks)

        self.d_model = d_model
        self.d_head = d_head
        self.num_heads = num_heads
        self.num_banks = num_banks

        # Define the projection to merge heads
        self.projection = VirtualLinear(d_head * num_heads, d_model, num_banks)

    def forward(self,
                tensor: torch.Tensor,
                bank_selection: SelectionSpec
                ) -> torch.Tensor:
        """
        Merges attention heads into a single dimension.

        Args:
            tensor (torch.Tensor): Input tensor with shape (..., num_heads, d_head).
            bank_selection (SelectionSpec): Selection specification for bank superposition.

        Returns:
            torch.Tensor: Output tensor reshaped to (..., d_model) after the transformation.
        """
        tensor = tensor.reshape(tensor.shape[:-2] + (self.num_heads * self.d_head,))
        tensor = self.projection(tensor, bank_selection)  # (..., d_model)
        return tensor


class VirtualFeedforward(VirtualLayer):
    """
    A Virtual feedforward process, with plenty of different
    of feedforward kernels that may vary depending on the
    exact configuration. Consists of two levels with
    an activation in between.
    """
    def __init__(self,
                 d_model: int,
                 d_hidden: int,
                 bank_size: int,
                 submodule_dropout: float,
                 device: torch.device = torch.device("cpu"),
                 dtype: torch.dtype = torch.float32,
                 ):
        super().__init__(bank_size)
        self.ff1 = VirtualLinear(d_model, d_hidden, bank_size, device=device, dtype=dtype)
        self.ff2 = VirtualLinear(d_hidden, d_model, bank_size, device=device, dtype=dtype)
        self.activation = torch.relu
        self.dropout = nn.Dropout(submodule_dropout)
    def forward(self,
                tensor: torch.Tensor,
                selection: SelectionSpec
                )-> torch.Tensor:
        """
        Runs the feedforward process under the given
        selection.
        :param tensor: The tensor to run feedforward with.
        :param selection: The virtual layer selection to use
        :return: The resulting tensor
        """
        tensor = self.ff1(tensor, selection)
        tensor = self.dropout(self.activation(tensor))
        tensor = self.ff2(tensor, selection)
        return tensor

class VirtualAdvancedLinear(VirtualLayer):
    """
    A small helper around virtual linear.
    It allows the declaration of arbitrary
    shape as target input, and arbitrary
    shape as output
    """
    def __init__(self,
                 in_shape: Tuple[int, ...],
                 out_shape: Tuple[int, ...],
                 bank_size: int,
                 dtype: torch.dtype = torch.float32,
                 device: torch.device = torch.device("cpu"),
                 ):
        super().__init__(bank_size)

        self.in_shape = torch.Size(in_shape)
        self.out_shape = torch.Size(out_shape)

        self.projector = VirtualLinear(self.in_shape.numel(), self.out_shape.numel(), bank_size,
                                       dtype=dtype, device=device
                                       )
    def forward(self, tensor: torch.Tensor, selection: SelectionSpec)->torch.Tensor:
        """
        Run the linear process, and include flattening and reshaping
        :param tensor: The tensor. Shape (..., *in_shape)
        :param selection: The selection
        :return: The output. Shape (..., *out_shape)
        """
        in_length = len(self.in_shape)
        if tensor.shape[-in_length:] != self.in_shape:
            msg = f"""
            Tensor had ending shape of {tensor.shape[-in_length:]}.
            However, constructor specified {self.in_shape}.
            """
            msg = textwrap.dedent(msg)
            raise ValueError(msg)
        tensor = tensor.flatten(-in_length, -1)
        tensor = self.projector(tensor, selection)
        tensor = tensor.unflatten(dim=-1, sizes=self.out_shape)
        return tensor