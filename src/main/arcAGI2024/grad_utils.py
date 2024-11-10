from abc import ABC, abstractmethod

import torch
import functools
from torch import nn
from .base import parallel_pytree_map, middle_quantiles_mean, TensorTree
from typing import Any, List


class BatchCollectiveReductiveGradNorm:
    """
    Performs collective gradient normalization on a group of gradients flowing through
    the class, on a per-batch basis.

    Steps:
    1. Reduce each gradient collection per batch.
    2. Find the signal that is largest per batch.
    3. Rescale all gradients in a batch to be smaller if they exceed the threshold.

    This effectively reduces all gradients flowing from one recurrent cell to another
    by the same amount, per batch.
    """

    def __init__(self,
                 num_batch_dims: int = 1,
                 rescale_threshold: float = 1.0,
                 reduction_mode: str = 'quartiles_mean',
                 verbose: bool = False,
                 ):
        """
        Initializes the gradient normalization class.

        :param num_batch_dims: The number of batch dimensions.
        :param rescale_threshold: Threshold for triggering rescaling.
        :param reduction_mode: Reduction method for gradient tensors;
              options:
              - "mean": Reduces using mean as option
              - "max": Reduces using max as an option
              - "sum": Reduces using sum as an option
              - "quartiles_mean": Performs a mean around only the middle two quantiles. Default
        """
        assert num_batch_dims >= 0
        assert reduction_mode in ['mean', 'max', 'sum', 'quartiles_mean']
        assert rescale_threshold >= 0.0

        self.num_batch_dims = num_batch_dims
        self.rescale_threshold = rescale_threshold
        self.reduction_mode = reduction_mode
        self.verbose = verbose

    def collect_gradients(self,
                          grads: torch.Tensor,
                          accumulator: List[torch.Tensor]
                          ):
        """
        Collects gradients and appends them to an accumulator.
        :param grads: Gradients to be processed.
        :param accumulator: List to store reduced gradients for later processing.
        """
        if grads.dim() < self.num_batch_dims:
            raise ValueError("Received fewer gradient dimensions than batch dimensions.")
        grads = grads.abs()
        grads = grads.flatten(self.num_batch_dims, -1)
        accumulator.append(grads)

    @staticmethod
    def rescale_gradients(grads: torch.Tensor, rescale_factor: torch.Tensor) -> torch.Tensor:
        """
        Rescales gradients by the given rescale factor.

        :param grads: Gradients to rescale.
        :param rescale_factor: Rescale factor per batch.
        :return: Updated gradients.
        """
        while rescale_factor.dim() < grads.dim():
            rescale_factor = rescale_factor.unsqueeze(-1)
        return rescale_factor * grads

    def __call__(self, grad_tree: Any) -> Any:
        """
        Processes the gradient collection for scaling as necessary.

        :param grad_tree: The gradient tree to handle.
        :return: The normalized gradient tree.
        """
        # Accumulate reduced gradients from all parts of the gradient tree
        grad_accumulator = []
        reduce_gradients = functools.partial(self.collect_gradients, accumulator=grad_accumulator)
        parallel_pytree_map(reduce_gradients, grad_tree)

        # Calculate reduced statistics per batch over entire collection.
        summaries = []
        for observation in grad_accumulator:
            if self.reduction_mode == "mean":
                summary = observation.mean(dim=-1)
            elif self.reduction_mode == "max":
                summary = observation.max(dim=-1).values
            elif self.reduction_mode == "sum":
                summary = observation.sum(dim=-1)
            elif self.reduction_mode == "quartiles_mean":
                summary = middle_quantiles_mean(observation, dim=-1)
            else:
                raise ValueError(f"Unrecognized reduction mode: {self.reduction_mode}")
            summaries.append(summary)
        observed_grads, _ = torch.stack(summaries, dim=-1).max(dim=-1)

        if self.verbose:
            print(observed_grads)

        # Compute rescale factors per batch
        requires_rescale = observed_grads > self.rescale_threshold
        rescale_factor = torch.where(requires_rescale, self.rescale_threshold / observed_grads, 1.0)

        # Rescale gradients across the tree based on batch-specific factors
        rescale_gradients = functools.partial(self.rescale_gradients, rescale_factor=rescale_factor)
        return parallel_pytree_map(rescale_gradients, grad_tree)


class BatchCollectiveQuantileClipping:
    """
    A extremum clipping algorithm that is resistant
    to outliers. It takes the mean within the
    quantiles around the mean, then clips anything
    that exceeds that mean by a factor.

    It operates on a provided pytree
    """

    def __init__(self,
                 num_batch_dims: int = 1,
                 clip_factor: float = 1000.0,
                 protection_threshold: float = 0.0001,
                 mean_mode: str = 'quantiles_mean',
                 verbose: bool = False,
                 ):
        """
        Initializes the clipping algorithm
        :param num_batch_dims: The number of batch dimensions. Each batch is clipped separately
        :param clip_factor: When the gradient exceeds the mean by this factor, clip.
        :param protection_threshold: Gradients that are smaller than this cannot be clipped.
        :param mean_mode: Either 'mean' or 'quantiles_mean'
        - mean: The mean is taken. It is then used to determine when to clip
        - quantiles_mean: The middle two quantiles are used to compute the mean instead. Default.
        """
        assert clip_factor >= 1.0
        assert mean_mode in ['mean', 'quantiles_mean']
        self.num_batch_dims = num_batch_dims
        self.mean_mode = mean_mode
        self.clip_factor = clip_factor
        self.protection_threshold = protection_threshold
        self.verbose = verbose

    def collect_gradients(self,
                          grads: torch.Tensor,
                          accumulator: List[torch.Tensor]
                          ):
        """
        Collects gradients and appends them to an accumulator.
        :param grads: Gradients to be processed.
        :param accumulator: List to store reduced gradients for later processing.
        """
        if grads.dim() < self.num_batch_dims:
            raise ValueError("Received fewer gradient dimensions than batch dimensions.")
        grads = grads.abs()
        grads = grads.flatten(self.num_batch_dims, -1)
        accumulator.append(grads)

    @staticmethod
    def clip_gradients(grads: torch.Tensor, clip_threshold: torch.Tensor) -> torch.Tensor:
        """
        Performs the actual gradient clipping process.
        :param grads: The gradients to clip.
        :param clip_threshold: The threshold to clip to if needed.
        :return: The clipped gradients
        """
        while clip_threshold.dim() < grads.dim():
            clip_threshold = clip_threshold.unsqueeze(-1)
        return torch.minimum(grads, clip_threshold)

    def __call__(self, grad_tree: Any) -> Any:
        """
        Runs the clipping algorithm.
        :param grad_tree: The gradient tree to clip
        :return: The clipped gradients
        """
        # Collect the gradients, so we can find the clip values
        accumulator = []
        collect_gradients = functools.partial(self.collect_gradients, accumulator=accumulator)
        parallel_pytree_map(collect_gradients, grad_tree)

        # Calculate clip threshold.
        observations = torch.concat(accumulator, dim=-1)

        if self.mean_mode == "mean":
            observed_grads = observations.mean(dim=-1)
        elif self.mean_mode == "quantiles_mean":
            observed_grads = middle_quantiles_mean(observations, dim=-1)
        else:
            raise ValueError(f"Unrecognized reduction mode: {self.mean_mode}")

        threshold = observed_grads * self.clip_factor
        threshold = threshold.clip(min=self.protection_threshold)
        if self.verbose:
            print(threshold)

        # Perform any required clipping.
        clip_gradients = functools.partial(self.clip_gradients, clip_threshold=threshold)
        return parallel_pytree_map(clip_gradients, grad_tree)


class GradClip:
    """
    A simple gradient clipper.

    Gradients that go above a certain threshold
    are clipped away.
    """

    def __init__(self, clip_threshold: float):
        self.clip_threshold = clip_threshold

    def clip_gradients(self, grad: torch.Tensor) -> torch.Tensor:
        return grad.clip(max=self.clip_threshold, min=-self.clip_threshold)

    def __call__(self, grad_tree: Any) -> Any:
        """
        Runs the grad clipping process
        :param grad_tree: A grad tree to clip
        :return: The clipped gradients
        """
        return parallel_pytree_map(self.clip_gradients, grad_tree)
class AbstractGradientControl(nn.Module, ABC):
    """
    A class which is dedicated completely to controlling
    gradients and preventing them from exploding.

    It should directly accept, then return, gradients.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, gradient_tree: TensorTree)->TensorTree:
        pass
class AutorescaleGradientControl(AbstractGradientControl):
    """
    Automatically rescales gradients flowing through
    it, and clips excessive values after that
    rescale.
    """
    def __init__(self,
                 rescale_threshold: float = 1.0,
                 rescale_reduction_mode: str = "mean",
                 clip_factor: float = 100,
                 num_batch_dims: int = 1,
                 ):
        """
        :param num_batch_dims: Number of batch dimensions.
        :param rescale_threshold: Grads that exceed this threshold when reduced will be scaled down to be in proportion
        :param rescale_reduction_mode: Reductive modes. Recommend "mean" or "max".
        :param clip_factor: When, after rescaling, we still exceed this we clip.
        """
        super().__init__()
        self.rescaling = BatchCollectiveReductiveGradNorm(num_batch_dims, rescale_threshold, rescale_reduction_mode)
        self.clipper = GradClip(clip_factor*rescale_threshold)
    def forward(self, gradient_tree: TensorTree) -> TensorTree:
        """
        Runs the actual gradient control process.
        """
        with torch.no_grad():
            gradient_tree = self.rescaling(gradient_tree)
            gradient_tree = self.clipper(gradient_tree)
        return gradient_tree
