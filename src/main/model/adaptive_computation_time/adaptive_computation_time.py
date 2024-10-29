import torch
from torch import nn
from torch.nn import functional as F
from src.main.model.base import parallel_pytree_map, TensorTree
from typing import Any, Optional, Tuple, Dict, Callable
from src.main.model.adaptive_computation_time.abstract import (AbstractACT, AbstractACTFactory,
                                                               act_factory_registry)


class AdaptiveComputationTime(AbstractACT):
    """
    Implements Adaptive Computation Time (ACT) for models, enabling dynamic halting
    of computation based on learned halting probabilities.

    ---- Core Mechanism ----
    Allows models to dynamically adjust computation depth on a per-sample basis using a
    learned halting probability. Supports lazy initialization, multi-output accumulation,
    and flexible tensor shapes.

    The constructor will specify a batch shape, which defines the
    shapes of the features to accumulate halting probabilities
    over. Once that is done, step may be called to lazily initialze
    output accumulators. These features may be stored in a pytree
    so long as the leaves are tensors matching the following specification.

    Suppose the constructor was invoked with a batch shape of
    (...batch_shape). You could pass in any collection of
    floating tensors that begin with that shape. For instance,
    (...batch_shape, d_model), or (...batch_shape, items, d_model)

    You can also get more complex. For instance, using a dict
    or tuple. For instance, {"normalizer" : (...batch_shape, d_model),
    "matrix" : (...batch_shape, d_model, d_model)}. So long as it
    begins with the same dimensions, it remains compatible.

    ---- statistics ----

    steps_taken: Per batch, how many steps were executed. A returned statistic
    probabilistic_steps_taken: The above. However, we add up the remaining probability instead of the step, so
                               each update can be smaller than 1.0. Useful for loss
    residual_probabilities: The probability residual used when the computation halted.

    Args:
        max_steps (int): Maximum computation steps allowed for each input.
        threshold (float): Probability threshold for halting.
        batch_shape (torch.Size): Expected batch shape for halting probabilities.
    """
    @property
    def halted_batches(self)->torch.Tensor:
        return self.has_halted

    def __init__(self,
                 halting_probabilities: torch.Tensor,
                 residual_probabilities: torch.Tensor,
                 has_halted: torch.Tensor,
                 steps_taken: torch.Tensor,
                 probabilistic_steps_taken: torch.Tensor,
                 accumulated_outputs,
                 threshold: float
                 ):

        super().__init__()

        # Store core information.
        self.threshold = threshold
        self.device = halting_probabilities.dtype
        self.dtype = halting_probabilities.device
        self.has_halted = has_halted
        self.batch_shape = halting_probabilities.shape

        # Store probabilistic information
        self.residual_probabilities = residual_probabilities
        self.halting_probabilities = halting_probabilities

        # Store statistic information
        self.steps_taken = steps_taken
        self.probabilistic_steps_taken = probabilistic_steps_taken

        # Store accumulators
        self.accumulated_outputs = accumulated_outputs

    def accumulate_with_prob(self, prob: torch.Tensor, outputs: Any) -> Any:
        """
        Accumulates each output with the weighted halting probability `prob`,
        adjusting dimensions as needed for broadcasting.

        Args:
            prob (torch.Tensor): Halting probability for the step, of shape (...batch_shape).
            outputs: Model outputs, each a tensor or a nested structure of tensors.

        Returns:
            Accumulated outputs updated with weighted probabilities.
        """

        def weighted_accumulate(output, accumulated_output):
            # Unsqueeze prob to match output's dimensions for broadcasting
            step_prob = prob
            has_halted_mask = self.has_halted
            while step_prob.dim() < output.dim():
                step_prob = step_prob.unsqueeze(-1)
                has_halted_mask = has_halted_mask.unsqueeze(-1)
            updated_outputs = accumulated_output + output * step_prob
            return torch.where(~has_halted_mask, updated_outputs, accumulated_output)

        return parallel_pytree_map(weighted_accumulate, outputs, self.accumulated_outputs)

    def step(self, halting_prob: torch.Tensor, **outputs: Any):
        """
        Perform one step of ACT, updating accumulated outputs and halting state.

        Args:
            halting_prob (torch.Tensor): Halting probability for each sample, shape (...batch_shape).
            **outputs: Model outputs, each being a tensor or a nested structure of tensors.
        """

        if halting_prob.shape != self.batch_shape:
            raise ValueError("Halting probability must match initialized batch shape.")

        # Calculate remaining probability
        remaining_prob = 1.0 - self.halting_probabilities

        # Calculate adjusted halting probability with epsilon tolerance,

        will_be_halted = self.halting_probabilities + halting_prob >= self.threshold
        next_halting_prob = torch.where(
            will_be_halted,
            remaining_prob,
            halting_prob
        )

        # Calculate the residual probabilities
        newly_halted = will_be_halted & ~self.has_halted
        self.residual_probabilities = torch.where(newly_halted, remaining_prob, self.residual_probabilities)

        # Accumulate outputs with weighted current probabilities
        self.accumulated_outputs = self.accumulate_with_prob(next_halting_prob, outputs)

        # Update halting probabilities and steps taken
        self.steps_taken += (~self.has_halted).int()
        self.probabilistic_steps_taken += (~self.has_halted) * self.halting_probabilities
        self.halting_probabilities += next_halting_prob
        self.has_halted |= self.halting_probabilities >= self.threshold

    def should_continue(self) -> bool:
        """
        Check whether further computation is needed for any samples.

        Returns:
            bool: True if additional computation is needed, False if all samples have halted or reached max steps.
        """
        return not self.has_halted.all()
    def get_statistics(self)->Dict[str, torch.Tensor]:
        """
        Gets the statistics. These include
        - Steps takne
        - Probabilistic steps taken
        - residual probabilities.
        """
        statistics = {}
        statistics["steps_taken"] = self.steps_taken
        statistics["probabilistic_steps_taken"] = self.probabilistic_steps_taken
        statistics["residual_probabilities"] = self.residual_probabilities
        return statistics
    def finalize(self) -> Dict[str, TensorTree]:
        """
        Finalize accumulated outputs by ensuring remaining probability mass is included.
        :return: Finish accumulators
        :raises: RuntimeError, if we never halted
        """

        if self.should_continue():
            raise RuntimeError("ACT process did not finish")
        return self.accumulated_outputs


@act_factory_registry.register("Default")
class AdaptiveComputationTimeFactory(AbstractACTFactory):
    """
    Runs most of the primary initialization features,
    and loads it into the dynamic class. Is a factory
    method designed to allow creation of act instances
    to run the computation.
    """

    def __init__(self,
                 threshold: float = 0.99,
                 device: Optional[torch.device] = None,
                 dtype: Optional[torch.dtype] = None,
                 ):
        super().__init__(device=device, dtype=dtype)
        self.threshold = threshold

    def create_accumulator_factory(self, batch_shape: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
        """
        Creates a function that will initialize a accumulator,
        and sanity check it
        :return: The initialized factory.
        """
        def initialize_accumulator(output: torch.Tensor) -> torch.Tensor:
            if not torch.is_tensor(output):
                raise ValueError("All accumulated outputs must be tensors.")
            if output.dtype != self.dtype or output.device != self.device:
                raise ValueError("All accumulated outputs must be the same device and of same dtype")
            batch_len = len(batch_shape)
            if len(output.shape) < batch_len:
                raise ValueError("All accumulated outputs must have dimensions at least batch length")
            if output.shape[:batch_len] != batch_shape:
                raise ValueError(f"All accumulated outputs must have initial dimensions matching {batch_shape}")
            return torch.zeros_like(output)
        return initialize_accumulator
    def forward(self,
                batch_shape: torch.Size,
                **accumulator_templates: TensorTree
                ) -> AdaptiveComputationTime:
        """
        Sets up the adaptive computation time instance.
        :param batch_shape: The shape of the batch we are dealing with
        :param accumulator_templates:
        :return: The setup adaptive computation time instance.
        """

        # Initialize probability features
        halting_probabilities = torch.zeros(*batch_shape, device=self.device, dtype=self.dtype)
        residual_probabilities = torch.zeros(*batch_shape, device=self.device, dtype=self.dtype)
        has_halted = torch.zeros(*batch_shape, dtype=torch.bool, device=self.device)

        # Initialize statisticlike features
        steps_taken = torch.zeros(*batch_shape, dtype=self.dtype, device=self.device)
        probabilistic_steps_taken = torch.zeros(*batch_shape, device=self.device, dtype=self.dtype)

        # Initialize loss like features, including residual probability container
        # probabilistic cumulative length

        # Perform initialization of accumulators

        accumulator_factory = self.create_accumulator_factory(batch_shape)
        accumulated_outputs = parallel_pytree_map(accumulator_factory, accumulator_templates)

        # Create and return the instance
        return AdaptiveComputationTime(
            halting_probabilities,
            residual_probabilities,
            has_halted,
            steps_taken,
            probabilistic_steps_taken,
            accumulated_outputs,
            self.threshold
        )

