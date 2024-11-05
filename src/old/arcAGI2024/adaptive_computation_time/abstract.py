from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Optional, Any, Tuple, Dict

import torch
from torch import nn

from ..base import TensorTree, DeviceDtypeWatch
from ..registry import InterfaceRegistry

class AbstractACT(ABC):
    """
    An abstract adaptive computation time (ACT) interface.

    ---- Core Mechanism ----

    The ACT mechanism allows models to dynamically adjust computation depth on a per-sample basis
    using a learned halting probability. It supports multi-output accumulation and flexible tensor
    shapes for adaptive computational processes.

    ---- flexibility ----

    Suppose the constructor is corrolated with a batch shape of (...batch_shape). You can pass in
    any collection of floating-point tensors that start with that shape, such as:
    - (...batch_shape, d_model)
    - (...batch_shape, items, d_model)

    Additionally, pytrees can also be encoded. For instance

    items = {"normalizer" : tensor(...batch_shape, d_model), "matrix" : tensor(...batch_shape, d_model)}

    And can be included using

    .step(halting_probabilities, items=  items)
    ---- Initialization ----

    All initialization actions should be performed in the factory
    method, with this just recieving the results.

    ---- Contract and Interface ----

    Implementers must provide concrete implementations for the following methods:

    - `get_statistics`: Retrieves statistics relevant to the ACT process.
    - `step`: Performs an ACT step, updating the accumulated outputs and halting state.
    - `should_continue`: Checks if further computation is required for any samples.
    - `finalize`: Returns the accumulated values upon completing the adaptive computation.

    """
    @property
    @abstractmethod
    def halted_batches(self)->torch.Tensor:
        """
        Shows what batches have halted
        """
    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """
        Returns some sort of set of statistics.
        :return:
        """

    @abstractmethod
    def step(self, halting_prob: torch.Tensor, **outputs: TensorTree):
        """
        Perform one step of ACT, updating accumulated outputs and halting state.

        :param halting_prob: Halting probability for each sample, shape (...batch_shape).
        :param outputs: Things to accumulate. Kwargs. All pytrees must have leaves ending
                        in tensors of shape (...batch_shape, ...). Leaves must have
                        consistent shape between iterations, and will be accumulated.
        """

    @abstractmethod
    def should_continue(self) -> bool:
        """
        Check whether further computation is needed for any samples.

        Returns:
            bool: True if additional computation is needed, False if all samples have halted or reached max steps.
        """

    @abstractmethod
    def finalize(self) -> Dict[str, TensorTree]:
        """
        Finalize accumulated outputs by ensuring remaining probability mass is included.

        :return: The dictionary containing the accumulated values
        """


class AbstractACTFactory(nn.Module):
    """
    A factory method for making an object that
    implements adaptive computation time and fufills
    an object contract.

    Adaptive computation time is an excellent computational
    technique that allows for variable length computation processes

    This factory pledges to create adaptive computation time
    instances when requested.
    """
    @property
    def device(self)->torch.device:
        return self.metainfo.device

    @property
    def dtype(self)->torch.dtype:
        return self.metainfo.dtype

    def __init__(self,
                 device: Optional[torch.device] = None,
                 dtype: Optional[torch.dtype] = None,
                 ):
        super().__init__()
        self.metainfo = DeviceDtypeWatch(device=device, dtype=dtype)

    @abstractmethod
    def forward(self,
                batch_shape: torch.Size,
                **accumulator_templates: TensorTree
                ) -> AbstractACT:
        """
        Sets up the act mechanism. Provided details must
        include the general batch shape, and then a pytree
        containing accumulators to initialize locations for

        :param batch_shape: The batch shape. (...batch_shape)
        :param accumulator_templates: A series of pytrees representing
              stateful features. Each leaf must be a tensor of shape (...batch_shape, ...)
        :return: The setup act instance.
        """

class ACTController(nn.Module):
    """
    A control mechanism for an adaptive computation
    process. It can both initialize state in the
    first place, and generate halting probabilities
    on the fly as needed.

    To get ACT setup, you must first provide a
    embedding and a set of accumulator templates.
    The response will be the setup act state. From
    that point forward, you must provide the same
    thing, but with a "act_state" keyword filled
    in with the last state.


    ---- simple example ---
    embedding = ....
    act_state = act_controller(embedding, output=embedding)
    while act_state.should_continue():
        ....
        embedding = ...
        act_state = act_controller(embedding, act_state, output=embedding)

    embedding, = act_state.finalize()

    ---- complex example ----
    embedding = ....
    lstm_state = ... (can be a pytree)
    act_state = act_controller(embedding, output=embedding, lstm_state=lstm_state)
    while lstm_state.should_continue():
        ....
        lstm_state = ...
        embedding = ...
        act_state = act_controller(embedding, act_state, output=embedding, lstm_state=lstm_state)
    embedding, lstm_state = act_state.finalize()
    """
    def __init__(self,
                 d_model: int,
                 act_factory: AbstractACTFactory,
                 device: torch.device,
                 dtype: torch.dtype
                 ):

        super().__init__()

        # Setup the probability projector
        self.probability_projector = nn.Linear(d_model, 1, device=device, dtype=dtype)

        # store the factory
        self.act_factory = act_factory
    def create_state(self,
                     batch_shape: torch.Size,
                     **accumulator_templates: TensorTree
                     )->AbstractACT:
        """
        Creates a blank ACT state.
        :param batch_shape: The batch shape to match
        :param accumulator_templates: The accumulations to template
        :return: The setup act process
        """
        batch_shape = torch.Size(batch_shape)
        return self.act_factory(batch_shape, **accumulator_templates)


    def forward(self,
                control_embedding: torch.Tensor,
                act_state: Optional[AbstractACT] = None,
                **accumulation_details: TensorTree
                )->AbstractACT:
        """
        Runs the ACT process, setting it up if needed.
        One either
        :param control_embedding: The embedding. Shape (...batch_shape, d_model).
            - Used to produce halting probabilities and arcAGI2024 features.
            - Also we can figure out batch shape from this.
        :param act_state: The act state
            - Optional. Not provided means we setup an ACT state before running
        :param intermediate_state: The intermediate state to be accumulated. Should
            be a collection of pytrees assigned by kwargs.
        :return: The updated abstract act instance.
        """

        if not act_state.should_continue():
            raise RuntimeError("You should have stopped when should continue was false")

        #Run ACT update. Generate halting probabilities, incorporate new
        # details in step

        halting_probability = self.probability_projector(control_embedding).squeeze(-1)
        halting_probability = torch.sigmoid(halting_probability)
        act_state.step(halting_probability, **accumulation_details)
        return act_state



act_factory_registry = InterfaceRegistry[AbstractACTFactory]("ACTFactory", AbstractACTFactory)


act_controller_registry = InterfaceRegistry[ACTController]("ACTController",
                                                           ACTController,
                                                           act_factory=act_factory_registry)
act_controller_registry.register_class("Default", ACTController)