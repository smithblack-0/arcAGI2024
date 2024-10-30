import textwrap
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Optional, Any, Tuple, Dict, Union, List, Callable

import torch
from torch import nn

from src.main.model.base import TensorTree, SavableState
from src.main.model.registry import InterfaceRegistry
BatchShapeType = Union[Tuple[int, ...], List[int], torch.Size]

class AbstractSupportStack(SavableState, ABC):
    """
    The abstract implementation of a computational support
    stack. It indicates the methods the external world
    must implement in order to work with the stack state,
    including the various adjustment mechanisms and the
    save/load requirements.

    It is designed to be fairly transparent, so the
    external world does not have to be concerned
    about the fact there is a stack.

    ---- initialization ----

    Initialization should be performed primarily
    in the factory layer. Conceptually, however,
    this is initialized by that layer with a stack
    of pytrees related to the given defaults.

    The abstract variation also sets up some helper
    functions for working with action probabilities.

    ---- pytree flexibility -----

    The computational support stack can accept pytree arguments
    and support them throughout the stack process. This is relevant
    principly during initialization, which is handled in the factory
    class, and during update, when the associated pytrees should get
    passed in for integration into the stack.

    For instance, if you have a setup support stack "css", you
    would be able to do:

    css.update(stack_probabilities, embedding=update, lstm_state = lstm_state)

    and reasonably expect to get away with it, despite lstm state being a
    tuple in torch. This only works if the leaves of pytrees in update have
    initial shapes (...batch_shape)

    ---- concepts ----

    - batch shape:
        - A batch shape to bind the stack to.
        - All tracked pytrees must have initial.
        - Shape (...batch_shape)
    - enstack, no_op, destack:
        - Conceptually, the stack can be manipulated by an enstack action, no_op action, and destack action.
        - How strongly the model wants to do each is provided as action probabilitie
    - controls:
        - Tensors used to manipulate the stack.
        - Generated in the controller. 
        - The layer should respond to them in adjust stack.
        - 0: enstack, 1:no_op, 2:destack
    - stack expression:
        - An expression of the stack in a form that downstream layers can digest.
        - Shape is equal to default pytree shape.
    - stack update:
        - Step that integrates update into the stack.
        - Shape is equal to default pytree shape.

    ---- contract ----
    
    Lets discuss the contract you must implement. It is as follows
        
    - pop: Returns the content of the stack based on its configuration. 
    - push: Updates the stack with new contents based on its configuration
    - adjust_stack: Adjusts the stack based 
    - get_statistics: Gets the statistics associated with the stack so far.
    - save_state: as per SavableState
    - load_state: as per SavableState

    """

    def __init__(self,
                 stack_depth: int,
                 batch_shape: BatchShapeType,
                 dtype: torch.dtype,
                 device: torch.device,
                 ):
        self.batch_shape = batch_shape
        self.dtype = dtype
        self.device = device
        self.stack_depth = stack_depth

    def is_tensor_sane(self, tensor: Any, name: str) -> torch.Tensor:
        """
        Purely a utility method.
        Capable of checking if a given tensor is a sane match for the stack being tracked.
        raises error if not
        :return: The sanitized tensor.
        """
        batch_len = len(self.batch_shape)
        if not torch.is_tensor(tensor):
            raise ValueError(f"'{name}'  is not a tensor")
        if tensor.shape[:batch_len] != self.batch_shape:
            msg = f"""
            Issue with '{name}'. Batch shape was '{self.batch_shape}',
            but {name} had shape {tensor.shape}. This meant that
            {tensor.shape[:batch_len]} does not match.
            """
            msg = textwrap.dedent(msg)
            raise ValueError(msg)
        if tensor.dtype != self.dtype:
            msg = f"""
            Issue with '{name}'.
            Expected dtype was '{self.dtype}', however, got {tensor.dtype}
            """
            msg = textwrap.dedent(msg)
            raise TypeError(msg)
        if tensor.device != self.device:
            msg = f"""
            Issue with '{name}'.
            Expected device was '{self.device}', however, got {tensor.device}
            """
            msg = textwrap.dedent(msg)
            raise TypeError(msg)
        return tensor
    @abstractmethod
    def get_statistics(self) -> Dict[str, torch.Tensor]:
        """
        Gets a cluster of statistics related to the stack.
        :return: The statistics.
        """

    @abstractmethod
    def adjust_stack(self, controls: Tuple[torch.Tensor, ...], batch_mask: Optional[torch.Tensor]):
        """
        Adjusts the stack to account for the directives from the broader model.
        Responsible for handling this in a differentiable manner.
        :param controls:
            - Adjustment controls, like action probabilities or focus.
            - Implementation dependent.
            - Presumably, one of them is action probabilities or action logits of 
              shape (...batch_shape, 3)
        :param batch_mask:
            - Optional
            - Indicates adjustments to not make, presumably due to halting.

        """

    @abstractmethod
    def pop(self, name: Optional[str] = None) ->  Union[Dict[str, TensorTree], TensorTree] :
        """
        Gets an expression of the stored stack, in terms of the
        various kwargs that were defined. Return something that
        is differentiable and lacking stack dimension.
        :param name: If provided, only the indicated
                     kwarg will be popped
        :return: The expressed version of the stack.
            - No stack dimension.
            - Exact shape depends on original pytree setup
            - With name, may just be tensortree.
        """

    @abstractmethod
    def push(self, batch_mask: Optional[torch.Tensor], **states):
        """
        Pushes new state information onto the stack for the setup
        kwarg accumulator. It will be inserted based on the current
        stack configuration.
        :param batch_mask:
            - Optional
            - Indicates adjustments to not make, presumably due to halting.
            - Shape (...batch_shape)
            - True means mask
        :param states: The state information to integrate into the stack
        """

    def __call__(self,
                 controls: Tuple[torch.Tensor, ...],
                 batch_mask: Optional[torch.Tensor] = None,
                 **updates: TensorTree
                 ) -> Dict[str, TensorTree]:
        """
        :param controls:
            - Adjustment controls, like action probabilities or focus.
            - Implementation dependent.
            - Presumably, one of them is action probabilities or action logits of 
              shape (...batch_shape, 3)
        :param batch_mask:
            - Optional
            - Indicates adjustments to not make, presumably due to halting.
            - Shape (...batch_shape)
            - True means mask
        :param updates: tensor pytrees whose updates we need to integrate
        :return: The extracted stack state. 
        """
        self.adjust_stack(controls, batch_mask)  # Adjust position
        read = self.pop()  # Read
        self.push(batch_mask, **updates)  # Write
        return read


class AbstractStackFactory(nn.Module, ABC):
    """
    A factory method for making an object that
    implements the support stack contract, and that is 
    corrolated with a particular implementation

    This factory pledges to create a abstract support
    stack instance when invoked with the batch shape,
    stack depth, and default stack contents.
    """

    def __init__(self,
                 dtype: torch.dtype,
                 device: torch.device
                 ):
        super().__init__()
        self.dtype = dtype
        self.device = device

    def is_tensor_sane(self,
                       tensor: Any,
                       name: str,
                       batch_shape: BatchShapeType,
                       ) -> torch.Tensor:
        """
        Purely a utility method.
        Capable of checking if a given tensor is a sane match for the stack being tracked.
        raises error if not
        :return: The sanitized tensor.
        """
        batch_len = len(batch_shape)
        if not torch.is_tensor(tensor):
            raise ValueError(f"'{name}'  is not a tensor")
        if tensor.shape[:batch_len] != batch_shape:
            msg = f"""
            Issue with '{name}'. Batch shape was '{batch_shape}',
            but {name} had shape {tensor.shape}. This meant that
            {tensor.shape[:batch_len]} does not match.
            """
            msg = textwrap.dedent(msg)
            raise ValueError(msg)
        if tensor.dtype != self.dtype:
            msg = f"""
            Issue with '{name}'.
            Expected dtype was '{self.dtype}', however, got {tensor.dtype}
            """
            msg = textwrap.dedent(msg)
            raise TypeError(msg)
        if tensor.device != self.device:
            msg = f"""
            Issue with '{name}'.
            Expected device was '{self.device}', however, got {tensor.device}
            """
            msg = textwrap.dedent(msg)
            raise TypeError(msg)
        return tensor

    @abstractmethod
    def forward(self,
                batch_shape: BatchShapeType,
                stack_depth: int,
                **defaults: TensorTree
                ) -> AbstractSupportStack:
        """
        The implementation ot use
        :param batch_shape: The batch shape to match to
        :param stack_depth: The depth to make the stack to
        :param defaults: The default stack pytrees. We will assume stack locations
                         that are empty should look like this
        :return: The initialized stack.
        """


class AbstractControlGates(nn.Module, ABC):
    """
    Produces the control feature used to control the
    abstract stack, from an embedding with a width
    of d_model.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    @abstractmethod
    def forward(self, control_embedding: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Produce the control features such as action probabilities or other out of the control
        embedding
        :param control_embedding: The control embedding of concern
        :return: The relevant control features.
        """


class AbstractStackController(nn.Module, ABC):
    """
    An abstract stack controller, intended to
    control and manipulate the entire show. This
    includes stack creation, and then stack
    manipulation.

    It is indented this be implemented into something that
    creates and provides the indicated mechanisms, with
    most of the needed work relating to extending
    initialization.
    """

    def __init__(self,
                 control_gate: AbstractControlGates,
                 stack_factory: AbstractStackFactory,
                 ):
        super().__init__()
        self.stack_factory = stack_factory
        self.control_gate = control_gate

    def create_state(self,
                     batch_shape: BatchShapeType,
                     stack_depth: int,
                     **defaults: TensorTree
                     ) -> AbstractSupportStack:
        """
        Sets up the abstract support stack.
        :param batch_shape: The batch shape to match to
        :param stack_depth: The depth to make the stack to
        :param defaults: The default stack pytrees. We will assume stack locations
                         that are empty should look like this
        :return: The initialized stack.
        """
        batch_shape = torch.Size(batch_shape)
        return self.stack_factory(batch_shape, stack_depth, **defaults)

    def forward(self,
                control_embedding: torch.Tensor,
                stack_state: AbstractSupportStack,
                batch_mask: Optional[torch.Tensor] = None,
                **tracked_states: TensorTree
                ) -> Dict[str, TensorTree]:
        """
        Runs an abstract stack update.
        :param control_embedding: The control embedding to use
        :param batch_mask: Used to indicate, for example, if we want certain batches to halt
                           because of ACT processes. Can be left as none, denying that masking
                           from happening
        :param stack_state: The stack state to use
        :param tracked_states: The tracked state
        :return: The resulting tensors, fetched from the stack. the stack is indirectly updated.
        """
        controls = self.control_gate(control_embedding)
        read = stack_state(controls, batch_mask, **tracked_states)
        return read

stack_controller_registry = InterfaceRegistry[AbstractStackController]("StackController", AbstractStackController)
