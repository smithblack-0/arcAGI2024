import torch
from torch import nn
from typing import List, Dict, Tuple, Any, Union, Optional, Callable
from abc import ABC, abstractmethod
from .subroutine_stubs import SubroutineLogicStub

from src.main.model.base import TensorTree, StatefulCore


# Define important types
SubroutineStubTree = Union[SubroutineLogicStub,
                           List['SubroutineStubTree'],
                           Tuple['SubroutineStubTree', ...],
                           Dict[str, 'SubroutineStubTree']
                           ]

StackTree = Union['SubroutineStackTracker',
                 List['StackTree'],
                 Tuple['StackTree', ...],
                 Dict[str, 'StackTree']]

# Define a very important utility function
def parallel_pytree_map(func: Callable[..., Any], *pytrees: Any) -> Any:
    """
    Recursively applies a function to corresponding leaves of multiple pytrees with the same structure.

    Args:
        func (Callable[..., Any]): A function to apply to corresponding leaves of the pytrees.
        *pytrees (NestedTensor): Multiple pytrees with the same structure.

    Returns:
        NestedTensor: A new pytree with the function applied to corresponding leaves.
    """
    # Check if all pytrees are lists, tuples, or dicts
    if all(isinstance(pytree, list) for pytree in pytrees):
        return [parallel_pytree_map(func, *elems) for elems in zip(*pytrees)]
    elif all(isinstance(pytree, tuple) for pytree in pytrees):
        return tuple(parallel_pytree_map(func, *elems) for elems in zip(*pytrees))
    elif all(isinstance(pytree, dict) for pytree in pytrees):
        return {key: parallel_pytree_map(func, *(pytree[key] for pytree in pytrees))
                for key in pytrees[0]}
    else:
        # These are leaves, apply the function to them
        return func(*pytrees)

# Define user features.
class SubroutineCore(StatefulCore):
    """
    The interface for the computational engine that can be
    put inside a subroutine driver. It provides the parameters
    to use the state of the stack to perform computations. It also
    will produce embeddings and state output that can be used
    to manage the subroutine stack
    """
    @abstractmethod
    def setup_state(self, tensor: torch.Tensor)->TensorTree:
        """
        Sets up state based on the provided tensor of embeddings. Note that
        if you do not use state, just return an empty dict.

        :param tensor: The tensor of embeddings
        :return: Whatever state we need. Can be none.
        """
    @abstractmethod
    def forward(self,
                tensor: torch.Tensor,
                states: TensorTree,
                **parameters)->Tuple[torch.Tensor, TensorTree]:
        """
        Performs the forward pass. Tensor is a tensor of embeddings, while states is any
        state information that needs to be tracked.
        :param tensor:
        :param states:
        :param parameters: Any extra parameters
        :return:
        """
        pass

class ActionsProbabilities:
    """
    Manages the creation of action probabilities from
    logits. Also tracks certain statistics and state
    information. Note that index 0: return, index 1: maintain,
    index 2: create.
    """

    @property
    def can_pop(self)->bool:
        return self.num_iterations >= self.num_iterations_before_pop

    def __init__(self,
                 num_iterations_before_can_pop: int,
                 num_iterations_before_forced_flush: int
                 ):
        """
        :param num_iterations_before_can_pop: Number of context expressions before returning off stack is option
        :param num_iterations_before_forced_flush: Number of context expressions before you MUSt return off stack
        """
        self.num_iterations = 0
        self.num_iterations_before_pop = num_iterations_before_can_pop
        self.num_before_flush = num_iterations_before_forced_flush
    def __call__(self, action_logits: torch.Tensor)->torch.Tensor:
        """
        :param action_logits: The action logits, used to make the action probabilities.
            - Shape (..., 3)
        :return: The action probabilities
            - Shape (..., 3)
        """
        logit = action_logits.clone()
        if self.num_iterations >= self.num_before_flush:
            # We are in the forced flush state. Mask out the create subroutine
            # and maintain subroutine option. We can then only return from
            # subroutine
            logit[..., 1] = -1e9
            logit[..., 2] = -1e9
        action_probabilities = torch.softmax(action_logits, dim=-1)
        self.num_iterations += 1
        return action_probabilities


class ProbabilisticPointers:
    """
    The probabilistic pointer management class. Tracks and modifies
    the probabilistic pointers.

    Probabilistic pointers distribute the focus across stack levels, allowing
    actions like `enstack`, `no-op`, and `destack` to adjust how attention is
    split across the levels of the stack.

    It handles rolling of the pointers based on the action probabilities and
    ensures that any probability that rolls off the end (or beginning) of the
    stack is masked out and discarded instead of wrapping around.
    """
    @property
    def is_active(self) -> bool:
        """
        Returns false when the stacks have all become inactive
        :return: bool.
        """
        return torch.any(self.is_batch_active)

    @property
    def is_batch_active(self)->torch.Tensor:
        """
        Checks, for all batch dims, whether the indicated elements of the
        stack are still active.
        :return: A tensor indicating boolean truth values
        """
        return self.stack_probability <= self.activity_epsilon

    @property
    def stack_probability(self)->torch.Tensor:
        """
        Indicates the remaining probability in the stack.
        :return: A tensor of shape (...). Probability
        """
        return self.pointers.sum(dim=0)


    def __init__(self,
                 probabilistic_pointers: torch.Tensor,
                 activity_epsilon: float = 1e-4
                 ):
        """
        Initializes the probabilistic pointer class.

        :param probabilistic_pointers: The probabilistic pointers matching the
                                       situation. It has been assumed they were loaded already.
                                       Shape (stack_size, ...)
        :param activity_epsilon: The epsilon value to use when deciding if a stack is active
               or not.
        """
        self.pointers = probabilistic_pointers
        self.activity_epsilon = activity_epsilon

    def get(self)->torch.Tensor:
        """
        Gets the current probabilistic pointers
        :return: The probabilistic pointers
        """
        return self.pointers
    def create_scenario_pointers(self,
                                can_pop: bool
                                )->Tuple[torch.Tensor, torch.Tensor]:
        """
        Creates the probabilistic scenario pointers. These are pointers
        that are, hypothetically, pointing at the location we will
        end up at if we were to do a "return", "maintain", or "create"
        subroutine action.

        This will, in essence, end up telling us where we need to write.

        :param can_pop: A bool. If this is false, we are not allowed to pop off
                        the end of the stack, and instead probability would accumulate
        :return:
            - The scenario pointers, covering the three scenarios.
                - Shape (stack_size, ..., 3)
            - The lost probability
                - Only nonzero if can pop is true
                - Shape (...)
                - Indicates what probability was lost off the top of the stack
        """

        # In order to accomplish the creation of the scenario pointers,
        # basically the return action is made to point one higher in the stack
        # the maintenance pointer does not change, and the create pointers go
        # one deeper. Then there is some masking and accumulation behavior.

        pointers = []

        # Create the return pointers. We roll earlier into the stack,
        # then must handle any probability accumulation or compensation

        return_pointers = self.pointers
        return_pointers = return_pointers.roll(-1, dims=0)

        if can_pop:
            # The lost probability is what rolled off the beginning of the stack,
            # to the end. Fetch it, then mask that out.
            lost_probability = return_pointers[-1].clone()
            return_pointers[-1] = 0
        else:
            # The lost probability is none, as we cannot lose probability yet.
            # We instead accumulate whatever rolled off the end in slot zero,
            # basically moving the probability if it rolled off the end of the stack.
            lost_probability = torch.zeros_like(return_pointers[-1])
            return_pointers[0] += return_pointers[-1]
            return_pointers[-1] = 0
        pointers.append(return_pointers)

        # Handle the maintain subroutine operation. It, perhaps unsuprisingly, means
        # no change is made in pointer positioning.

        pointers.append(self.pointers)

        # Handle the create suproutine operation. In this case, we roll one deeper
        # into the stack. Also, since we cannot roll probability off the end of the stack,
        # we instead accumulate probability near the end and mask out the beginning.

        # Roll to go one deeper into the stack
        create_pointers = self.pointers.roll(1, dims=0)

        # Now, some probability rolled off the end of the stack and
        # ended up at the beginning. Move it back to the end, and mask
        # the beginning

        create_pointers[-1] += create_pointers[0]
        create_pointers[0] = 0

        # Append
        pointers.append(create_pointers)

        # Now, create the final return, and return

        return torch.stack(pointers, dim=-1), lost_probability

    def update_superposition(self,
                             action_probabilities: torch.Tensor,
                             scenario_pointers: torch.Tensor
                             ):
        """
        Updates the internal pointer superposition based on the
        scenario pointers, and the action probabilities. Basically,
        we end up doing a weighted sum across the probability distributions.

        Which still ends up with probability distributions.
        :param action_probabilities: The action probabilities. Shape (..., 3)
        :param scenario_pointers: The scenario pointers. Shape (stack_size, ..., 3)
        """
        self.pointers = torch.sum(scenario_pointers * action_probabilities.unsqueeze(0), dim=-1)

class SubroutineStackTracker:
    """
    Manages the subroutine stack and associated ideas.
    Also responsible for proposing new stack state options
    under the return from subroutine, maintain subroutine,
    and create subroutine computation branches.
    """
    def __init__(self,
                 stack: torch.Tensor,
                 logic: SubroutineLogicStub
                 ):
        self.stack = stack
        self.logic = logic

    def get(self, pointer_probabilities: torch.Tensor)->torch.Tensor:
        """
        Gets the subroutine context, based on the provided pointer probabilities.
        :param pointer_probabilities: The pointer probabilities. Shape (stack_depth, ...batch)
        :return: The expressed context.. Shape (...batch, ...custom)
        """

        # The pointer probabilities may not be expressed over the entire stack.
        # Unsqueeze it a bit if needed
        while pointer_probabilities.dim() < self.stack.dim():
            pointer_probabilities = pointer_probabilities.unsqueeze(-1)

        # Run weighting action, then add.
        return torch.sum(pointer_probabilities*self.stack, dim=0)
    def compute_scenario_outcomes(self)->torch.Tensor:
        """
        Computes the three parallel stack state outcomes for all
        depths in the stack. This basically returns every possible
        outcome. We will use probabilities to actually figure out
        how much each update matters.

        :return: The scenario outcomes.
            - Shape (stack_size, ..., 3)
            - for 3: "return", "maintain", or "create"
            - What we look like after each activity
        """
        outcomes: List[torch.Tensor] = []
        outcomes.append(self.logic.return_from_subroutine(self.stack))
        outcomes.append(self.logic.create_subroutine(self.stack))
        outcomes.append(self.logic.maintain_subroutine(self.stack))
        return torch.stack(outcomes, dim=-1)
    def integrate_update(self,
                         action_probabilities: torch.Tensor,
                         scenario_pointers: torch.Tensor,
                         update: torch.Tensor
                         ):
        """
        Integrates a given set of scenario and action information to
        produced the revised stack. Does this by running all possibilities
        in parallel then updating according to pointer weights.

        :param action_probabilities: The action probabilities
            - Indicates how strongly we thing the return, maintain, create action occurs,
              respectively, along 3
            - Shape (..., 3)
        :param scenario_pointers: The pointers we write to under the three scenarios
            - Shape (stack_size, ..., 3)
            - Again, return, maintain, create.
        :param update: A tensor we need to integrate into the stack at the proper position.
            - Shape (...)
        """

        # Begin by computing the outcomes for performing each scenario at
        # each stack level.

        scenario_options = self.compute_scenario_outcomes() #(stack_depth, ..., 3)

        # These options now have the update integrated into them

        scenario_options = self.logic.update_subroutine(update, scenario_options)

        # The actual stacks during each computation branch are an interpolation
        # between these outcomes, and how much we can actually write into
        # each position.

        scenario_stacks = self.stack*(1-scenario_pointers) + scenario_options*scenario_pointers

        # And finally, we merge these branches using the action probabilities

        self.stack = torch.sum(scenario_stacks*action_probabilities.unsqueeze(0), dim=-1)


class DifferentiableSubroutine:
    """
    A differentiable stack for handling subroutine creation, execution, and
    return, along with integration of subroutine updates into the subroutine
    stacks.

    Multiple stacks can be managed in parallel with their own unique create,
    maintain, return, and update logic. However, they all are syncronized
    in terms of a single set of probabilistic pointers that let us display only a single
    context when running the rest of the model.

    Updates must be provided in the same manner as the stack trackers were provided.
    """

    def __init__(self,
                 actions_manager: ActionsProbabilities,
                 probabilistic_pointers: ProbabilisticPointers,
                 subroutines: StackTree
                 ):
        """
        :param actions_manager: Creates action probabilities, and tracks iterations
        :param probabilistic_pointers: The probabilistic pointers
        :param subroutines: A Pytree stack containing the setup subroutine stacks.
        """
        self.actions_manager = actions_manager
        self.probabilistic_pointers = probabilistic_pointers
        self.subroutines = subroutines

    def get(self)->TensorTree:
        """
        Gets an expression of the subroutine context
        :return: The subroutine tensortree.
        """
        pointer_probabilities = self.probabilistic_pointers.get()
        def get_expressed_state(stack_tracker: SubroutineStackTracker)->torch.Tensor:
            return stack_tracker.get(pointer_probabilities)
        return parallel_pytree_map(get_expressed_state, self.subroutines)


    def update(self,
               action_logits: torch.Tensor,
               updates: TensorTree
               )->TensorTree:
        """
        Performs an update against the differentiable subroutine.
        Each individual expressed subroutine state is expected to have subroutine
        updates that will be created and expressed in the update tree, in the same
        way as was seen during the get action. These will be integrated according
        to the subroutine logic.

        :param action_logits: The action logits.
        :param updates: The updates for each subroutine state piece, in the
                        stackless space.
        :return: The lost subroutine state. Found by multiplying the updates by the
                 lost probability - these will never end up somewhere inside the stack.
        """

        # Perform the probabilistic computation actions.
        action_probabilities = self.actions_manager(action_logits)
        scenario_pointers, lost_probability = self.probabilistic_pointers.create_scenario_pointers(
                                                                    self.actions_manager.can_pop)
        self.probabilistic_pointers.update_superposition(action_probabilities, scenario_pointers)

        #Define and perform the actual state update.
        def perform_update(update: torch.Tensor,
                           stack_tracker: SubroutineStackTracker,
                           )->TensorTree:
            """
            Updates the subroutine stack trackers. Creates the lost state output trees
            :param update: The update to incorporate
            :param stack_tracker: The state tracker to incorporate it on
            :return: The lost update. Used later for accumulator action
            """

            # Perform the actual update action. Integration is performed
            # using the action probabilities and the scenario pointers
            stack_tracker.integrate_update(action_probabilities,
                                           scenario_pointers,
                                           update)

            # Now, go weight the update by the lost probability. That much
            # of the update went nowhere, and thus is viewed as the return
            # of the entire routine
            my_lost_probability = lost_probability
            while my_lost_probability.dim() < update.dim():
                my_lost_probability = my_lost_probability.unsqueeze(-1)
            return update*my_lost_probability

        outcome = parallel_pytree_map(perform_update, updates, self.subroutines)

        return outcome

class SubroutineSetupFactory:
    """
    A factory method to setup a differentiable subroutine
    from a pytree of subroutine stubs. In order to setup
    such a subroutine, basically, we need to make the
    probability pointers with stack and make each
    individual stub stack as well.
    """


class SubroutineDriver:
    """
    A core stack driver layer designed to allow the usage
    of the differentiable stack factory automatically with an
    internal core computation layer.
    """

    @classmethod
    def parallel_pytree_map(cls, func: Callable[..., Any], *pytrees: Any) -> Any:
        """
        Recursively applies a function to corresponding leaves of multiple pytrees with the same structure.

        Args:
            func (Callable[..., Any]): A function to apply to corresponding leaves of the pytrees.
            *pytrees (NestedTensor): Multiple pytrees with the same structure.

        Returns:
            NestedTensor: A new pytree with the function applied to corresponding leaves.
        """
        # Check if all pytrees are lists, tuples, or dicts
        if all(isinstance(pytree, list) for pytree in pytrees):
            return [cls.parallel_pytree_map(func, *elems) for elems in zip(*pytrees)]
        elif all(isinstance(pytree, tuple) for pytree in pytrees):
            return tuple(cls.parallel_pytree_map(func, *elems) for elems in zip(*pytrees))
        elif all(isinstance(pytree, dict) for pytree in pytrees):
            return {key: cls.parallel_pytree_map(func, *(pytree[key] for pytree in pytrees))
                    for key in pytrees[0]}
        else:
            # These are leaves, apply the function to them
            return func(*pytrees)

    def __init__(self,
                 d_model: int,
                 stack_depth: int,
                 core: SubroutineCore
                 ):
        self.actions = nn.Linear(d_model, 3)
        self.stack_factory = SubroutineStackFactory(stack_depth, d_model)
        self.core = core

    def initialize_stacks(self,
                          tensor: torch.Tensor,
                          state: TensorTree,
                          min_iterations_before_destack: int,
                          max_iterations_before_flush: int,
                          )->Tuple[DifferentiableSubroutineStack, StackTree]:
        """
        Initializes the stacks using the tensor and the state, for the subsequent
        computation.

        :param tensor: The tensor to initialize with
        :param state: The state to initialize with
        :param min_iterations_before_destack: The number of iterations to wait until stack removal is allowed
        :param max_iterations_before_flush: The maximum number of iterations before the stack is forced to flus
                                            its contents.
        :return:
            - The tensor stack
            - The state stack
        """
        setup_stack = lambda x: self.stack_factory(x, min_iterations_before_destack, max_iterations_before_flush)
        tensor_stack = self.stack_factory(tensor, min_iterations_before_destack, max_iterations_before_flush)
        state_stack = self.parallel_pytree_map(setup_stack, state)
        return tensor_stack, state_stack

    def initialize_accumulators(self,
                                tensor: torch.Tensor,
                                state: TensorTree
                                )->Tuple[torch.Tensor, TensorTree]:
        """
        Initialize the output accumulators as well.
        :param tensor: The tensor whose shape we accumulat
        :param state: The state we will accumulate
        :return: The setup accumulators
        """
        setup_accumulator = lambda x: torch.zeros_like(x)
        tensor_accumulator = torch.zeros_like(tensor)
        state_accumulator = self.parallel_pytree_map(setup_accumulator, state)
        return tensor_accumulator, state_accumulator

    def forward(self,
                tensor: torch.Tensor,
                max_computation_iterations: int,
                state: Optional[TensorTree] = None,
                min_iterations_before_destack: int = 0
                )->Tuple[torch.Tensor, TensorTree]:
        """
        Performs the stack drive action.

        :param tensor: The tensor of, presumably, embeddings to process
        :param max_computation_iterations:
            - The number of iterations to wait until the stack is forced to flush.
        :param state: Any state that is available.
        :param min_iterations_before_destack: AS it says
        :return: The embedding output, and any new state
        """
        # Setup state if needed
        if state is None:
            state = self.core.setup_state(tensor)
            assert state is not None


        # Setup features
        tensor_stack, state_stack = self.initialize_stacks(tensor, state,
                                                           min_iterations_before_destack,
                                                           max_computation_iterations)
        tensor_accumulators, state_accumulators = self.initialize_accumulators(tensor, state)

        # Drive solution. We
        while not tensor_stack.stack_empty:

            # Get the current embeddings, state
            get_from_diff_stack = lambda x : x.get()
            embeddings = tensor_stack.get()
            state = self.parallel_pytree_map(get_from_diff_stack, state_stack)

            # Run using core, and compute stack actions

            embeddings, state = self.core(embeddings, state)
            actions = self.actions(embeddings)

            # Update stacks, get outputs

            update_action = lambda stack, update : stack.update(actions, update)
            tensor_outcome = tensor_stack.update(actions, embeddings) # Tuple(probs, embeddings)
            state_outcome = self.parallel_pytree_map(update_action, state_stack, state)

            # Use outputs to update accumulators
            def accumulate_action(accumulator: torch.Tensor,
                                  update: Tuple[torch.Tensor, torch.Tensor]
                                  ):
                probs, embeddings = update
                accumulator += embeddings*probs.unsqueeze(-1)

            accumulate_action(tensor_accumulators, tensor_outcome)
            self.parallel_pytree_map(accumulate_action, state_accumulators, state_outcome)

        return tensor_accumulators, state_accumulators






