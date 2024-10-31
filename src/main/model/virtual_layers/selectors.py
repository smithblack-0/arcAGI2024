from abc import ABC, abstractmethod
from typing import Optional, Any, Tuple, Dict

import torch
from torch import nn

from src.main.model.virtual_layers import SelectionSpec, DropoutLogits
from src.main.model import registry

# Some general helper layers. Maybe I should put these elsewhere?


## Bank selector logic.
#
# There will be some helper functions, then
# actual bank selection mechanisms
##


def make_top_p_selection_mask(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """
    Selection mechanism for the top-p selection type, or nucleus
    selection. In this mode, we collect cases until the probablity
    mass exceeds a threshold, then no more. We form a mask that shows
    what elements we want to select.

    :param logits: The logits to select from. Shape (..., logits)
    :param top_p: The cumulative probability threshold for top-p selection.
    :return: The selected top k mask. Shape (..., logits). Bool of true means selected
    """
    # Basic sanity check
    if not 1.0 >= top_p >= 0.0:
        raise ValueError(f"Top p should have been between 0 and 1 inclusive. Given was {top_p}")

    # Create the default selection mask
    selection_mask = torch.zeros_like(logits, dtype=bool)

    # Skip further computation and return immediately
    # if top p is set to not catch anything. If not,
    # we activate our logits so we can do probability
    # mass sampling
    if top_p == 0.0:
        return selection_mask

    probabilities = torch.softmax(logits, dim=-1)

    # We must perform nucleus sampling. This is tricky
    # when vectorized. What we must do is sort the probabilities
    # in ascending order, then figure out when the cumulative
    # sum is under a threshold and mask out everything above it
    #
    # However, what we are ACTUALLY doing is figuring out what
    # the mask looks like in the sorted domain, then moving
    # that back into the unsorted mask.
    #
    # We perform a roll here to make sure that we are only considering
    # the probabilities selected so far.

    ordered_probabilities, sorting_index = probabilities.sort(dim=-1, descending=True)
    cumulative_probabilities = ordered_probabilities.cumsum(dim=-1)
    cumulative_probabilities[..., -1] = 0.0
    cumulative_probabilities = cumulative_probabilities.roll(dims=-1, shifts=1)
    cumulative_mask = cumulative_probabilities <= top_p

    # We now transfer the cumulative mask back into the selection
    # mask, in the designated order.
    selection_mask.scatter_(dim=-1, index=sorting_index, src=cumulative_mask)

    return selection_mask


def make_top_k_selection_mask(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """
    Selection mechanism to make a top k mask. Selects the
    top k elements, and returns a mask indicating what
    elements were selected.

    Returns a top k selection mask based on the provided logits
    :param logits: The logits to select a top k from. (..., logits)
    :param top_k: Integer, indicating how many to select.
    :return: The selected top k mask. Shape (..., logits). Bool of true means selected
    """
    if not top_k >= 0:
        raise ValueError(f"Top k should have been greater than or equal to 0. Given was {top_k}")
    if top_k > logits.size(-1):
        top_k = logits.size(-1)

    selection_mask = torch.zeros_like(logits, dtype=bool)
    if top_k > 0:
        # Only bother to actually compute the top k while the
        # mode is active. We get the indexes associated with
        # the top k, and mark those parts of the mask as active
        _, index = torch.topk(logits, k=top_k, dim=-1)
        src = torch.full_like(index, True, dtype=torch.bool)
        selection_mask.scatter_(dim=-1, index=index, src=src)
    return selection_mask


def make_random_selection_mask(logits: torch.Tensor, num: int) -> torch.Tensor:
    """
    Creates a selection mask with num elements selected randomly.

    :param logits: The logits to select a top k from. (..., logits)
    :param num: Integer, indicating how many to randomly select.
    :return: The selected top k mask. Shape (..., logits). Bool of true means selected
    """
    # Basic sanity checking
    if not num >= 0:
        raise ValueError(f"num of selected should have been greater than or equal to 0. Given was {num}")
    if num > logits.size(-1):
        num = logits.size(-1)

    # Create the default selection mask
    selection_mask = torch.zeros_like(logits, dtype=bool)

    # If you are not going to select ANY, just return the default mask
    if num == 0:
        return selection_mask

    # Select a certain number randomly from what remains.
    # We do this by creating a random matrix of the same
    # shape, sorting it, and slicing out the sections needed.
    # This ends up behaving like a vectorized torch.randperm

    random_weights = torch.rand_like(logits)
    randomized_indices = torch.argsort(random_weights, dim=-1)
    randomized_indices = randomized_indices[..., :num]

    src = torch.full_like(randomized_indices, True, dtype=torch.bool)
    selection_mask.scatter_(dim=-1, index=randomized_indices, src=src)

    # Return the selection
    return selection_mask


class AbstractBankSelector(nn.Module, ABC):
    """
    The `AbstractBankSelector` provides a framework for selecting parameter banks (virtual layers)
    from logits and dynamically building a `SelectionSpec`. It supports sparse selection mechanisms
    such as top-k, top-p (nucleus), and random sampling. This class serves as a base for subclasses
    that implement custom logic for generating logits used for bank selection.

    ---- Key Concepts ----
    - `select_logits`: This method generates the final `SelectionSpec`, combining any sparse selections
      such as top-k, top-p, or random sampling. It processes logits into selected indices and their associated
      probabilities. **Subclasses must call this method with their generated logits** to create the selection.
    - `SelectionSpec`: Defines the indices and probabilities for selecting virtual parameters from a bank.
    - `sparse_mode`: Controls how sparse selection is performed, using `top_k`, `top_p`, or `rand`. If none are
      specified, dense mode is used, where **all logits are included, but weighted by their probabilities**.

    ---- Usage ----
    - Subclasses **must implement the `forward` method** to generate logits that represent potential selections.
    - The `select_logits` method should be called within `forward` to convert generated logits into a `SelectionSpec`.
    - `top_k`, `top_p`, and `rand` are mutable fields and can be modified dynamically to adjust selection behavior
      at any point, without reinitializing the selector.

    ---- Initialization ----
    - `top_k`, `top_p`, and `rand` control how many logits are included in the selection:
        * `top_k`: Selects the highest k logits.
        * `top_p`: Uses nucleus sampling to select logits until the cumulative probability reaches p.
        * `rand`: Selects a random subset of logits.
      If all are inactive (set to zero), dense mode is used, where all logits contribute based on their probabilities.
    - `dropout_rate` is applied to the logits before selection, allowing for stochastic behavior during training.

    :param top_k: The number of top logits to select (optional, defaults to 0).
    :param top_p: The cumulative probability threshold for top-p selection (optional, defaults to 0.0).
    :param rand: The number of random logits to select (optional, defaults to 0).
    :param control_dropout: Dropout rate to apply to the logits during selection (optional, defaults to 0.0).
    """

    #TODO: Add bank selector metrics and, optionally, bank balancing.
    @property
    def is_dense(self) -> bool:
        """
        Tells us whether selection is operating in dense mode. It is dynamically
        computed in case someone changes the fields.
        """
        if not self.top_k == 0:
            return False
        if not self.top_p == 0.0:
            return False
        if not self.rand == 0:
            return False
        return True

    def select_logits(self, logits: torch.Tensor) -> SelectionSpec:
        """
        Creates a working selection spec out of the logits, based on the initialization
        conditions. A variety of configurations can be utilized, and this supports them
        all.

        :param logits: The logits to select from. Shape (..., logits)
        :return: The created selection spec
        """

        # Generate the logit index
        logit_index = torch.arange(
            logits.shape[-1],
            device=logits.device,
        )

        # Perform normalization
        logits = self.dropout(logits)

        # Detect dense logits, and handle them
        # separately

        if not self.is_dense:

            # Handle and combine all sparse selection mechanisms
            selection_mask = torch.zeros_like(logits, dtype=torch.bool)
            selection_mask = selection_mask | make_top_k_selection_mask(logits, self.top_k)
            selection_mask = selection_mask | make_top_p_selection_mask(logits, self.top_p)
            selection_mask = selection_mask | make_random_selection_mask(logits, self.rand)

            # The number of required logits is determined by counting up how many
            # selections are active, then keeping the maximum. This is used to initialize
            # a space to hold the selected logits. The space is initialized with highly
            # negative defaults, to ensure anything not transferred ends up masked

            num_required_logits = selection_mask.sum(dim=-1).max()
            final_logits = torch.full([*logits.shape[:-1], num_required_logits],
                                      fill_value=-1e+9,
                                      device=logits.device, dtype=logits.dtype)

            # In order to transfer the selections into the final logit feature, we need
            # both a source and a destination mask. The destination mask is made by sorting
            # the selection_mask, then retaining num_required_logits mask entries. This ensures
            # there is always the same number of active entries per batch, ensuring that transfer
            # will be successful. Note that logit order is not guaranteed.

            transfer_mask, transfer_index = torch.sort(selection_mask, dim=-1, descending=True)
            transfer_mask = transfer_mask[..., :num_required_logits]
            transfer_index = transfer_index[..., :num_required_logits]

            # Transfer into the final logit space. Also, use an arange
            # to get an associated index selection

            final_logits[transfer_mask] = logits[selection_mask]
            final_index = logit_index[transfer_index]

        else:
            # Handle dense mode operation. For now, this
            # consists of using the sparse logic.

            final_index = logit_index
            while final_index.dim() < logits.dim():
                final_index = final_index.unsqueeze(0)

            final_index = final_index.expand_as(logits)
            final_logits = logits

        # Finish up. Active the logits, then return the result
        probabilities = torch.softmax(final_logits, dim=-1)
        return SelectionSpec(final_index, probabilities)

    def __init__(self,
                 top_k: Optional[int] = None,
                 top_p: Optional[float] = None,
                 rand: Optional[int] = None,
                 control_dropout: Optional[float] = None
                 ):
        """
        The abstract bank selection process cna be defined here
        We can say how many sparsely to include in terms of top-p (nucleous),
        top-k, or randomly selected.

        Do note that not providing any specification for k, p, or rand puts
        us in dense mode, in which we do not reduce at all.

        :param top_k: The top k logits are included.
        :param top_p: Nucleus sampling is done to get the top p logits
        :param rand: This many logits are randomly selected
        :param control_dropout: The dropout rate to apply to the logits
        """
        super().__init__()

        # Data standardization
        if top_k is None:
            top_k = 0
        if top_p is None:
            top_p = 0.0
        if rand is None:
            rand = 0
        if control_dropout is None:
            control_dropout = 0.0

        # Setup
        self.top_k = top_k
        self.top_p = top_p
        self.rand = rand
        self.dropout = DropoutLogits(control_dropout)

    @abstractmethod
    def forward(self, embedding: torch.Tensor, state: Optional[torch.Tensor]) -> Tuple[SelectionSpec, Any]:
        """
        Abstract definition of the forward mechanism. This method accepts arbitrary content
        (args and kwargs) and returns a `SelectionSpec` along with any updated recurrent state.

        ---- Parameters ----

        :param args: Positional arguments to be used in the concrete implementation.
        :param kwargs: Keyword arguments to be used in the concrete implementation.
        :return: A tuple containing:
            - `SelectionSpec`: A structure that defines the selected bank indices and the associated probabilities.
            - `state`: The recurrent state, passed along for future calls. Will be invoked as kwargs
        """
        pass


selector_registry = registry.InterfaceRegistry[AbstractBankSelector]("BankSelector", AbstractBankSelector)


@selector_registry.register("LinearBankSelector")
class LinearBankSelector(AbstractBankSelector):
    """
    The `LinearBankSelector` is a simple bank selector that uses a single linear
    layer to generate logits for selecting parameter banks (virtual layers).

    This selector applies the given embedding to the linear layer, processes
    the logits using selection mechanisms (top-k, top-p, or random sampling),
    and returns a `SelectionSpec` for the selected virtual layers.

    ---- Key Features ----
    - Embedding-based selection: Uses the provided embedding to create logits in
      the bank space.
    - Sparse Selection: Applies top-k, top-p, or random sampling to select a subset
      of banks.
    - Simplest form of bank selection: One linear layer and no recurrent state.
    - Providing no sparse selection input places us in dense mode. This is computationally intense.
    """

    def __init__(self,
                 d_embedding: int,
                 bank_size: int,
                 top_k: Optional[int] = None,
                 top_p: Optional[float] = None,
                 rand: Optional[int] = None,
                 control_dropout: Optional[float] = None
                 ):
        """
        Initializes the `LinearBankSelector` with the given embedding size, bank size,
        and selection configuration.

        :param d_embedding: The size of the embeddings that will be provided.
        :param bank_size: The size of the bank selection to create.
        :param top_k: The number of top logits selected (optional, defaults to 0).
        :param top_p: The probability mass to select by (optional, defaults to 0.0).
        :param rand: The number of random logits to include (optional, defaults to 0).
        :param control_dropout: Logit dropout rate (optional, defaults to 0.0).
        """
        super().__init__(top_k, top_p, rand, control_dropout)
        self.projector = nn.Linear(d_embedding, bank_size)

    def forward(self, embedding: torch.Tensor, state: None) -> Tuple[SelectionSpec, None]:
        """
        Generates logits for selecting parameter banks based on the provided embedding
        and processes them into a `SelectionSpec` using the configured sparse selection
        mechanisms (top-k, top-p, or random sampling).

        :param embedding: The embedding to process for selecting banks.
        :param state: Unused
        :return:
            - The `SelectionSpec` containing the selected indices and probabilities.
            - `None` as there is no recurrent state in this implementation.
        """
        logits = self.projector(embedding)
        return self.select_logits(logits), None

@selector_registry.register("PseudoMarkovBankSelector")
class PseudoMarkovBankSelector(AbstractBankSelector):
    """
    The Pseudo Markov Bank Selector is a more sophisticated bank selector that
    utilizes persistent state and combines transition-based biases with immediate
    computational results to influence virtual layer selection.

    This class assumes that related virtual layers should be closely linked,
    and it aims to model this through a Markov-like process. The Markov biases,
    or logits, act as transition preferences between virtual layers. These biases
    are dynamically updated and applied to the selection process, promoting or
    demoting certain layers based on the last expressed selection - basically the last
    markov state.

    However, this is a "pseudo" Markov process because, while these biases exist,
    the immediate computation results (from embeddings or other inputs) also play
    a significant role. This means the model can adapt and override the Markov
    biases on-the-fly based on the current task, emphasizing or demoting specific
    virtual layers as needed. This combination of transition biases and current
    computations results in a more flexible and context-sensitive virtual layer
    selection process.

    ---- Fields ----
    - `markov_biases`: A set of trainable biases representing transition preferences
      between virtual layers. These are dynamically updated and applied in combination
      with the embedding-based logits to influence the selection of virtual layers.
    - `state`: The recurrent state passed between iterations, which tracks the current
      transition probabilities across virtual layers.
    """

    def __init__(self,
                 d_embedding: int,
                 bank_size: int,
                 top_k: Optional[int] = None,
                 top_p: Optional[float] = None,
                 rand: Optional[int] = None,
                 control_dropout: Optional[float] = None
                 ):
        """
        Initializes the layer with the given embedding size, bank size,
        and selection configuration.

        :param d_embedding: The size of the embeddings that will be provided.
        :param bank_size: The size of the bank selection to create.
        :param top_k: The number of top logits selected (optional, defaults to 0).
        :param top_p: The probability mass to select by (optional, defaults to 0.0).
        :param rand: The number of random logits to include (optional, defaults to 0).
        :param control_dropout: Logit dropout rate (optional, defaults to 0.0).
        """
        super().__init__(top_k, top_p, rand, control_dropout)
        self.d_model = d_embedding
        self.bank_size = bank_size

        # This bears a little explanation. It turns out
        # running a probability distribution through a linear
        # layer is basically equivalent to weighting a bunch of
        # markov biases by how active that state was. So we
        # can just use a linear projection for that.
        #
        # The embedding projector just works like normal, though.
        self.projector = nn.Linear(d_embedding, bank_size)
        self.markov_biases = nn.Linear(bank_size, bank_size)

    def forward(self,
                tensor: torch.Tensor,
                state: Optional[torch.Tensor] = None
                ) -> Tuple[SelectionSpec, Dict[str, torch.Tensor]]:
        """
        Runs the forward method for the pseudo markov process
        :param tensor: The tensor to build local influences from. Shape (..., d_model)
        :param state: A tensor representing the probabilities associated with each virtual layer (bank).
              This state is initialized as a softmax distribution over the banks, with the first
              bank receiving full probability (set to 1) on the first pass. The state biases
              the current selection towards related virtual layers based on past transitions.
              Shape (..., bank_size).
        :return:
            - The SelectionSpec
            - The dict with state in it
        """

        # Standardize data. If state was never setup,
        # it becomes a probability distribution set to
        # 1 at the first element
        if state is None:
            state = torch.zeros([*tensor.shape[:-1], self.bank_size], device=tensor.device, dtype=tensor.dtype)
            state[..., 0] = 1.0

        # Compute the logits. Then combine them with the transition biases.
        # Some options will likely become impossible due to the transition
        # biases

        logits = self.projector(tensor)
        logits = logits + self.markov_biases(state)

        # Activate and store the state. Get the selection. Return results
        state = {"state": torch.softmax(logits, dim=-1)}
        return self.select_logits(logits), state
