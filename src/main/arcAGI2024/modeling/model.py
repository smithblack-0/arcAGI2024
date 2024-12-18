"""
Core model builders and functions
"""
import functools
import time
import copy
import tqdm
import os
import textwrap
import shutil
import json

from concurrent import futures
from typing import Type
from typing import Any, List, Tuple, Dict, Union, Callable, Optional
import torch
from torch import nn
from torch.autograd import profiler
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict

from .decoder import RecurrentDecoder
from ..vocabulary import Vocabulary, AdditionalSpecialTokens
from ..base import (get_rng_state, set_rng_state, parallel_pytree_map,
                                      DeviceDtypeWatch, GradientSubstitutionEndpoint, TensorTree)
from ..losses import MainLossInterface, MemAccessLossInterface
from ..sampling import SamplingInterface
from ..grad_utils import AbstractGradientControl

@dataclass
class CoreConfig:
    """
    Config for a functioning decoder. What each parameter does
    is indicated in depth.

    --- Top level parameters ---
    :param num_layers:
    - Number of transformer layers.
    - Each is a DecoderLayer
    :param dropout_rate:
    - Dropout rate, within the primary recurrent model
    :param sublayers_dropout_rate:
    - Dropout rate, within sublayers rather than the core proces.
    :param decoder_flavor:
    - What kind of decoder to use.
    - Can currently use "fast" or "deep".

    --- DecoderLayer Parameters
    Everything here is occurring with respect to d_core.

    :param d_core:
    - Bottleneck rate. d_model is bottlenecked down to this dimension to save compute time
    - Much of the computation occurs at this rate
    :param d_hidden:
    - Size of the hidden layer.
    - Choose it with respect to d_core.
    :param d_address:
    - A smaller subset of d_core. Can be anything.
    - Represents the width of the attn memory addresses.
    - Not used when decoder flavor is fast.
    :param d_memory:
    - Represents the width of the memory value addresses.
    - Can be different.
    :param num_read_heads:
    - Number of heads used for the memory read process
    - Larger means more of the memory can be read from per step
    :param num_write_heads:
    - Number of heads used for the memory write process
    - Larger means more of the memory can be written to per step
    :param num_memories:
    - Number of independent memory states
    :param numeric_write_factor:
    - The maximum probability that can be written to a memory slot in one go.
    - Should be less than 1.0 for numeric reasons. 0.9 would probably be fine.
    - Numeric divergence can be reduced by making it smaller, at the cost of
      some fidelity.

    ---- final ---
    :param numeric_write_factor: The factor that controls the maximum commitment to writing the model can make
    :param dtype: The dtype
    :param device: The device
    :return: A setup RecurrentDecoder
    """
    # Primary specifications
    num_layers: int
    num_read_heads: int
    num_write_heads: int
    num_memories: int
    decoder_flavor: str

    # Helper specifics

    dropout_rate: float
    sublayers_dropout_rate: float

    # These have defaults.
    numeric_write_factor: Optional[float] = None
    d_core: Optional[int] = None
    d_memory: Optional[int] = None
    d_address: Optional[int] = None
    d_hidden: Optional[int] = None
    dtype: torch.dtype = None

    def save_to_folder(self, directory: Union[str, os.PathLike]):
        """
        Saves the config to the folder. The dtype and
        device are not saved.

        :param directory: The directory to put it at
        :return:
        """
        with open(os.path.join(directory, "decoder_config.json"), "w") as f:
            config = asdict(self)
            config['device'] = str(config['device'])
            config['dtype'] = str(config['dtype'])
            json.dump(config, f, indent=4)

    @classmethod
    def load_from_folder(cls,
                         directory: Union[str, os.PathLike],
                         ) -> 'CoreConfig':
        """
        Loads the model from the folder. The dtype and device
        have to provided, since they are not saved.

        :param directory: Directory to load from
        :return: The config.
        """
        with open(os.path.join(directory, "decoder_config.json"), "r") as f:
            config = json.load(f)
            config["dtype"] = torch.get_autocast_dtype(config['dtype'])
            config = cls(**config)
        return config


class CausalLMCore(nn.Module):
    """
    A central place in which all parameters which belong
    to a particular model can be placed. This will
    contain the vocabulary and the model feature.
    """

    @property
    def device(self) -> torch.device:
        return self._metainfo.device

    @property
    def dtype(self) -> torch.dtype:
        return self._metainfo.dtype

    @classmethod
    def build_model_using_config(cls,
                                 vocabulary: Vocabulary,
                                 config: CoreConfig
                                 ) -> 'CausalLMCore':
        """
        Builds a model using the provided config.
        :param vocabulary: The vocabulary structure to bind to.
        :param config: The config to use
        :return: The created CausalLMCore model.
        """
        config = copy.deepcopy(config)

        # Load the causal lm head
        device = torch.device("cpu")
        vocabulary = vocabulary.to(dtype=config.dtype, device=device)
        d_model = vocabulary.d_model

        # Standardize defaults
        if config.d_core is None:
            config.d_core = d_model // 8
        if config.d_address is None:
            config.d_address = config.d_core // 4
        if config.d_memory is None:
            config.d_memory = config.d_core
        if config.d_hidden is None:
            config.d_hidden = config.d_core * 4
        if config.dtype is None:
            config.dtype = torch.float32
        if config.numeric_write_factor is None:
            config.numeric_write_factor = 0.999

        # Setup the model for training
        decoder = build_decoder(
            d_model,
            config.num_layers,
            config.decoder_flavor,
            config.d_core,
            config.d_hidden,
            config.d_address,
            config.d_memory,
            config.num_read_heads,
            config.num_write_heads,
            config.num_memories,
            config.dropout_rate,
            config.sublayers_dropout_rate,
            config.numeric_write_factor,
            dtype=config.dtype,
            device=device
        )

        # Return instance
        return cls(vocabulary, decoder, config)

    def rebase_model_onto_vocabulary(self, vocabulary: Vocabulary) -> 'CausalLMCore':
        """
        Rebases the model to interface with a different vocabulary struct.
        Note this will require fine tuning or even pretraining a bit.

        But most of the learning should be preserved
        :param vocabulary: The vocabulary to base us on now
        :return: The new casual lm core
        """
        decoder = self.decoder.rebuild_at_different_width(vocabulary.d_model)
        return CausalLMCore(vocabulary, decoder, self.config)

    def save_to_folder(self,
                       directory: Union[str, os.PathLike]
                       ):
        """
        Saves the causal core to an indicated directory
        :param directory_name: The directory to save to. Will be created if needed
        """
        if os.path.isdir(directory):
            shutil.rmtree(directory)
        os.makedirs(directory, exist_ok=True)

        self.vocabulary.save_pretrained_vocabulary(directory)
        torch.save(self.decoder, os.path.join(directory, "decoder.pt"))
        self.config.save_to_folder(directory)

    @classmethod
    def load_from_folder(cls,
                         directory: Union[str, os.PathLike],
                         ) -> 'CausalLMCore':
        """
        Loads the saved causal lm core file from the given directory.
        :param directory: The directory to load from
        :return: A setup CausalLMCore
        """
        decoder = torch.load(os.path.join(directory, "decoder.pt"))
        vocabulary = Vocabulary.load_pretrained_vocabulary(directory)
        config = CoreConfig.load_from_folder(directory, dtype, device)

        return cls(vocabulary, decoder, config)

    def __init__(self,
                 vocabulary: Vocabulary,
                 decoder: RecurrentDecoder,
                 config: CoreConfig,
                 device: Optional[torch.device] = None,
                 dtype: Optional[torch.dtype] = None,
                 ):
        super().__init__()
        assert decoder.d_model == vocabulary.d_model, "Decoder and vocabularies had different widths."
        self._metainfo = DeviceDtypeWatch(device=device, dtype=dtype)
        self.vocabulary: Vocabulary = vocabulary.to(device=device, dtype=dtype)
        self.decoder: RecurrentDecoder = decoder.to(device=device, dtype=dtype)
        self.config = config

    def __reduce__(self):
        msg = """
        Should not use pickle or torch.save to save the model.
        Instead, invoke CausalLMCore.save_to_folder(directory),
        and load with CausualLMCore.load_from_folder(directory)
        """
        msg = textwrap.dedent(msg)
        raise NotImplementedError(msg)


##
#
# Begin defining the core mechanisms. This is the trainer
#
##



class AbstractTrainerCore(nn.Module, ABC):
    """
    The abstract definition of the trainer core the following
    trainer can work. Sometimes, compiling might be required,
    hence this mechanism.
    """
    __trainer_cores: Dict[str, Type['AbstractTrainerCore']] = {}

    def __init_subclass__(cls, **kwargs):
        if issubclass(cls, AbstractTrainerCore):

            cls.__trainer_cores[cls.__name__] = cls
        super().__init_subclass__(**kwargs)

    def __init__(self, core: CausalLMCore, **init_kwargs):
        super().__init__()
        self.core = core
        self.init_kwargs = init_kwargs

    def save_to_folder(self, directory: Union[str, os.PathLike]):
        """
        Saves the trainer core to a directory, allowing resumption
        of later training

        :param directory: The directory to save to
        """
        if os.path.isdir(directory):
            shutil.rmtree(directory)
        os.makedirs(directory, exist_ok=True)

        with open(os.path.join(directory, "trainer_core_config.txt"), "w") as f:
            config = {"core_name" : self.__name__,
                      "init_kwargs" : self.init_kwargs
                      }
            json.dump(config, f)
        self.core.save_to_folder(directory)
    @classmethod
    def load_from_folder(cls,
                         directory: Union[str, os.PathLike],
                         device: torch.device,
                         ) -> 'AbstractTrainerCore':
        with open(os.path.join(directory, "trainer_core_config.txt"), "r") as f:
            config = json.load(f)
        core = CausalLMCore.load_from_folder(directory)
        core = core.to(device=device)
        subclass = cls.__trainer_cores[config["core_name"]]
        return subclass(core, **config["init_kwargs"])

    @abstractmethod
    def embed(self, token: torch.Tensor) -> torch.Tensor:
        """
        Embeds a token
        :param token: The token to embed. Shape (...)
        :return: The embedded token. shape (..., d_model)
        """

    @abstractmethod
    def logits(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Takes embeddings, and produces logits out of them
        :param embedding: The embedding. Shape (..., d_model)
        :return: Shape (..., num_logits)
        """

    @abstractmethod
    def create_state(self, batch_shape: torch.Size) -> List[DeepMemoryState]:
        """
        Sets up the recurrent state bound
        to a particular batch shape

        :param batch_shape: The batch shape to match
        :return: A list of memory states. One for each layer.
        """

    @abstractmethod
    def reverse(self,
                embedding: torch.Tensor,
                batch_mask: torch.Tensor,
                next_memories: List[DeepMemoryState]
                ) -> Tuple[Tuple[torch.Tensor, List[DeepMemoryState]], List[DeepMemoryState]]:
        """
        Runs the reverse process. This means figuring out the
        previous memory states and setting them up for gradient
        accumulation. And of course returning the final output

        :param embedding: The input embedding. Whatever it might be
        :param batch_mask: Shape (...). Indicates whether memory can be updated. True means No.
        :param next_memories: The memories from the NEXT step
        :return:
        - Tuple:
            - The final embedding, ready for usage in logits.
            - The memory states for this timestep. It has a graph. We need to insert gradients here
        - The memory from the last timestep. Setup to accumulate gradients and continue the chain.
        """

    @abstractmethod
    def forward(self,
                embedding: torch.Tensor,
                batch_mask: torch.Tensor,
                previous_memories: List[DeepMemoryState]
                ) -> Tuple[torch.Tensor, List[DeepMemoryState]]:
        """
        The forward mechanism. Performs a forward pass through the model.
        This will usually occur without gradients.
        :param embedding: The embedding being processed
        :param batch_mask: Shape (...). Indicates whether memory can be updated. True means No.
        :param previous_memories: The memory states from the last timestep
        :return:
        - The output. Same whether forward or backwards
        - The memory states for the next timestep.
        """

    @abstractmethod
    def release_memory(self, *items: TensorTree):
        """
        Releases memory however is needed
        :param items: The items whose memory is being released
        """


class StandardTrainerCore(AbstractTrainerCore):
    """
    The normal trainer core
    """

    def __init__(self, core: CausalLMCore, **unused_kwargs):
        super().__init__(core, **unused_kwargs)

    def embed(self, token: torch.Tensor) -> torch.Tensor:
        """
        Embeds a token
        :param token: The token to embed. Shape (...)
        :return: The embedded token. shape (..., d_model)
        """
        return self.core.vocabulary.embeddings(token)

    def logits(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Takes embeddings, and produces logits out of them
        :param embedding: The embedding. Shape (..., d_model)
        :return: Shape (..., num_logits)
        """
        return self.core.vocabulary.logit_projector(embedding)

    def create_state(self, batch_shape: torch.Size) -> List[DeepMemoryState]:
        """
        Sets up the recurrent state bound
        to a particular batch shape

        :param batch_shape: The batch shape to match
        :return: A list of memory states. One for each layer.
        """
        return self.core.decoder.create_state(batch_shape)

    def reverse(self,
                embedding: torch.Tensor,
                batch_mask: torch.Tensor,
                next_memories: List[DeepMemoryState]
                ) -> Tuple[Tuple[torch.Tensor, List[DeepMemoryState]], List[DeepMemoryState]]:
        """
        Runs the reverse process. This means figuring out the
        previous memory states and setting them up for gradient
        accumulation. And of course returning the final output

        :param embedding: The input embedding. Whatever it might be
        :param batch_mask: Shape (...). Indicates whether memory can be updated. True means No.
        :param next_memories: The memories from the NEXT step
        :return:
        - Tuple:
            - The final embedding, ready for usage in logits.
            - The memory states for this timestep. It has a graph. We need to insert gradients here
        - The memory from the last timestep. Setup to accumulate gradients and continue the chain.
        """
        return self.core.decoder.reverse(embedding, batch_mask, next_memories)

    def forward(self,
                embedding: torch.Tensor,
                batch_mask: torch.Tensor,
                previous_memories: List[DeepMemoryState]
                ) -> Tuple[torch.Tensor, List[DeepMemoryState]]:
        """
        The forward mechanism. Performs a forward pass through the model.
        This will usually occur without gradients.
        :param embedding: The embedding being processed
        :param batch_mask: Shape (...). Indicates whether memory can be updated. True means No.
        :param previous_memories: The memory states from the last timestep
        :return:
        - The output. Same whether forward or backwards
        - The memory states for the next timestep.
        """
        return self.core.decoder(embedding, batch_mask, previous_memories)

    def release_memory(self, *items: TensorTree):
        """
        Releases memory however is needed
        :param items: The items whose memory is being released
        """

        def release_tensor(tensor: torch.Tensor):
            del tensor

        parallel_pytree_map(release_tensor, items)
        for item in items:
            del item
        del items


class Logger:
    """
    A logger that uses an existing executor to perform asynchronous logging.
    """

    def __init__(self,
                 executor: futures.Executor,
                 terminal_callback: Callable[[str], None],
                 metric_callback: Callable[[Dict[str, Any]], None]):
        self.executor = executor
        self.terminal_callback = terminal_callback
        self.metric_callback = metric_callback

    def update_terminal_status(self, message: str):
        """
        Submits a terminal update task to the executor.
        """
        self.executor.submit(self.terminal_callback, message)

    def commit_metrics(self, metrics: Dict[str, Any]):
        """
        Submits a metrics update task to the executor.
        """
        self.executor.submit(self.metric_callback, metrics)


class ForwardPassProgress:
    def __init__(self,
                 total_tokens: int,
                 batch_width: int,
                 verbose: bool,
                 logger: Logger):
        """
        Context manager for tracking forward pass progress with tqdm.

        Args:
            total_tokens (int): The total number of tokens to be processed.
            batch_width (int): Number of tokens processed per step (batch size).
            verbose (bool): If False, disables progress tracking (acts as a no-op).
            terminal_callback: Where to send the message string.
        """
        self.total_tokens = total_tokens
        self.n = 0
        self.batch_width = batch_width
        self.start_time = None
        self.elapsed_time = None
        self.verbose = verbose
        self.logger = logger

    def update(self, cumulative_loss: float, total_correct: float, total_examined: float):
        """
        Update the cumulative loss, tokens per second, and progress count.
        """

        if not self.verbose:
            return  # No-op if verbose is False
        self.n += 1
        elapsed_time = time.time() - self.start_time
        tokens_per_second = float(self.n * self.batch_width) / elapsed_time
        accuracy = total_correct / total_examined if total_examined > 0 else 0

        postfix = {"forward_loss": f"{cumulative_loss:.4f}",
                   "tokens_per_second": f"{tokens_per_second:.4f}",
                   "running_accuracy": f"{accuracy:.4f}",
                   }

        progress_string = tqdm.tqdm.format_meter(
            n=self.n,
            total=self.total_tokens,
            elapsed=time.time() - self.start_time,
            postfix=postfix
        )
        self.logger.update_terminal_status(progress_string)

    def __enter__(self):
        # Initialize tqdm progress bar if verbose is True
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Store elapsed time if progress bar was initialized
        self.elapsed_time = time.time() - self.start_time


class ReversePassProgress:
    def __init__(self,
                 total_tokens: int,
                 batch_width: int,
                 verbose: bool,
                 logger: Logger
                 ):
        """
        Context manager for tracking reverse pass progress with tqdm.

        Args:
            total_tokens (int): The total number of tokens to be processed.
            batch_width (int): Number of tokens processed per step (batch size).
            verbose (bool): If False, disables progress tracking (acts as a no-op).
            terminal_callback: Where to send the message string.
        """
        self.n = 0
        self.total_tokens = total_tokens
        self.batch_width = batch_width
        self.verbose = verbose
        self.start_time = None
        self.logger = logger

    def update(self, loss: float, numeric_divergence: float, numeric_percent_error: float):
        """
        Update the cumulative loss, tokens per second, progress count, and optional metrics.

        Args:
            loss (float): Incremental loss to add to total reverse loss.
            numeric_divergence (float, optional): Value for numeric divergence, updated if provided.
            numeric_percent_error (float, optional): Value for numeric percent error, updated if provided.
        """
        if not self.verbose:
            return  # No-op if verbose is False

        self.n += 1

        # Calculate tokens per second
        elapsed_time = time.time() - self.start_time
        tokens_per_second = float(self.n * self.batch_width) / elapsed_time

        # Create the postfix
        postfix = {
            "reverse_loss": f"{loss:.4f}",
            "tokens_per_sec": f"{tokens_per_second:.2f}",
            "numeric_error": f"{numeric_divergence:.4f}",
            "numeric_percent_error": f"{numeric_percent_error:.4f}"
        }

        # Create the progress string
        progress_string = tqdm.tqdm.format_meter(
            n=self.n,
            total=self.total_tokens,
            elapsed=time.time() - self.start_time,
            postfix=postfix
        )
        self.logger.update_terminal_status(progress_string)

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.elapsed_time = time.time() - self.start_time


# Main trainer
class CausalLMTrainer(nn.Module):
    """
    A model designed specifically to exist in a training loop, and
    act to train the model core. That core can then be passed in
    to another generation instance for use.

    ---- sharp bits ----

    Be warned that a "reverse pass" is NOT the same as a "backwards pass".
    While it is true backwards passes happen during the reverse pass,
    there are also graph building mechanisms that occur.

    ---- differences and needs ----

    Due to the unusual nature of the model that has been developed
    - namely that we have a reverse pass to avoid having to hold
    the entire graph the entire time - a lot of functionality that
    would normally be held in a training loop instead must lie in here.

    Lets consider what a normal training loop for a causalLM would look
    like. You pass the embeddings in to get output embeddings. You
    then turn those outputs into logits, under a next-token-prediction
    assumption. You train using your loss in one go. You can schedule
    your training rate by adjusting the strength of the loss. This would
    be the normal way to train the model. Underlying this whole process
    is a reliance on the cpu or gpu to hold the graph and it's activations
    during training. This works when dealing with only a few hundred
    tokens.

    In this model, however, that standard paradigm breaks down.
    This is a recurrent model, and recurrent models train optimally
    under very large batch sizes, and over long sequences. Unfortunately,
    that would very quickly saturate the GPU memory.

    Instead, we can imagine the trainer  as computing the loss with a single token at a time,
    gathering gradients on "recurrent memories", and then manually propogating updates from
    those memories back in time. This is done in a forward pass, then a reverse pass. It
    also means there are some differences to the standard paradigm.

    #1): You pass in the targets, and pass in the loss function. The trainer decides where and when to use it,
         since it needs to apply the losses per token.
    #2): You also need to pass in your schedule parameters, for similar reasons. These will be passed into
         the schedule interface for the two losses.
    #3): Because this is a recurrent mechanism, there is also a recurrence loss based on how
         effectively the recurrent state has been used. Currently, it is a mem access loss. This
         also has a scheduling rate.

    The way it works under the hood is running a forward pass that produces reversable memories. Then we
    get gradients and start performing a backwards pass, reversing the memories along the way.
    numerical checkpointing is used to ensure numeric divergence does not cause huge issues, since
    in undoing the forward pass numeric differences might develop.

    ---- usage ----

    This layer would be placed within a training loop.
    Something like
    model_core = ...make model cor
    optim = optim.SGD(model_core.parameters())
    trainer = CausalLMTrainer(....)
    for tokens, targets, mask in batches:
        trainer(tokens, targets, mask)
        optim.step()
        optim.zero_grad()

    Observe how the optim exists outside the trainer, but almost everything
    else is inside it.
    """

    # Maintainers, be aware we make heavy use of parallel_pytree_map
    # in this function. It is a specialized function for working
    # with pytrees, even extended ones, much like jax pytree mapping.
    #
    # See base for details, but basically it takes parallel pytrees,
    # walks through the branches until reaching tensor trees, and then
    # calls the map function with all found leaves and rebuilds the tree.

    def __init__(self,
                 trainer_core: AbstractTrainerCore,
                 main_loss: MainLossInterface,
                 mem_loss: MemAccessLossInterface,
                 gradient_normalization: AbstractGradientControl,
                 numeric_cache_rate: int = 1000,
                 save_cached_to_cpu: bool = True,
                 verbose: bool = False,

                 ):
        super().__init__()

        self.core = trainer_core
        self.main_loss = main_loss
        self.mem_loss = mem_loss
        self.gradient_normalization = gradient_normalization
        self.numeric_cache_rate = numeric_cache_rate
        self.save_cached_to_cpu = save_cached_to_cpu
        self.mse_error = nn.MSELoss(reduction='mean')
        self.verbose = verbose

    def get_model_core(self) -> CausalLMCore:
        """
        Returns the model core in it's current state.
        :return: nn.Module
        """
        return self.core.core

    @staticmethod
    def save_to_cpu(tensor: torch.Tensor):
        with torch.no_grad():
            tensor = tensor.to(device=torch.device("cpu"))
            tensor = tensor.pin_memory()
        return tensor

    @staticmethod
    def load_from_cpu(tensor: torch.Tensor, device: torch.device):
        with torch.no_grad():
            tensor = tensor.to(device=device)
        return tensor

    def run_forward_pass(self,
                         tokens: torch.Tensor,
                         targets: torch.Tensor,
                         batch_mask: torch.Tensor,
                         main_schedule: float,
                         memories: List[DeepMemoryState],
                         logger: Logger,
                         ) -> Dict[str, Any]:
        """
        Runs the model forward pass. Records numeric metrics, random
        seeds, and other similar details as we go.

        :param tokens: The tokens to process. Shape (batch_size, items, d_model)
        :param targets: The target. Shape (batch_size, items)
        :param batch_mask: The batch mask. Shape (batch_size, items). True indicated padding.
        :param main_schedule: The weight for the main loss. Used when gathering the forward loss
        :param memories: The existing memories
        :param logger: The logging callback container.
        :return:
        - A dictionary containing a cache of features used for the backwards pass. In specific,
          it contains
            - memories: The final recurrent memories after the forward pass
            - rng_states: Rng states per token
            - numeric_cache: Cached numeric memories for stability and metrics.
            - forward_loss: The loss calculated during the forward pass.
        """
        num_tokens = tokens.shape[-1]
        batch_width = tokens.shape[0]
        # Setup the various required caches and metrics

        rng_states = []
        numeric_caches = []
        num_correct = 0.0
        num_processed = 0.0

        # Perform the forward pass to gain access to the
        # memories I shall need. This consists of recurrently
        # updating again and again. We discard the final state

        forward_loss = torch.tensor(0.0, device=tokens.device, dtype=tokens.dtype)
        with (torch.no_grad(), profiler.record_function("train_step: forward pass"),
              ForwardPassProgress(num_tokens,
                                  batch_width,
                                  self.verbose,
                                  logger) as progress
              ):
            for i in range(num_tokens):
                # Get the features
                embedding = self.core.embed(tokens[..., i])
                target = targets[..., i]
                mask = batch_mask[..., i]

                # Run forward pass
                rng_state = get_rng_state(embedding.device)
                output_embedding, memories = self.core.forward(embedding, mask, memories)

                # Compute loss. We use this to monitor numeric divergence, among
                # other things. However, it will not be used in backprop
                logits: torch.Tensor = self.core.logits(output_embedding)
                case_loss = self.main_loss(logits, target, main_schedule)
                forward_loss = forward_loss + case_loss

                # Handle accuracy metric

                num_correct += (logits.argmax(dim=-1) == target).sum()
                num_processed += (~mask).sum()

                # Integrate rng into cache. In order to reproducibly
                # run the reverse pass the same way as the forward
                # pass, we must cache the rng state at the beginning.
                #
                # This will be restored on the backwards pass.

                rng_states.append(rng_state)

                # Integrate numeric checkpoints into the cache
                #
                # This includes moving it to the cpu if relevant
                if i % self.numeric_cache_rate == 0:
                    if self.save_cached_to_cpu:
                        saved_memories = parallel_pytree_map(self.save_to_cpu, memories)
                    else:
                        saved_memories = memories
                    package = i, saved_memories
                    numeric_caches.append(package)
                else:
                    numeric_caches.append(None)

                progress.update(forward_loss, float(num_correct), float(num_processed))

                # Release anything I no longer need
                self.core.release_memory(embedding,
                                         target,
                                         mask,
                                         output_embedding,
                                         case_loss,
                                         )

        # Return the results
        results = {}
        results["memories"] = memories
        results["rng_states"] = rng_states
        results["numeric_cache"] = numeric_caches
        results["forward_loss"] = forward_loss
        results["forward_time"] = progress.elapsed_time
        results["accuracy"] = float(num_correct / num_processed)
        return results

    def setup_reverse_pass(self,
                           access_schedule: Optional[float],
                           cache_dict: Dict[str, Any],
                           device: torch.device,
                           ) -> Tuple[List[DeepMemoryState], torch.Tensor]:
        """
        Sets up state to be used in the reverse pass.

        This includes setting up the memories to be used
        in the backwards pass, running the backprop
        for the mem access loss, and setting aside a memory
        state for later.

        :param access_schedule: The scheduling details for the access loss.
        :param cache_dict: the results of run forward pass
        :param device: The device under consideration. Used to track seeds.
        :return: The final memory state, to be returned
        :return: The final RNG to put us in.
        """
        with profiler.record_function("train_step: Setup for reverse pass"):
            start_time = time.time()
            memory_state: List[DeepMemoryState] = cache_dict["memories"]

            # Make a copy pf the final rng state,
            # to restore it later. Also copy the final memories
            final_rng = get_rng_state(device)
            final_memories = parallel_pytree_map(lambda x: x.clone().detach(), memory_state)

            # Perform modifications to the final memory state, readying
            # it for backpropogation and gradient accumulation. This
            # will accumulate grads on the tensor now
            def enable_backprop_accumulation(tensor: torch.Tensor) -> torch.Tensor:
                tensor = tensor.detach()
                tensor.requires_grad_(True)
                tensor.retain_grad()
                return tensor

            memory_state = parallel_pytree_map(enable_backprop_accumulation, memory_state)

            # Run the backwards pass concerning the write distribution
            # loss.
            with profiler.record_function("train_step: Setup Backpropagation"):
                device = memory_state[0].write_probability_mass.device
                dtype = memory_state[0].write_probability_mass.dtype

                loss = torch.tensor(0.0, device=device, dtype=dtype)
                for memory in memory_state:
                    loss += self.mem_loss(memory.write_probability_mass, access_schedule)
                loss.backward()

            # Store the revised memories, and the beginnings of
            # the loss metric.

            cache_dict["memories"] = memory_state
            cache_dict["forward_loss"] += loss.detach()
            cache_dict["reverse_loss"] = loss.detach()
            cache_dict["setup_time"] = time.time() - start_time

        return final_memories, final_rng

    def run_reverse_pass(self,
                         tokens: torch.Tensor,
                         targets: torch.Tensor,
                         batch_mask: torch.Tensor,
                         loss_schedule: Optional[float],
                         cache_dict: Dict[str, Any],
                         logger: Logger
                         ) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        Runs the reverse pass basically. This includes most of the
        backpropagation algorithm, alongside managing certain metrics.

        :param tokens: The tokens to process. Shape (..., items, d_model)
        :param targets: The target. Shape (..., items)
        :param batch_mask: The batch mask. Shape (batch_size, items). True indicated padding.
        :param loss_schedule: The schedule weight on the main loss functionm
        :param cache_dict: The main cache, containing memories
        :param logger: The logging class

        We will dump the metrics into the logging callback
        """
        num_tokens = tokens.shape[-1]
        batch_width = tokens.shape[0]
        numeric_percent_error = 0.0
        numeric_error = 0.0

        load_from_cpu = functools.partial(self.load_from_cpu, device=tokens.device)

        with (profiler.record_function("train_step: Reverse pass"),
              ReversePassProgress(num_tokens,
                                  batch_width,
                                  self.verbose,
                                  logger)
              as progress):
            # Setup states, and grad cache containers
            memories: List[DeepMemoryState] = cache_dict["memories"]
            loss_metric = cache_dict["reverse_loss"].clone().detach()

            def get_mem_grads(tensor: torch.Tensor) -> torch.Tensor:
                if tensor.grad is not None:
                    return tensor.grad.detach()
                return tensor.grad

            mem_grads = parallel_pytree_map(get_mem_grads, memories)

            # Run iteration and backwards pass
            #
            # Make sure all tensors are detached too
            for i in reversed(range(num_tokens)):

                # Get the features
                with torch.no_grad():
                    token = tokens[..., i].clone().detach()
                    target = targets[..., i]
                    mask = batch_mask[..., i]

                embedding = self.core.embed(token)

                # Manage RNG.
                #
                # Restore the original rng state, and
                # make sure to move the cached state back to the gpu
                # before usage if needed.
                rng_state = cache_dict["rng_states"].pop()
                set_rng_state(rng_state, embedding.device)

                # Numeric caching needs to be resolved here.
                #
                # This basically only has an effect if there was previously
                # a numeric cache inserted at this embedding during the forward
                # pass.
                #
                # We have two primary tasks. First, we need to swap
                # out the cached numeric stability memory for the provided one.
                # Second, we will measure how far the predictions have diverged
                # by running predictions with both memories, then comparing
                # the distributions.

                numeric_cache = cache_dict["numeric_cache"].pop()
                if numeric_cache is not None:
                    with torch.no_grad():
                        # Get forward statistics ready to go.
                        entry_num, forward_memories = numeric_cache
                        if self.save_cached_to_cpu:
                            forward_memories = parallel_pytree_map(load_from_cpu, forward_memories)

                        # Compute the numeric divergence
                        case_numeric_error = 0.0
                        case_numeric_percent_error = 0.0

                        def compute_mse_error(memory: torch.Tensor, actual_memory: torch.Tensor):
                            nonlocal case_numeric_percent_error
                            nonlocal case_numeric_error

                            numeric_divergence = self.mse_error(memory, actual_memory)
                            percent_error = numeric_divergence / (actual_memory.mean() + 1e-4)

                            case_numeric_error = max(numeric_divergence, case_numeric_error)
                            case_numeric_percent_error = max(percent_error, case_numeric_percent_error)

                        parallel_pytree_map(compute_mse_error, memories, forward_memories)
                        numeric_percent_error += case_numeric_percent_error
                        numeric_error += case_numeric_error

                        # Replace the memories with the stored numeric memories. Transfer the
                        # gradients so backprop continues to work
                        memories = forward_memories

                        # Release anything claimed within the block

                # Perform the loss computation. Then store the metrics
                with profiler.record_function("train_step: computing loss"):
                    (output_embedding, next_memory), last_memory = self.core.reverse(embedding, mask, memories)
                    logits = self.core.logits(output_embedding)
                    loss = self.main_loss(logits, target, loss_schedule)
                    loss_metric += loss.detach()

                # Integrate the gradients into the memories
                # locations as manually injected gradients,
                # and combine the endpoint triggers into the losses
                def integrate_endpoints(memory_tensor: torch.Tensor,
                                        memory_grad_tensor: torch.Tensor
                                        ) -> torch.Tensor:

                    nonlocal loss
                    if memory_grad_tensor is not None:
                        loss += GradientSubstitutionEndpoint.apply(memory_tensor,
                                                                   memory_grad_tensor)

                parallel_pytree_map(integrate_endpoints, next_memory, mem_grads)

                # Run actual backpropagation.
                with profiler.record_function("train_step: Backpropagation"):
                    try:
                        loss.backward()
                    except Exception as err:
                        print("Beginning state dump")
                        with open('dump.txt', 'w') as f:
                            f.write(f'loss {loss}\n')

                            def write_it(tensor: torch.Tensor):
                                if tensor is not None:
                                    f.write(f'max {tensor.max()}\n')
                                    f.write(str(tensor))

                            parallel_pytree_map(write_it, memories)
                            parallel_pytree_map(write_it, mem_grads)
                        raise err

                # Advance to the prior memory. Rescale and normalize if needed
                mem_grads = parallel_pytree_map(get_mem_grads, last_memory)
                mem_grads = self.gradient_normalization(mem_grads)
                memories = parallel_pytree_map(lambda x: x.clone().detach(), last_memory)

                # Explicitly clear out tensors and detach graphs
                # on ANYTHING that actually had a graph attached.
                self.core.release_memory(
                    memories,
                    next_memory,
                    embedding,
                    logits,
                    loss,
                    mem_grads,
                    numeric_cache,
                    last_memory,
                )

                # Update progress when verbose
                progress.update(loss_metric, numeric_error, numeric_percent_error)

            metrics = {
                "forward_loss": cache_dict["forward_loss"],
                "reverse_loss": loss_metric,
                "accuracy": cache_dict["accuracy"],
                "numeric_percent_error": numeric_percent_error,
                "numeric_error": numeric_error,

                "forward_time": cache_dict["forward_time"],
                "setup_time": cache_dict["setup_time"],
            }
        metrics["reverse_time"] = progress.elapsed_time
        metrics["total_time"] = cache_dict["forward_time"] + cache_dict["setup_time"] + progress.elapsed_time
        return metrics

    def step(self,
             tokens: torch.Tensor,
             targets: torch.Tensor,
             batch_mask: torch.Tensor,
             logger: Logger,
             memories: Optional[List[DeepMemoryState]] = None,
             scheduling_rates: Optional[Tuple[float, float]] = None,
             ) -> List[DeepMemoryState]:
        """
        Performs a single training step. This consists of
        embedding, going through the forward pass, and accumulating
        loss during the reverse pass.

        Note there IS no return, as it is assumed your optim
        will handle the updates.

        Numeric cache rate bears a little explanation. Although the
        reconstruction mechanism is fairly fantastic at it's job, in some
        situations it is possible for numerical divergences to happen. This
        will eventually cause behavior on the forward and backwards pass to
        diverge.

        To prevent this, every so many tokens we cache rather than discard
        the forward memory state, then resume from that point using that
        memory state. Something like 1000 might work, but keep in mind
        that too aggressive a value will eat up a lot of memory.

        :param tokens: Tokens. Shape (..., items)
        :param targets: Targets. Shape (..., items)
        :param batch_mask: The batch mask. Shape (..., items). True indicated padding.
                          - Memory is not updated where true.
        :param memories: The existing memories, if any.
        :param scheduling_rates: Weights attached to the losses for #1: mem access, and
                                 #2: token loss. If none, no adjustment happens. See
                                 main class string for more details.
        :return: The final memory state. In case you want to continue training or something.
        :return: The various metrics that are monitored.
        """
        with profiler.record_function("train_step: Embedding and intake"):
            # Unwrap and standardize the scheduling details
            if scheduling_rates is None:
                access_schedule, main_schedule = None, None
            else:
                access_schedule, main_schedule = scheduling_rates
                access_schedule = float(access_schedule)
                main_schedule = float(main_schedule)

            # setup the initial memory state
            if memories is None:
                memories = self.core.create_state(tokens.shape[:-1])

        # Run forward pass, setup, and reverse pass.
        pass_cache = self.run_forward_pass(
            tokens,
            targets,
            batch_mask,
            main_schedule,
            memories,
            logger
        )
        final_memory, final_rng = self.setup_reverse_pass(access_schedule, pass_cache, tokens.device)
        metrics = self.run_reverse_pass(tokens,
                                        targets,
                                        batch_mask,
                                        main_schedule,
                                        pass_cache,
                                        logger
                                        )

        # Perform restoration of final rng state
        set_rng_state(final_rng, tokens.device)
        logger.commit_metrics(metrics)
        return final_memory


class CausalLMGenerator(nn.Module):
    """
    The predictive model engine. Basically,
    this is a text-to-text generative model,
    but with long term context abilities.

    This is designed to use the recurrent
    CausalLMCore to make predictions and
    even maintain a conversation. It is a
    self contained unit which, when setup,
    contains everything needed to move a block
    of text into model responses.

    A running 'memories' is maintained, which
    can be provided and read into with fresh
    content from a user or prompt as needed,
    and once the content is read the model
    is prompted to generate.

    Once generation is finished, the resulting
    text and the memory state is returned, allowing
    continuance of conversation in a fairly
    easy manner - just pass the memories back
    in if you have followup questions!
    """

    @property
    def dtype(self) -> torch.dtype:
        return self.core.dtype

    @property
    def device(self) -> torch.device:
        return self.core.device

    def __init__(self,
                 core: CausalLMCore,
                 sampling_layer: SamplingInterface,
                 ):

        super().__init__()

        # Generative details
        self.sampling_layer = sampling_layer

        # Models
        self.core = core
        self.decoder = core.decoder
        self.vocabulary = core.vocabulary

    def prepare_prompt_tokens(self,
                              text: List[str],
                              ) -> torch.Tensor:
        """
        Performs the prompt tokenization process.

        The result will be a collection of tokens and
        masks that can be used to make predictions
        with the model.

        :param text: The text to tokenize
        :return:
        - The tokens
        - the batch mask
        """
        # Indicates to the model what we are reviewing is going to be
        # a prompt it needs to consume
        start_of_prompt = AdditionalSpecialTokens.prompt_token.value
        text = [start_of_prompt + " " + item for item in text]

        # tokenize it!
        encoded = self.vocabulary.tokenizer.batch_encode_plus(text,
                                                              add_special_tokens=True,
                                                              padding=True,
                                                              truncation=False,
                                                              return_tensors="pt",
                                                              return_attention_mask=True,
                                                              )

        # Extract and convert to correct device and dtype
        # return results

        tokens = encoded["input_ids"].to(device=self.device, dtype=torch.long)
        batch_mask = ~encoded["attention_mask"].to(device=self.device, dtype=torch.bool)
        return tokens, batch_mask

    def read_prompt(self,
                    tokens: torch.Tensor,
                    batch_mask: torch.Tensor,
                    memories: List[DeepMemoryState],
                    ) -> List[DeepMemoryState]:
        """
        Reads the prompt into the memory state,
        then returns the new state.

        :param tokens: The tokens to read into the memory
        :param batch_mask: The padding masks. True means not used
        :param memories: The memory state
        :return: The updated memory state
        """
        embeddings = self.vocabulary.embeddings(tokens)
        for embedding, mask_case in zip(embeddings.unbind(-2), batch_mask.unbind(-1)):
            _, memories = self.decoder(embedding, mask_case, memories)
        return memories

    def generate_response(self,
                          batch_width: int,
                          max_generated_tokens: int,
                          temperature: float,
                          memories: List[DeepMemoryState],
                          ) -> Tuple[torch.Tensor, List[DeepMemoryState]]:
        """
        Generates a response based on the given memory collection,
        by priming the model into the response mode.

        :param batch_width: The batch width. Lets us know how wide to prime the input.
        :param max_generated_tokens: The maximum number of tokens to generate
        :param temperature: The generation temperature. Passed into the sampling mechanism.
        :param memories: The memories to consider for this process
        :return: The chosen response tokens.
        """
        # Setup EOS tracking. when all batches have
        # seen an EOS we stop

        has_seen_eos = torch.zeros([batch_width], device=self.device, dtype=torch.bool)

        # Prime the model.
        #
        # In order to see itself ready to make responses,
        # the model needs to see a beginning of stream token,
        # then a response token. This lets it know it
        # is the model's turn to talk
        #
        # We setup such a series of tokens, then load
        # all but the last into memory. The last will
        # instead be used as the prompt during gen.

        prompt_string = self.vocabulary.tokenizer.bos_token
        prompt_string += " "
        prompt_string += AdditionalSpecialTokens.beginning_of_response_token.value
        batch_strings = [prompt_string] * batch_width
        encoding = self.vocabulary.tokenizer.batch_encode_plus(batch_strings,
                                                               return_tensors="pt",
                                                               add_special_tokens=False)

        prompt_tokens = encoding["input_ids"].to(device=self.device, dtype=torch.long)
        for token in prompt_tokens.unbind(-1)[:-1]:
            embeddings = self.vocabulary.embeddings(token)
            _, memories = self.decoder(embeddings, has_seen_eos, memories)

        # With the model primed, we run sampling
        tokens = []
        token = prompt_tokens[..., -1]
        for _ in range(max_generated_tokens):
            # Perform actual predictive process
            embedding = self.vocabulary.embeddings(token)
            embedding_output, memories = self.decoder(embedding, has_seen_eos, memories)
            prediction_logits = self.vocabulary.logit_projector(embedding_output)
            token = self.sampling_layer(prediction_logits, temperature)

            # Perform bookkeeping.
            tokens.append(token)
            has_seen_eos |= (token == self.vocabulary.tokenizer.eos_token_id)
            if torch.all(has_seen_eos):
                break

        tokens = torch.stack(tokens, dim=-1)
        return tokens, memories

    def forward(self,
                text: Union[str, List[str]],
                temperature: float = 1.0,
                max_gen_tokens: int = 10000,
                memories: Optional[List[DeepMemoryState]] = None,
                ) -> Tuple[Union[str, List[str]], List[DeepMemoryState]]:
        """
        Performs the generation action. This consists of reading
        in any additional context, then predicting the next token
        until done generating.
        :param text: The additional text to integrate
        :param memories: The memories to consider.
        :param temperature: The temperature to use
        :param max_gen_tokens: The maximum number of tokens to generate
        :return:
        - The resulting text. In whatever format you passed in
        - The resulting memories. This can be used to continue the conversation.
        """
        with torch.no_grad():

            # Standardize the incoming data.
            if not isinstance(text, list):
                text = [text]
                remove_list_at_end = True
            else:
                remove_list_at_end = False

            batch_width = len(text)
            if memories is None:
                memories = self.decoder.create_state([batch_width])

            # Tokenize. Read the prompt into the memories,
            # then generate the response tokens.

            prompt_tokens, prompt_mask = self.prepare_prompt_tokens(text)
            memories = self.read_prompt(prompt_tokens, prompt_mask, memories)
            response_tokens, memories = self.generate_response(batch_width, max_gen_tokens, temperature, memories)

            # Detokenize.
            response = self.vocabulary.tokenizer.batch_decode(response_tokens)

            # Finally, handle edge case and return.
            if remove_list_at_end:
                response = response[0]
            return response, memories
