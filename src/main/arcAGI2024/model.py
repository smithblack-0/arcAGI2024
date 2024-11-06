"""
Core model builders and functions
"""
import functools
import os
import textwrap
import shutil
from typing import Callable, List
from typing import Any, List, Tuple, Dict, Union, Callable, Optional
import torch
from torch import nn
from torch.autograd import profiler
from torch.nn import functional as F

from .decoder import RecurrentDecoder, MemoryState, build_decoder
from .vocabulary import VocabularyStruct, AdditionalSpecialTokens
from .base import get_rng_state, set_rng_state, parallel_pytree_map, DeviceDtypeWatch
from .losses import MainLossInterface, MemAccessLossInterface
from .sampling import SamplingInterface
from .grad_utils import BatchCollectiveReductiveGradNorm, BatchCollectiveQuantileClipping

class CasualLMCore(nn.Module):
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
    def build_model_on_top_of_pretrained_head(cls,
                                              # Primary specifications
                                              head_model_name: str,
                                              num_layers: int,
                                              num_read_heads: int,
                                              num_write_heads: int,
                                              num_memories: int,

                                              # Helper specifics

                                              dropout_rate: float,
                                              auxilary_dropout_rate: float,

                                              # These have defaults.
                                              d_core: Optional[int] = None,
                                              d_memory: Optional[int] = None,
                                              d_address: Optional[int] = None,
                                              d_hidden: Optional[int] = None,
                                              dtype: torch.dtype = None,
                                              device: torch.device = None
                                              ):
        """
        Creates a functioning recurrent decoder, with
        forward and reverse modes, ready for integration
        into a broader architecture.

        The returned model is designed to be entirely
        recurrent.

        --- Top level parameters ---
        :param num_layers:
        - Number of transformer layers.
        - Each is a DecoderLayer
        :param head_model_name:
        - The huggingface model to get the CausalLM head from.
        :param dropout_rate:
        - Dropout rate, within the primary recurrent model
        :param auxilary_dropout_rate:
        - Dropout rate, within the core decoder layers and computation models.

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

        ---- final ---
        :param dtype: The dtype
        :param device: The device
        :return: A setup RecurrentDecoder
        """
        # Load the causal lm head
        vocabulary = VocabularyStruct.auto_load_from_pretrained(head_model_name)
        vocabulary = vocabulary.to(dtype=dtype, device=device)
        d_model = vocabulary.d_model

        # Standardize defaults
        if d_core is None:
            d_core = d_model // 8
        if d_address is None:
            d_address = d_core // 4
        if d_memory is None:
            d_memory = d_core
        if d_hidden is None:
            d_hidden = d_core * 4

        # Setup the model for training
        decoder = build_decoder(
            d_model,
            num_layers,
            d_core,
            d_hidden,
            d_address,
            d_memory,
            num_read_heads,
            num_write_heads,
            num_memories,
            dropout_rate,
            auxilary_dropout_rate,
            dtype=dtype,
            device=device
        )

        # Return instance
        return cls(vocabulary, decoder)

    def rebase_model_onto_vocabulary(self, vocabulary: VocabularyStruct) -> 'CasualLMCore':
        """
        Rebases the model to interface with a different vocabulary struct.
        Note this will require fine tuning or even pretraining a bit.

        But most of the learning should be preserved
        :param vocabulary: The vocabulary to base us on now
        :return: The new casual lm core
        """
        decoder = self.decoder.rebuild_at_different_width(vocabulary.d_model)
        return CasualLMCore(vocabulary, decoder)

    def save_to_folder(self,
                       directory: Union[str, os.PathLike]
                       ):
        """
        Saves the causal core to an indicated directory
        :param directory_name: The directory to save to. Will be created if needed
        """
        if os.path.isdir(directory):
            shutil.rmtree(directory)

        self.vocabulary.save_pretrained_vocabulary(directory)
        torch.save(self.decoder, os.path.join(directory, "decoder.pt"))

    @classmethod
    def load_from_folder(cls, directory: Union[str, os.PathLike]) -> 'CasualLMCore':
        """
        Loads the saved causal lm core file from the given directory.
        :param directory: The directory to load from
        :returns: The loaded causal lm core model
        """
        decoder = torch.load(os.path.join(directory, "decoder.pt"))
        vocabulary = VocabularyStruct.load_pretrained_vocabulary(directory)
        return cls(vocabulary, decoder)

    def __init__(self,
                 vocabulary: VocabularyStruct,
                 decoder: RecurrentDecoder,
                 device: Optional[torch.device] = None,
                 dtype: Optional[torch.dtype] = None,
                 ):
        super().__init__()
        assert decoder.d_model == vocabulary.d_model, "Decoder and vocabularies had different widths."
        self._metainfo = DeviceDtypeWatch(device=device, dtype=dtype)
        self.vocabulary: VocabularyStruct = vocabulary.to(device=device, dtype=dtype)
        self.decoder: RecurrentDecoder = decoder.to(device=device, dtype=dtype)

    def __reduce__(self):
        msg = """
        Should not use pickle or torch.save to save the model.
        Instead, invoke CausalLMCore.save_to_folder(directory),
        and load with CausualLMCore.load_from_folder(directory)
        """
        msg = textwrap.dedent(msg)
        raise NotImplementedError(msg)


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
    that would very quickly saturate the GPU.

    Instead, we can imagine the trainer  as computing the loss with a single token at a time,
    gathering gradients on "recurrent memories", and then manually propogating updates from
    those memories back in time. This is done in a forward pass, then a reverse pass. It
    also means there are some differences to the standard paradym.

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
                 model_core: CasualLMCore,
                 main_loss_function: MainLossInterface,
                 mem_access_loss_function: MemAccessLossInterface,
                 grad_norm_threshold: float = 0.01,
                 grad_clip_factor: float = 1000,
                 grad_norm_mode: str = "quantiles_mean"
                 ):
        super().__init__()

        # Training details
        self.main_loss_function = main_loss_function
        self.mem_access_loss_function = mem_access_loss_function
        self.batch_grad_norm = BatchCollectiveReductiveGradNorm(rescale_threshold=grad_norm_threshold,
                                                                reduction_mode=grad_norm_mode
                                                                )
        self.batch_grad_clip = BatchCollectiveQuantileClipping(clip_factor=grad_clip_factor)

        # Models
        self.core = model_core
        self.decoder = model_core.decoder
        self.vocabulary = model_core.vocabulary
        self.mse_metric = nn.MSELoss(reduction="mean")

    def run_forward_pass(self,
                         embeddings: torch.Tensor,
                         batch_mask: torch.Tensor,
                         numerics_cache_rate: int,
                         save_cached_to_cpu: bool,
                         memories: List[MemoryState]
                         ) -> Dict[str, Any]:
        """
        Runs the model forward pass. Records numeric metrics, random
        seeds, and other similar details as we go.

        :param embeddings: The embeddings to process. Shape (batch_size, items, d_model)
        :param batch_mask: The batch mask. Shape (batch_size, items). True indicated padding.
        :param numerics_cache_rate: The rate to cache numeric metrics and subsitutions
        :param save_cached_to_cpu: Whether to save the numeric caches to cpu
        :param memories: The existing memoies
        :return: Various features required for the backwards pass.
        """

        # Setup the various required caches.

        rng_states = []
        numeric_caches = []

        # Perform the forward pass to gain access to the
        # memories I shall need. This consists of recurrently
        # updating again and again. We discard the final state
        save_to_cpu = lambda x: x.cpu()
        output_embeddings = []
        with torch.no_grad(), profiler.record_function("train_step: forward pass"):
            for i, (embedding, mask) in enumerate(zip(embeddings.unbind(-2), batch_mask.unbind(-1))):
                # Run forward pass
                rng_state = get_rng_state(embedding.device)
                embedding, memories = self.decoder(embedding, mask, memories)
                output_embeddings.append(embedding.detach())

                # Integrate rng into cache. In order to reproducibly
                # run the reverse pass the same way as the forward
                # pass, we must cache the rng state at the beginning.
                #
                # This will be restored on the backwards pass.

                rng_states.append(rng_state)

                # Integrate numeric checkpoints into the cache
                #
                # This includes moving it to the cpu if relevant
                if i % numerics_cache_rate == 0:
                    if save_cached_to_cpu:
                        saved_memories = parallel_pytree_map(save_to_cpu, memories)
                    else:
                        saved_memories = memories
                    package = i, saved_memories
                    numeric_caches.append(package)
                else:
                    numeric_caches.append(None)

                def view_memories(tensor: torch.Tensor):
                    print(tensor.abs().mean())
                parallel_pytree_map(view_memories, memories)

        # Return the results
        results = {}
        results["memories"] = memories
        results["rng_states"] = rng_states
        results["numeric_cache"] = numeric_caches
        return results, output_embeddings

    def setup_reverse_pass(self,
                           access_schedule: Optional[float],
                           cache_dict: Dict[str, Any],
                           device: torch.device,
                           ) -> Tuple[List[MemoryState], torch.Tensor]:
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
            memory_state: List[MemoryState] = cache_dict["memories"]

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
                    loss += self.mem_access_loss_function(memory.write_probability_mass, access_schedule)
                loss.backward()

            # Store the revised memories, and the beginnings of
            # the loss metric.

            cache_dict["memories"] = memory_state
            cache_dict["loss"] = loss.detach()

        return final_memories, final_rng

    def run_reverse_pass(self,
                         embeddings: torch.Tensor,
                         targets: torch.Tensor,
                         batch_mask: torch.Tensor,
                         loss_schedule: Optional[float],
                         save_cached_to_cpu: bool,
                         cache_dict: Dict[str, Any]
                         ) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        Runs the reverse pass basically. This includes most of the
        backpropagation algorithm, alongside managing certain metrics.

        :param embeddings: The embeddings to process. Shape (..., items, d_model)
        :param targets: The target. Shape (..., items)
        :param batch_mask: The batch mask. Shape (batch_size, items). True indicated padding.
        :param loss_schedule: The schedule weight on the main loss functionm
        :param save_cached_to_cpu: Whether the intermediate caches were saved to the cpu or gpu
        :param cache_dict: The main cache, containing memories
        :return: The numerics metrics. KVDivegence at various locations between the forward and
                                       backwards predictions. Also mean prob difference.
        :return: The total loss. Is a detached metric, not a backprop feature.

        """

        numerics_metrics = {}
        with (profiler.record_function("train_step: Reverse pass")):
            # Setup states, and grad cache containers
            memories: List[MemoryState] = cache_dict["memories"]
            loss_metric = cache_dict["loss"]

            def get_mem_grads(tensor: torch.Tensor) -> torch.Tensor:
                if tensor.grad is not None:
                    return tensor.grad.detach()
                return tensor.grad

            mem_grads = parallel_pytree_map(get_mem_grads, memories)

            # Run iteration and backwards pass
            #
            # Make sure all tensors are detached too
            with torch.no_grad():
                iterator = zip(reversed(embeddings.unbind(-2)),
                               reversed(targets.unbind(-1)),
                               reversed(batch_mask.unbind(-1)))

            for embedding, target, mask in iterator:
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
                        if save_cached_to_cpu:
                            load_from_cpu = lambda x: x.to(device=embedding.device)
                            forward_memories = parallel_pytree_map(load_from_cpu, forward_memories)

                        # Compute the numeric divergence
                        numeric_divergences = []
                        def compute_mse_error(memory: torch.Tensor, actual_memory: torch.Tensor):
                            nonlocal numeric_divergences
                            numeric_divergences.append(self.mse_metric(memory, actual_memory))
                        parallel_pytree_map(compute_mse_error, memories, forward_memories)
                        numeric_divergences = torch.stack(numeric_divergences).max()
                        numerics_metrics[entry_num] = numeric_divergences

                        # Replace the memories with the stored numeric memories. Transfer the
                        # gradients so backprop continues to work
                        memories = forward_memories

                # Perform the actual backwards pass.
                #
                # We setup to do the backwards pass all at
                # once in a single go. To do this, we use a little
                # clever trick. We take each memory graph, attach a
                # hook to inject the grads into place where they
                # should end up, then sum and multiply by zero, then
                # add it to a loss
                #
                # Now when backprop is initiated, gradients that we
                # do not care about propogate backwards into the hook,
                # we discard them, and we replace them with what
                # is actually required.

                (embedding, next_memory), last_memory = self.decoder.reverse(embedding, mask, memories)
                logits = self.vocabulary.logit_projector(embedding)
                loss = self.main_loss_function(logits, target, loss_schedule)
                loss_metric += loss.detach()

                mem_grads_trigger = torch.zeros_like(loss)
                def inject_grad_hooks(tensor: torch.Tensor, grads: torch.Tensor):
                    nonlocal mem_grads_trigger
                    if grads is not None:
                        print(grads.abs().mean(), tensor.abs().mean())
                        item = tensor.abs().mean()
                        if tensor.abs().mean() > 1000:
                            print(tensor)
                    tensor.register_hook(lambda x, capture=grads: capture)
                    mem_grads_trigger += 0 * tensor.sum()

                parallel_pytree_map(inject_grad_hooks, next_memory, mem_grads)

                with profiler.record_function("train_step: Backpropagation"):
                    mem_grads_trigger.backward(retain_graph=True)
                    loss.backward()

                # Explicitly clear out tensors and detach graphs
                # on ANYTHING that actually had a graph attached.
                del memories
                del next_memory
                del embedding
                del logits
                del loss


                # Advance to the prior memory. Rescale and normalize if needed
                mem_grads = parallel_pytree_map(get_mem_grads, last_memory)
                with torch.no_grad(), profiler.record_function("train step: grad sanity"):
                    mem_grads = self.batch_grad_norm(mem_grads)
                    memories = parallel_pytree_map(lambda x: x.clone(), last_memory)

        return numerics_metrics, loss_metric

    def step(self,
             tokens: torch.Tensor,
             targets: torch.Tensor,
             batch_mask: torch.Tensor,
             memories: Optional[List[MemoryState]] = None,
             scheduling_rates: Optional[Tuple[float, float]] = None,
             numerics_cache_rate: int = 500,
             save_cached_to_cpu: bool = False,
             ) -> Tuple[List[MemoryState], Dict[int, torch.Tensor], torch.Tensor]:
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
        :param numerics_cache_rate: How frequently to perform numeric caching and how
               frequently the numeric stability metrics will be checked
        :param save_cached_to_cpu: Whether to save cached values to cpu. If
                you are running out of gpu memory, this might help. It controls
                where the numerics caches are stored.
        :return: The final memory state. In case you want to continue training or something.
        :return: The MSE between the forward and reverse predictions at cache rate locations.
        :return: The total loss. Is a detached metric, not a backprop feature.
        """
        with profiler.record_function("train_step: Embedding and intake"):
            # Unwrap and standardize the scheduling details
            if scheduling_rates is None:
                access_schedule, main_schedule = None, None
            else:
                access_schedule, main_schedule = scheduling_rates

            # embed the tokens and setup the initial memory state
            embeddings = self.vocabulary.embeddings(tokens)
            if memories is None:
                memories = self.decoder.create_state(embeddings.shape[:-2])

        # Run forward pass, setup, and reverse pass.
        pass_cache, _ = self.run_forward_pass(
            embeddings,
            batch_mask,
            numerics_cache_rate,
            save_cached_to_cpu,
            memories
        )
        final_memory, final_rng = self.setup_reverse_pass(access_schedule, pass_cache, embeddings.device)
        numerics_metrics, loss_metric = self.run_reverse_pass(embeddings,
                                                              targets,
                                                              batch_mask,
                                                              main_schedule,
                                                              save_cached_to_cpu,
                                                              pass_cache
                                                              )

        # Perform restoration of final rng state
        set_rng_state(final_rng, embeddings.device)

        return final_memory, numerics_metrics, loss_metric


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
                 core: CasualLMCore,
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
                    memories: List[MemoryState],
                    ) -> List[MemoryState]:
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
                          memories: List[MemoryState],
                          ) -> Tuple[torch.Tensor, List[MemoryState]]:
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
                memories: Optional[List[MemoryState]] = None,
                ) -> Tuple[Union[str, List[str]], List[MemoryState]]:
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
