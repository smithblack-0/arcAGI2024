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
from .decoder import RecurrentDecoder, MemoryState, build_decoder
from .vocabulary import VocabularyStruct
from .base import get_rng_state, set_rng_state, parallel_pytree_map
from .losses import MainLossInterface, MemAccessLossInterface
from .sampling import SamplingInterface


class CasualLMCore(nn.Module):
    """
    A central place in which all parameters which belong
    to a particular model can be placed. This will
    contain the vocabulary and the model feature.
    """

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
            d_core = d_model//8
        if d_address is None:
            d_address = d_core//4
        if d_memory is None:
            d_memory = d_core
        if d_hidden is None:
            d_hidden = d_core*4

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

    def rebase_model_onto_vocabulary(self, vocabulary: VocabularyStruct)->'CasualLMCore':
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
    def load_from_folder(cls, directory: Union[str, os.PathLike])->'CasualLMCore':
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
                 ):
        super().__init__()
        assert decoder.d_model == vocabulary.d_model, "Decoder and vocabularies had different widths."
        self.vocabulary: VocabularyStruct = vocabulary
        self.decoder: RecurrentDecoder = decoder

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
    A model designed specifically
    to exist in a training loop, and
    act to train the model core.

    It will expect to be invoked with
    token, target collections and asked
    to accumulate losses.
    """

    def __init__(self,
                 model_core: CasualLMCore,
                 main_loss_function: MainLossInterface,
                 mem_access_loss_function: MemAccessLossInterface,
                 ):
        super().__init__()

        # Generative details
        sampling_function: Callable[[torch.Tensor, torch.Tensor, Optional[float]], torch.Tensor]

        # Training details
        self.main_loss_function = main_loss_function
        self.mem_access_loss_function = mem_access_loss_function

        # Models
        self.core = model_core
        self.decoder = model_core.decoder
        self.vocabulary = model_core.vocabulary
    def run_forward_pass(self,
                         embeddings: torch.tensor,
                         numerics_cache_rate: int,
                         save_cached_to_cpu: bool,
                         memories: List[MemoryState]
                         )->Dict[str, Any]:
        """
        Runs the model forward pass. Records numeric metrics, random
        seeds, and other similar details as we go.

        :param embeddings: The embeddings to process. Shape (batch_size, items, d_model)
        :param numerics_cache_rate: The rate to cache numeric metrics and subsitutions
        :param save_cached_to_cpu: Whether to save the numeric caches to cpu
        :param memories: The existing memoies
        :return: Various features required for the backwards pass
        """

        # Setup the various required caches.

        rng_states = []
        numeric_caches = []

        # Perform the forward pass to gain access to the
        # memories I shall need. This consists of recurrently
        # updating again and again. We discard the final state

        with torch.no_grad(), profiler.record_function("train step forward pass"):
            for i, embedding in enumerate(embeddings.unbind(-2)):
                # Run forward pass
                rng_state = get_rng_state(embedding.device)
                logits, memories = self.model(embedding, memories)

                # Integrate rng into cache.
                #
                # This includes moving it to the cpu if relevant
                if save_cached_to_cpu:
                    rng_state = rng_state.to(device="cpu")
                rng_states.append(rng_state)

                # Integrate numeric checkpoints into the cache
                #
                # This includes moving it to the cpu if relevant
                if i % numerics_cache_rate == 0:
                    probabilities = torch.softmax(logits, dim=-1)
                    if save_cached_to_cpu:
                        save_to_cpu = lambda x: x.cpu()
                        probabilities = save_to_cpu(probabilities)
                        saved_memories = parallel_pytree_map(save_to_cpu, memories)
                    package = i, saved_memories, probabilities
                    numeric_caches.append(package)
                else:
                    numeric_caches.append(None)

        # Return the results
        results = {}
        results["memories"] =


    def forward(self,
                tokens: torch.Tensor,
                targets: torch.Tensor,
                schedule_details: Optional[Tuple[float, float]] = None,
                numerics_cache_rate: int = 500,
                save_cached_to_cpu: bool = False,
                ) -> List[MemoryState]:
        """
        Performs a single training step. This consists of
        embedding, going through the forward pass, and accumulating
        loss during the reverse pass.

        Note there IS no return, as it is assumed your optim
        will handle the updates.

        Numeric cache rate bears a little explantation. Although the
        reconstruction mechanism is fairly fantastic at it's job, in some
        situations it is possible for numerical divergences to happen. This
        will eventually cause behavior on the forward and backwards pass to
        diverge.

        To prevent this, every so many tokens we cache rather than discard
        the forward memory state, then resume from that point using that
        memory state. Something like 1000 might work, but keep in mind
        that too aggressive a value will eat up a lot of memory.

        :param tokens: Tokens. Shape (..., items, d_model)
        :param targets: Targets. Shape (..., items, d_model)
        :param schedule_details: An additional parameter, which can insert scheduling
                                 information into the loss functions. When provided,
                                 the first float will be fed to the mem_access_loss_function
                                 function, and the second to the main loss function.
        :param numerics_cache_rate: How frequently to perform numeric caching and how
               frequently the numeric stability metrics will be checked
        :param save_cached_to_cpu: Whether to save cached values to cpu. If
                you are running out of gpu memory, this might help. It controls
                where the numerics caches are stored.
        :return: the final memory state. In case you want to continue training or something.
        :
        """
        with profiler.record_function("train step embedding and intake"):
            # Unwrap and standardize the scheduling details
            if schedule_details is None:
                access_schedule, main_schedule = None, None
            else:
                access_schedule, main_schedule = schedule_details

            # embed the tokens and setup the initial memory state
            embeddings = self.vocabulary.embeddings(tokens)
            memories = self.model.create_state(embeddings.shape[:-1])


        # Setup for the reverse pass.
        #
        # We use a special function that maps over
        # datastructures until finding tensor leaves
        # to efficiently make the memories require
        # grads.
        #
        # We also setup a function to continue the
        # memory gradient chain.

        with profiler.record_function("train step intermission"):

            final_rng = get_rng_state(embeddings.device)
            final_memories = memories
            numeric_metrics = {}

            def assign_as_leafs(tensor: torch.Tensor) -> torch.Tensor:
                # Sets up tensor as leaf
                tensor.requires_grad_(True)
                return tensor

            def propogate_memory_grads(tensor: torch.Tensor) -> torch.Tensor:
                # Moves gradients further backwards
                if tensor.grad is not None:
                    tensor.backward(tensor.grad, retain_graph=True)

            def transfer_memory_grads(original_tensor: torch.Tensor,
                                      new_tensor: torch.Tensor,
                                      ) -> torch.Tensor:
                # Sets up the new tensor to be a leaf,
                # and transfers the gradients onto the new
                # tensor so propagate works correctly
                new_tensor = assign_as_leafs(new_tensor)
                new_tensor.grad = original_tensor.grad
                return new_tensor

            memories: List[MemoryState] = parallel_pytree_map(assign_as_leafs, memories)

            # Additional reverse pass setup. We are going to focus exclusively on
            # the write probability masses, and we are going to form losses on them
            # based on the mass, which is then backpropogated into the leaf. This
            # can then continue backwards to influence prior decisions
            #
            # Also, we set it aside since it is a sort of metric that tells
            # us how the model is thinking
            if self.mem_access_loss_function is not None:
                access_loss = torch.tensor(0)
                for memory_state in memories:
                    access_loss += self.mem_access_loss_function(memory_state.write_probability_mass, access_schedule)
                access_loss.backward()
                assert memories[0].write_probability_mass.grad is not None

        # Perform the reverse pass, and accumulate gradients
        numerics_metrics = {}
        with profiler.record_function("train step reverse pass"):
            for embedding, target in reversed(zip(embedding, targets.unbind(-1))):
                # Manage RNG. Decrement counter
                rng_state = rng_states.pop()
                set_rng_state(rng_state, embedding.device)

                # If we are at one of the numeric controls, handle
                # setting up the tensor and transfering gradients.
                #
                # We also need to transfer back to the gpu or whatever if
                # we are saving on the cpu.
                numeric_entry = numeric_caches.pop()
                if numeric_entry is not None:
                    loc, memories, probabilities = numeric_metrics
                    if save_cached_to_cpu:
                        load_from_cpu : lambda x : x.to(device=embedding.device)
                        memories = parallel_pytree_map(load_from_cpu, memories, numeric_entry)
                        forward_probabilities = load_from_cpu(probabilities)
                    memories = parallel_pytree_map(transfer_memory_grads, memories, numeric_entry)

                with profiler.record_function("train step backwards prop"):
                    # Run reverse step
                    embedding, new_memories = self.model.reverse(embedding, memories)

                    # Continue memory losses, when applicable.
                    #
                    # The first memory state, being a leaf, will not
                    # have any gradients. Further ones are marked
                    # to retain grads, however.
                    if not is_first_step:
                        parallel_pytree_map(propogate_memory_grads, memories)
                    else:
                        is_first_step = False
                    memories = new_memories

                    # Now, perform the loss process. We can now safely discard
                    # the graph.
                    logits = self.vocabulary.logit_projector(embedding)
                    loss = self.main_loss_function(logits, target, main_schedule)
                    loss.backward()

                # Handle the numerics metrics
                if numeric_entry is not None:
                    with torch.no_grad():
                        backwards_probabilities = torch.softmax(logits, dim=-1)
                        probability_difference = (backwards_probabilities - forward_probabilities).abs()


        # Finish up by placing the RNG back to where it is supposed
        # to be.
        set_rng_state(final_rng, embeddings.device)
        return final_memories


class CausalLMGenerator(nn.Module):
    """
    The predictive model container.

    Used to actually make predictions using
    the core causal language model. It can
    be invoked with text and prior memory state,
    then integrates those and responds.

    """

    def __init__(self,
                 core: CasualLMCore,
                 sampling_layer: SamplingInterface,
                 device: torch.device,
                 ):

        super().__init__()

        # Generative details
        self.sampling_layer = sampling_layer
        self.device = device

        # Models
        self.core = core
        self.decoder = core.decoder
        self.vocabulary = core.vocabulary

    def read(self,
             text: List[str],
             memories: Optional[List[MemoryState]] = None
             ) -> List[MemoryState]:
        """
        Reads a collection of text into the model, inserting
        the begin of reading action at the beginning.
        :param text: The text to read in
        :param memories: The current memories, if they exist
        :return: The information, after read in.
        """
        # Tokenize.
        #
        # We insert the start of read token,
        # then run from there.
        #
        # Then embed

        start_of_read = self.vocabulary.tokenizer.special_tokens_map["read_token"]
        text = [start_of_read + " " + item for item in text]
        tokens = self.vocabulary.tokenizer.batch_encode_plus(text, add_special_tokens=True, return_tensors="pt",
                                                             padding=True, truncation=False,
                                                             )
        tokens = tokens.to(self.device)
        embeddings = self.vocabulary.embeddings(tokens)

        # Setup the required state. This includes
        # running and final memory containers,
        # and things to track whether we are done
        # generating
        if memories is None:
            memories = self.model.create_state(tokens.shape)
        finished_memories = parallel_pytree_map(lambda x: x.clone(), memories)

        # Run the forward pass.
        for embedding, token in zip(embeddings.unbind(-2), tokens.unbind(-1)):
            # Run the pass.
            _, memories = self.model(embedding, memories)

            # Update the final memories which will be returned
            newly_finished = token == self.vocabulary.tokenizer.eos_token_id

            def save_memories(final_memory_tensor: torch.Tensor,
                              iteration_memory_tensor: torch.Tensor
                              ) -> torch.Tensor:
                mask_copy = newly_finished
                while mask_copy.dim() < final_memory_tensor.dim():
                    mask_copy = mask_copy.unsqueeze(-1)
                return torch.where(mask_copy, iteration_memory_tensor, final_memory_tensor)

            finished_memories = parallel_pytree_map(save_memories, finished_memories, memories)

        # Return the read state
        return finished_memories

    def predict_next_tokens(self,
                            token: torch.Tensor,
                            memories: List[MemoryState],
                            temperature: float
                            ) -> Tuple[torch.Tensor, List[MemoryState]]:
        """
        Predicts the integer ids associated with the next
        token based on the given token, the memory, and
        the temperature

        :param token: The token under consideration.
        :param memories: The memories. Presumably you read in a prompt before using, so not optional.
        :param temperature: The generation temperature
        :return:
        - The predicted next token. Has been sampled
        - The updated memory state
        """
        embedding = self.vocabulary.embeddings(token)
        embedding_output, memories = self.model(embedding, memories)
        prediction_logits = self.vocabulary.logit_projector(embedding_output)
        predicted_token = self.sampling_layer(prediction_logits, temperature)
        return predicted_token, memories

    def forward(self,
                text: Union[str, List[str]],
                memories: Optional[List[MemoryState]] = None,
                temperature: float = 1.0
                ) -> Tuple[Union[str, List[str]], List[MemoryState]]:
        """
        Performs the generation action. This consists of reading
        in any additional context, then predicting the next token
        until done generating.
        :param text: The additional text to integrate
        :param memories: The memories to consider.
        :param temperature: The temperature to use
        :return: The result.
        """
        # Standardize the incoming data.
        #
        # Then run the read pass
        with torch.no_grad():
            if not isinstance(text, list):
                text = [text]
                remove_list_at_end = True
            else:
                remove_list_at_end = False
            memories = self.read(text, memories)

            # Setup response tokenization and prompting.
            eos_seen = torch.zeros([len(text)], dtype=bool, device=self.device)
            finished_memories = parallel_pytree_map(lambda x: x.clone(), memories)
            tokens = []
            tokens.append(torch.full_like([len(text)],
                                          fill_value=self.vocabulary.tokenizer.bos_token_id,
                                          dtype=torch.long, device=self.device)
                          )
            while not torch.all(eos_seen):
                # Run step
                token, memories = self.predict_next_tokens(tokens, memories, temperature)
                tokens.append(token)

                # Update the final memories which will be returned
                newly_finished = token == self.vocabulary.tokenizer.eos_token_id

                def save_memories(final_memory_tensor: torch.Tensor,
                                  iteration_memory_tensor: torch.Tensor
                                  ) -> torch.Tensor:
                    mask_copy = newly_finished
                    while mask_copy.dim() < final_memory_tensor.dim():
                        mask_copy = mask_copy.unsqueeze(-1)
                    return torch.where(mask_copy, iteration_memory_tensor, final_memory_tensor)

                finished_memories = parallel_pytree_map(save_memories, finished_memories, memories)

            # Stack up all those tokens. Decode.
            tokens = torch.stack(tokens, dim=-1)
            response = self.vocabulary.tokenizer.batch_decode(tokens)
            if remove_list_at_end:
                response = response[0]
            return response
