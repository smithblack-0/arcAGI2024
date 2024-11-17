import os
import shutil
import unittest
from typing import Optional

import torch
import time
from torch import nn

from concurrent.futures import ThreadPoolExecutor
from src.main.arcAGI2024 import load_vocabulary_off_huggingface_model
from src.main.arcAGI2024.model import (CausalLMCore, CausalLMTrainer, CausalLMGenerator, StandardTrainerCore,
                                       RecurrentDecoder, Vocabulary, CoreConfig, Logger)
from src.main.arcAGI2024.losses import CrossEntropyLoss, UniformMemLoss
from src.main.arcAGI2024.base import parallel_pytree_map
from src.main.arcAGI2024.sampling import TopLogitSampling
from src.main.arcAGI2024.grad_utils import AutorescaleGradientControl

class TestCausalLMCore(unittest.TestCase):

    def setUp(self):
        self.head_name = "gpt2"
        self.temp_directory = "temp_test_directory"

        # Clean up if the directory exists already
        if os.path.exists(self.temp_directory):
            shutil.rmtree(self.temp_directory)

    def tearDown(self):
        # Remove the temporary directory after each test
        if os.path.exists(self.temp_directory):
            shutil.rmtree(self.temp_directory)

    def initialized_correctly(self, model):
        # Manually process a collection of tokens
        #
        # They will not have meaning.

        tokens = torch.randint(0, 1000, [3])
        mask = torch.rand([3]) > 0.1

        mem_state = model.decoder.create_state([3])
        embeddings = model.vocabulary.embeddings(tokens)
        output, _ = model.decoder(embeddings, mask, mem_state)
        logits = model.vocabulary.logit_projector(output)
        response_tokens = logits.argmax(-1)
        final_text = model.vocabulary.tokenizer.decode(response_tokens)


    def test_basic_sanity(self):

        vocabulary = load_vocabulary_off_huggingface_model("gpt2")
        config = CoreConfig(
            num_layers=2,
            num_read_heads=10,
            num_write_heads=10,
            num_memories=2,
            dropout_rate=0.1,
            sublayers_dropout_rate=0.1,
            decoder_flavor="fast"
        )
        model = CausalLMCore.build_model_using_config(vocabulary, config)
        self.initialized_correctly(model)



    def test_save_load_no_directory(self):
        # Initialize and save when the directory does not exist
        vocabulary = load_vocabulary_off_huggingface_model("gpt2")
        config = CoreConfig(
            num_layers=2,
            num_read_heads=10,
            num_write_heads=10,
            num_memories=2,
            dropout_rate=0.1,
            sublayers_dropout_rate=0.1,
            decoder_flavor="fast"
        )
        model = CausalLMCore.build_model_using_config(vocabulary, config)

        # Ensure the directory does not exist before saving
        if os.path.exists(self.temp_directory):
            shutil.rmtree(self.temp_directory)

        # Save the model
        model.save_to_folder(self.temp_directory)

        # Load the model and validate initialization
        loaded_model = CausalLMCore.load_from_folder(self.temp_directory)
        self.initialized_correctly(loaded_model)

    def test_save_load_directory_exists(self):
        # Initialize and save when the directory exists (contains a junk file)
        vocabulary = load_vocabulary_off_huggingface_model("gpt2")
        config = CoreConfig(
            num_layers=2,
            num_read_heads=10,
            num_write_heads=10,
            num_memories=2,
            dropout_rate=0.1,
            sublayers_dropout_rate=0.1,
            decoder_flavor="fast"
        )
        model = CausalLMCore.build_model_using_config(vocabulary, config)

        # Create the directory and add a junk file
        os.makedirs(self.temp_directory)
        with open(os.path.join(self.temp_directory, "junk.txt"), "w") as f:
            f.write("This file should be deleted on save.")

        # Save the model
        model.save_to_folder(self.temp_directory)

        # Check that the junk file was deleted
        self.assertFalse(os.path.exists(os.path.join(self.temp_directory, "junk.txt")))

        # Load the model and validate initialization
        loaded_model = CausalLMCore.load_from_folder(self.temp_directory)
        self.initialized_correctly(loaded_model)

    def test_masking_sanity(self):
        vocabulary = load_vocabulary_off_huggingface_model("gpt2")
        config = CoreConfig(
            num_layers=2,
            num_read_heads=10,
            num_write_heads=10,
            num_memories=2,
            dropout_rate=0.1,
            sublayers_dropout_rate=0.1,
            decoder_flavor="fast"
        )
        model = CausalLMCore.build_model_using_config(vocabulary, config)


        tokens = torch.randint(0, 1000, [3])

        # Test memory does not update when masked
        memories = model.decoder.create_state([3])
        mask = torch.ones_like(tokens, dtype=torch.bool)
        embeddings = model.vocabulary.embeddings(tokens)
        _, new_memories = model.decoder(embeddings, mask, memories)
        def is_same(tensor: torch.tensor, new_tensor: torch.Tensor):
            self.assertTrue(torch.equal(tensor, new_tensor))
            return tensor
        parallel_pytree_map(is_same, memories, new_memories)

        # Test memory does update when not masked
        memories = model.decoder.create_state([3])
        mask = torch.zeros_like(tokens, dtype=torch.bool)
        embeddings = model.vocabulary.embeddings(tokens)
        _, new_memories = model.decoder(embeddings, mask, memories)
        def is_different(tensor: torch.tensor, new_tensor: torch.Tensor):
            self.assertTrue(torch.any(tensor != new_tensor))
            return tensor
        parallel_pytree_map(is_different, memories, new_memories)
class TestCausalLMTrainer(unittest.TestCase):
    def setUp(self):
        # Setup model core
        vocabulary = load_vocabulary_off_huggingface_model("gpt2")
        config = CoreConfig(
            num_layers=2,
            num_read_heads=10,
            num_write_heads=10,
            num_memories=2,
            dropout_rate=0.1,
            sublayers_dropout_rate=0.1,
            decoder_flavor="fast"
        )
        model = CausalLMCore.build_model_using_config(vocabulary, config)
        trainer_core = StandardTrainerCore(model)

        self.trainer_core = trainer_core
        self.vocabulary = vocabulary

        # Initialize loss functions
        self.main_loss_fn = CrossEntropyLoss(padding_token_id=0)
        self.mem_access_loss_fn = UniformMemLoss()
        self.gradient_norm =AutorescaleGradientControl()

    def create_trainer(self, save_cached_to_cpu: bool, device: Optional[torch.device] = None):
        trainer = CausalLMTrainer(
            trainer_core=self.trainer_core,
            main_loss=self.main_loss_fn,
            mem_loss=self.mem_access_loss_fn,
            gradient_normalization=self.gradient_norm,
            numeric_cache_rate = 1,
            save_cached_to_cpu=save_cached_to_cpu,
            verbose=True
        )
        trainer = trainer.to(device)
        return trainer
    def test_random_tokens(self):
        """
        Test the causal lm trainer using exact steps.

        Does the forward and backwards pass match? If so
        the numerics metrics should be small, since we
        are not going any tensors without replacing the old
        one.
        """
        # Initialize the CausalLMTrainer
        trainer = self.create_trainer(save_cached_to_cpu=False, device=torch.device("cpu"))
        torch.autograd.set_detect_anomaly(True)

        # Create some mock training data to utilize in the process
        #
        # It is 100 tokens in a batch of 3
        batch_size = 3
        num_tokens = 4
        tokens = torch.randint(0, self.vocabulary.tokenizer.true_vocab_size, [batch_size, num_tokens])
        targets = torch.randint(0, self.vocabulary.tokenizer.true_vocab_size, [batch_size, num_tokens])
        masks = torch.rand([batch_size, num_tokens]) > 0.5

        # Setup the logger. We are just going to print to the terminal

        with ThreadPoolExecutor(3) as executer:

            logger = Logger(executer, lambda x : print(x), lambda x : print(x))

            # Setup an optim
            optim = torch.optim.SGD(trainer.parameters(), lr=0.1)
            memories = trainer.step(tokens, targets, masks, logger)
            optim.step()
    def test_random_tokens_on_gpu(self):

        # Initialize the CausalLMTrainer
        device = torch.device("cuda")
        trainer = self.create_trainer(save_cached_to_cpu=False, device=device)
        torch.autograd.set_detect_anomaly(True)

        # Create some mock training data to utilize in the process
        #
        # It is 100 tokens in a batch of 3
        batch_size = 3
        num_tokens = 4
        tokens = torch.randint(0, self.vocabulary.tokenizer.true_vocab_size, [batch_size, num_tokens])
        targets = torch.randint(0, self.vocabulary.tokenizer.true_vocab_size, [batch_size, num_tokens])
        masks = torch.rand([batch_size, num_tokens]) > 0.5


        tokens = tokens.to(device)
        targets = targets.to(device)
        masks = masks.to(device)

        # Setup the logger. We are just going to print to the terminal

        with ThreadPoolExecutor(1) as executer:

            logger = Logger(executer, lambda x : print(x), lambda x : print(x))

            # Setup an optim
            optim = torch.optim.SGD(trainer.parameters(), lr=0.1)
            memories = trainer.step(tokens, targets, masks, logger)
            optim.step()

    def test_caching_on_cpu(self):

        # Initialize the CausalLMTrainer
        device = torch.device("cuda")
        trainer = self.create_trainer(save_cached_to_cpu=True, device=device)
        torch.autograd.set_detect_anomaly(True)

        # Create some mock training data to utilize in the process
        #
        # It is 100 tokens in a batch of 3
        batch_size = 3
        num_tokens = 4
        tokens = torch.randint(0, self.vocabulary.tokenizer.true_vocab_size, [batch_size, num_tokens])
        targets = torch.randint(0, self.vocabulary.tokenizer.true_vocab_size, [batch_size, num_tokens])
        masks = torch.rand([batch_size, num_tokens]) > 0.5


        tokens = tokens.to(device)
        targets = targets.to(device)
        masks = masks.to(device)

        # Setup the logger. We are just going to print to the terminal

        with ThreadPoolExecutor(1) as executer:

            logger = Logger(executer, lambda x : print(x), lambda x : print(x))

            # Setup an optim
            optim = torch.optim.SGD(trainer.parameters(), lr=0.1)
            memories = trainer.step(tokens, targets, masks, logger)
            optim.step()



class TestCausalLMGen(unittest.TestCase):

    def setUp(self):
        # Setup model core and sampling mode
        vocabulary = load_vocabulary_off_huggingface_model("gpt2")
        config = CoreConfig(
            num_layers=2,
            num_read_heads=10,
            num_write_heads=10,
            num_memories=2,
            dropout_rate=0.1,
            sublayers_dropout_rate=0.1,
            decoder_flavor="fast"
        )
        model_core = CausalLMCore.build_model_using_config(vocabulary, config)
        self.model_core = model_core

        # Purely deterministic. Hopefully.
        self.model_core.eval()
        self.sampling = TopLogitSampling()

    def test_basic_gen(self):
        # Test generation pipeline. Though without training it will
        # not produce sane content, at least we get to see it
        # in

        gen_model = CausalLMGenerator(self.model_core, self.sampling)
        test_string = "The quick brown fox jumps over the lazy dog"
        max_length = 40
        temperature = 1.0

        # Test pass with no memories
        output_text, memories = gen_model(test_string, max_gen_tokens=max_length, temperature=temperature)
        print(output_text)

        # Test pass with initialized memories
        output_text, memories = gen_model(test_string, temperature, max_length, memories)
        print(output_text)

    def test_batched_gen(self):
        # Test generation pipeline. Though without training it will
        # not produce sane content, at least we get to see it
        # in

        gen_model = CausalLMGenerator(self.model_core, self.sampling)
        test_string = "The quick brown fox jumps over the lazy dog"
        test_strings = [test_string]*20
        max_length = 40
        temperature = 1.0

        # Test pass with no memories
        output_texts, memories = gen_model(test_strings, max_gen_tokens=max_length, temperature=temperature)
        print(output_texts)

        # Test pass with initialized memories
        output_texts, memories = gen_model(test_strings, temperature, max_length, memories)
        print(output_texts)


