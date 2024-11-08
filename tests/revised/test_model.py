import os
import shutil
import unittest
import torch
import time
from torch import nn

from src.main.arcAGI2024.model import (CausalLMCore, CausalLMTrainer, CausalLMGenerator,
                                       RecurrentDecoder, VocabularyStruct)
from src.main.arcAGI2024.losses import CrossEntropyLoss, UniformMemLoss
from src.main.arcAGI2024.base import parallel_pytree_map
from src.main.arcAGI2024.sampling import TopLogitSampling
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

        tokens = torch.randint(0, model.vocabulary.tokenizer.true_vocab_size, [3])
        mask = torch.rand([3]) > 0.1

        mem_state = model.decoder.create_state([3])
        embeddings = model.vocabulary.embeddings(tokens)
        output, _ = model.decoder(embeddings, mask, mem_state)
        logits = model.vocabulary.logit_projector(output)
        response_tokens = logits.argmax(-1)
        final_text = model.vocabulary.tokenizer.decode(response_tokens)


    def test_basic_sanity(self):
        model = CausalLMCore.build_model_on_top_of_pretrained_head(
            head_model_name=self.head_name,
            num_layers=2,
            num_read_heads=10,
            num_write_heads=10,
            num_memories=2,
            dropout_rate=0.1,
            auxilary_dropout_rate=0.1
        )
        self.initialized_correctly(model)



    def test_save_load_no_directory(self):
        # Initialize and save when the directory does not exist
        model = CausalLMCore.build_model_on_top_of_pretrained_head(
            head_model_name=self.head_name,
            num_layers=2,
            num_read_heads=1,
            num_write_heads=1,
            num_memories=2,
            dropout_rate=0.1,
            auxilary_dropout_rate=0.1
        )

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
        model = CausalLMCore.build_model_on_top_of_pretrained_head(
            head_model_name=self.head_name,
            num_layers=2,
            num_read_heads=1,
            num_write_heads=1,
            num_memories=2,
            dropout_rate=0.1,
            auxilary_dropout_rate=0.1
        )

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
        model = CausalLMCore.build_model_on_top_of_pretrained_head(
            head_model_name=self.head_name,
            num_layers=2,
            num_read_heads=10,
            num_write_heads=10,
            num_memories=2,
            dropout_rate=0.1,
            auxilary_dropout_rate=0.1
        )

        tokens = torch.randint(0, model.vocabulary.tokenizer.true_vocab_size, [3])

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
        model_core = CausalLMCore.build_model_on_top_of_pretrained_head(
            head_model_name="gpt2",
            num_layers=2,
            num_read_heads=2,
            num_write_heads=2,
            num_memories=2,
            dropout_rate=0.1,
            auxilary_dropout_rate=0.1
        )
        self.model_core = model_core

        # Initialize loss functions
        self.main_loss_fn = CrossEntropyLoss(padding_token_id=0)
        self.mem_access_loss_fn = UniformMemLoss()

    def test_initialization(self):
        # Initialize the CausalLMTrainer
        trainer = CausalLMTrainer(
            model_core=self.model_core,
            main_loss_function=self.main_loss_fn,
            mem_access_loss_function=self.mem_access_loss_fn
        )

        # Check main attributes were set correctly
        self.assertEqual(trainer.main_loss_function, self.main_loss_fn)
        self.assertEqual(trainer.mem_access_loss_function, self.mem_access_loss_fn)
        self.assertEqual(trainer.core, self.model_core)
        self.assertEqual(trainer.decoder, self.model_core.decoder)
        self.assertEqual(trainer.vocabulary, self.model_core.vocabulary)

        # Confirming the type of decoder and vocabulary is maintained
        self.assertIsInstance(trainer.decoder, RecurrentDecoder)
        self.assertIsInstance(trainer.vocabulary, VocabularyStruct)

        # Ensure model core's embedding and logit projection layers are accessible
        self.assertIsInstance(trainer.vocabulary.embeddings, nn.Embedding)
        self.assertIsInstance(trainer.vocabulary.logit_projector, nn.Linear)
    def test_numeric_sanity(self):
        """
        Test the causal lm trainer using exact steps.

        Does the forward and backwards pass match? If so
        the numerics metrics should be small, since we
        are not going any tensors without replacing the old
        one.
        """
        # Initialize the CausalLMTrainer
        trainer = CausalLMTrainer(
            model_core=self.model_core,
            main_loss_function=self.main_loss_fn,
            mem_access_loss_function=self.mem_access_loss_fn
        )
        torch.autograd.set_detect_anomaly(True)

        # Create some mock training data to utilize in the process
        #
        # It is 100 tokens in a batch of 3
        tokens = torch.randint(0, self.model_core.vocabulary.tokenizer.true_vocab_size, [3, 100])
        targets = torch.randint(0, self.model_core.vocabulary.tokenizer.true_vocab_size, [3, 100])
        masks = torch.rand([3, 100]) > 0.5

        # Setup an optim
        optim = torch.optim.SGD(self.model_core.parameters(), lr=0.1)
        memories, numeric_metrics, loss_metric = trainer.step(tokens, targets, masks, numerics_cache_rate=1)

    def test_numeric_sanity_gpu(self):
        """
        Test the causal lm trainer using exact steps.

        Does the forward and backwards pass match? If so
        the numerics metrics should be small, since we
        are not going any tensors without replacing the old
        one.
        """


        # Create some mock training data to utilize in the process
        #
        # It is 100 tokens in a batch of 3
        batch_size = 100
        num_tokens = 50
        cache_rate = 50

        tokens = torch.randint(0, self.model_core.vocabulary.tokenizer.true_vocab_size, [batch_size, num_tokens])
        targets = torch.randint(0, self.model_core.vocabulary.tokenizer.true_vocab_size, [batch_size, num_tokens])
        masks = torch.rand([batch_size, num_tokens]) > 0.5

        device = torch.device("cuda")
        model = self.model_core.to(device=device)
        tokens = tokens.to(device=device)
        targets = targets.to(device=device)
        masks = masks.to(device=device)

        # Initialize the CausalLMTrainer
        trainer = CausalLMTrainer(
            model_core=model,
            main_loss_function=self.main_loss_fn,
            mem_access_loss_function=self.mem_access_loss_fn
        )

        # Setup an optim
        optim = torch.optim.SGD(model.parameters(), lr=0.1)
        torch.compile()
        # Run steps
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA],
                                    record_shapes=True, profile_memory=True) as prof:
            memories, numeric_metrics, loss_metric = trainer.step(tokens, targets, masks, numerics_cache_rate=cache_rate)
            optim.step()
            optim.zero_grad()

        print(numeric_metrics)
        print(prof.key_averages())
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
        print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))

    def test_cpu_transfer(self):
        # Create some mock training data to utilize in the process
        #
        # It is 100 tokens in a batch of 3
        batch_size = 3
        num_tokens = 100
        cache_rate = 50

        tokens = torch.randint(0, self.model_core.vocabulary.tokenizer.true_vocab_size, [batch_size, num_tokens])
        targets = torch.randint(0, self.model_core.vocabulary.tokenizer.true_vocab_size, [batch_size, num_tokens])
        masks = torch.rand([batch_size, num_tokens]) > 0.5

        device = torch.device("cuda")
        model = self.model_core.to(device=device)
        tokens = tokens.to(device=device)
        targets = targets.to(device=device)
        masks = masks.to(device=device)

        # Initialize the CausalLMTrainer
        trainer = CausalLMTrainer(
            model_core=model,
            main_loss_function=self.main_loss_fn,
            mem_access_loss_function=self.mem_access_loss_fn
        )

        # Run pass
        memories, numeric_metrics = trainer(tokens, targets, masks,
                                            numerics_cache_rate=cache_rate,
                                            save_cached_to_cpu=True
                                            )
    def test_normal_parameters(self):
        # Test with more typical parameters
        model_core = CausalLMCore.build_model_on_top_of_pretrained_head(
            head_model_name="gpt2",
            num_layers=10,
            num_read_heads=10,
            num_write_heads=10,
            num_memories=80,
            dropout_rate=0.1,
            auxilary_dropout_rate=0.1
        )
        trainer = CausalLMTrainer(model_core, self.main_loss_fn, self.mem_access_loss_fn)

        # Create some mock training data to utilize in the process
        #
        batch_size = 10
        num_tokens = 100
        cache_rate = 50

        tokens = torch.randint(0, self.model_core.vocabulary.tokenizer.true_vocab_size, [batch_size, num_tokens])
        targets = torch.randint(0, self.model_core.vocabulary.tokenizer.true_vocab_size, [batch_size, num_tokens])
        masks = torch.rand([batch_size, num_tokens]) > 0.5

        device = torch.device("cuda")
        trainer = trainer.to(device=device)
        tokens = tokens.to(device=device)
        targets = targets.to(device=device)
        masks = masks.to(device=device)

        # Run pass
        start_time = time.time()
        memories, numeric_metrics, loss_metric = trainer(tokens, targets, masks,
                                            numerics_cache_rate=cache_rate,
                                            save_cached_to_cpu=True
                                            )
        end_time = time.time()
        print(f"Time taken: {end_time - start_time} seconds")


class TestCausalLMGen(unittest.TestCase):

    def setUp(self):
        # Setup model core and sampling mode
        model_core = CausalLMCore.build_model_on_top_of_pretrained_head(
            head_model_name="gpt2",
            num_layers=2,
            num_read_heads=1,
            num_write_heads=1,
            num_memories=2,
            dropout_rate=0.1,
            auxilary_dropout_rate=0.1
        )
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
        max_length = 200
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
        max_length = 200
        temperature = 1.0

        # Test pass with no memories
        output_texts, memories = gen_model(test_strings, max_gen_tokens=max_length, temperature=temperature)
        print(output_texts)

        # Test pass with initialized memories
        output_texts, memories = gen_model(test_strings, temperature, max_length, memories)
        print(output_texts)


