import os
import shutil
import unittest
import torch
from src.main.ArcAGI2024.model import CasualLMCore  # Adjust the import path as needed


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
        mem_state = model.decoder.create_state([3])
        embeddings = model.vocabulary.embeddings(tokens)
        output, _ = model.decoder(embeddings, mem_state)
        logits = model.vocabulary.logit_projector(output)
        response_tokens = logits.argmax(-1)
        final_text = model.vocabulary.tokenizer.decode(response_tokens)
    def test_basic_sanity(self):
        model = CasualLMCore.build_model_on_top_of_pretrained_head(
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
        model = CasualLMCore.build_model_on_top_of_pretrained_head(
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
        loaded_model = CasualLMCore.load_from_folder(self.temp_directory)
        self.initialized_correctly(loaded_model)

    def test_save_load_directory_exists(self):
        # Initialize and save when the directory exists (contains a junk file)
        model = CasualLMCore.build_model_on_top_of_pretrained_head(
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
        loaded_model = CasualLMCore.load_from_folder(self.temp_directory)
        self.initialized_correctly(loaded_model)

class TestCausalModelTrainer()