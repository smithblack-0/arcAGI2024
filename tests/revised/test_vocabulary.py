import unittest
import os
import shutil
import torch
from transformers import AutoTokenizer
from src.main.arcAGI2024.vocabulary import VocabularyStruct, SpecialTokens, AdditionalSpecialTokens


class TestVocabularyStruct(unittest.TestCase):
    TEST_DIR = "./test_vocab_directory"
    MODEL_NAME = "gpt2"

    @classmethod
    def setUpClass(cls):
        # Ensure the test directory is available and clean
        if os.path.exists(cls.TEST_DIR):
            shutil.rmtree(cls.TEST_DIR)
        os.makedirs(cls.TEST_DIR)

    @classmethod
    def tearDownClass(cls):
        # Clean up the test directory after all tests
        if os.path.exists(cls.TEST_DIR):
            shutil.rmtree(cls.TEST_DIR)

    def check_special_tokens_in_tokenizer(self, vocabulary_struct):
        """Check that all special tokens are present in the tokenizer."""
        for token_enum in list(SpecialTokens) + list(AdditionalSpecialTokens):
            token = token_enum.value
            self.assertIn(token, vocabulary_struct.tokenizer.all_special_tokens,
                          f"Token '{token}' not found in tokenizer special tokens.")

    def check_special_tokens_embedding(self, vocabulary_struct):
        """Check that all special tokens can be embedded."""
        token_ids = vocabulary_struct.tokenizer.convert_tokens_to_ids(
            [token.value for token in list(SpecialTokens) + list(AdditionalSpecialTokens)]
        )
        token_tensor = torch.tensor(token_ids).to(vocabulary_struct.embeddings.weight.device)
        embedding_output = vocabulary_struct.embeddings(token_tensor)
        self.assertEqual(embedding_output.shape[0], len(token_ids),
                         "Embedding output shape mismatch for special tokens.")

    def check_special_tokens_logits(self, vocabulary_struct):
        """Check that all special tokens are included in the logits projection."""
        num_tokens = vocabulary_struct.tokenizer.vocab_size
        self.assertEqual(vocabulary_struct.logit_projector.out_features,
                         vocabulary_struct.embeddings.num_embeddings
                         )


    def is_correct_initialize(self, vocabulary_struct):
        """Run all checks to confirm correct initialization of VocabularyStruct."""
        self.check_special_tokens_in_tokenizer(vocabulary_struct)
        self.check_special_tokens_embedding(vocabulary_struct)
        self.check_special_tokens_logits(vocabulary_struct)

    def test_basic_sanity(self):
        """Test basic initialization and validation with a model."""
        vocab = VocabularyStruct.auto_load_from_pretrained(self.MODEL_NAME)
        self.is_correct_initialize(vocab)

    def test_save_load(self):
        """Test saving and loading the vocabulary."""
        # Initialize and save vocabulary
        vocab = VocabularyStruct.auto_load_from_pretrained(self.MODEL_NAME)
        vocab.save_pretrained_vocabulary(self.TEST_DIR)

        # Load vocabulary
        loaded_vocab = VocabularyStruct.load_pretrained_vocabulary(self.TEST_DIR)

        # Run initialization checks on loaded vocabulary
        self.is_correct_initialize(loaded_vocab)


