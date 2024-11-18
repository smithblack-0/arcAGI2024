from src.main.arcAGI2024.vocabulary import Vocabulary, load_vocabulary_off_huggingface_model
from src.main.arcAGI2024.data.pretraining import create_pretokenized_datasets, data_collator
import unittest
import torch
import datasets
import time
import numpy as np
import functools

vocabulary_donor = 'gpt2'


class TestCreatePretokenizedDataset(unittest.TestCase):
    """
    Test that the create pretokenized dataset mechanism
    works, and ends up producing data that is tokenized
    with input IDs.
    """
    debugging_dataset = 'tiny_shakespeare'
    vocabulary_donor = 'gpt2'

    def setUp(self):
        # Load the debugging dataset
        self.dataset_origin = datasets.load_dataset("wikitext", "wikitext-2-raw-v1")
        self.vocab = load_vocabulary_off_huggingface_model(self.vocabulary_donor)
        self.tokenizer = self.vocab.tokenizer

    def test_pretokenize_dataset(self):
        # Test pretokenization without caching
        start_time = time.time()
        pretokenized_dataset = create_pretokenized_datasets(
            self.dataset_origin, self.tokenizer, "text" , use_cache=False
        )
        no_cache_time = time.time() - start_time

        # Verify the dataset has been tokenized into input IDs
        self.assertIn("input_ids", pretokenized_dataset["train"].features)
        self.assertNotIn("attention_mask", pretokenized_dataset["train"].features,
                         "Attention mask should not be generated during pretokenization")

        # Test that when cached, the function runs much faster
        start_time = time.time()

        def should_not_tokenize(*x):
            raise RuntimeError("Should not tokenize")

        pretokenized_cached = create_pretokenized_datasets(
            self.dataset_origin, self.tokenizer,"text", use_cache=True
        )
        cached_time = time.time() - start_time

        self.assertLess(cached_time, no_cache_time, "Caching should make pretokenization faster")

        # Test the cached entries detokenize sanely
        sample_entry = pretokenized_cached["train"][0]
        input_ids = sample_entry["input_ids"]

        # Detokenize and compare
        detokenized_text = self.tokenizer.decode(input_ids)
        original_text = self.dataset_origin["train"][0]["text"]

        self.assertIn(original_text, detokenized_text, "Detokenized text should match the original")


class TestBatchCollator(unittest.TestCase):
    """
    Test that the batch collator function works properly
    when provided with a batch that needs to be collated,
    consisting of already-defined tokens.
    """

    def setUp(self):
        """
        Set up a batch of mock tokens for testing the collator.
        Each entry in the batch contains a tokenized sequence of random length.
        """

        # Create a mock batch of sequences of varying lengths
        self.mock_batch = [{"input_ids": np.random.randint(0, 100, size=np.random.randint(5, 15)).tolist()}
                           for _ in range(10)]
        self.truncate_length = 10
        self.pad_id = 101

        # Bind the data_collator function with the tokenizer and truncate length
        self.collator = functools.partial(data_collator,
                                          pad_id=self.pad_id,
                                          truncate_length=self.truncate_length)

    def test_collator_shapes(self):
        """
        Test that the collator correctly produces inputs, targets, and attention masks
        with expected shapes.
        """
        inputs, targets, batch_mask = self.collator(self.mock_batch)

        # Assert that the inputs and targets have the correct shape

        self.assertEqual(inputs.shape, targets.shape)
        self.assertGreaterEqual(self.truncate_length, inputs.shape[-1])

        # Assert that the attention mask matches the input shape
        self.assertEqual(batch_mask.shape, inputs.shape)

    def test_padding_correctness(self):
        """
        Test that sequences are padded to the same length and that padding tokens
        are correctly set to 0.
        """
        inputs, targets, batch_mask = self.collator(self.mock_batch)

        # Check that padding tokens appear after sequence content
        for seq in inputs:
            content = seq.numpy().tolist()
            padding_start = content.index(self.pad_id) if self.pad_id in content else len(content)
            self.assertTrue(all(x == self.pad_id for x in content[padding_start:]))

    def test_truncation(self):
        """
        Test that sequences longer than the truncate_length are truncated correctly.
        """
        long_sequence = [{"input_ids": np.random.randint(0, 100, size=50).tolist()}]
        inputs, targets, batch_mask = self.collator(long_sequence)

        # Assert that the sequence is truncated to truncate_length - 1 (because of slicing)
        self.assertEqual(inputs.shape[1], self.truncate_length - 1)

    def test_attention_mask(self):
        """
        Test that the attention mask correctly distinguishes between content and padding.
        """
        inputs, targets, batch_mask = self.collator(self.mock_batch)

        # Verify that attention mask matches non-padding tokens
        for seq, mask in zip(inputs, batch_mask):
            seq_list = seq.numpy().tolist()
            mask_list = mask.numpy().tolist()
            padding_start = seq_list.index(self.pad_id) if self.pad_id in seq_list else len(seq_list)

            # Content tokens should have mask 1, padding tokens should have mask 0
            self.assertTrue(all(m == 1 for m in mask_list[:padding_start]))
            self.assertTrue(all(m == 0 for m in mask_list[padding_start:]))

