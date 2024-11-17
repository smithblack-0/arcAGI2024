import unittest
import numpy as np
import torch
import timeit
from typing import List, Iterable
from src.main.arcAGI2024.data.base import NumpyBufferedStream, make_buffered_pipeline, BufferedBatchSampler
from torch.utils.data import Dataset
class TestNumpyBufferedStream(unittest.TestCase):

    def setUp(self):
        """Set up a batched mock stream and an instance of NumpyBufferedStream for testing."""
        # Define a sample stream with batches of sequences of different lengths
        self.sample_stream = [
            [np.array([i] * length) for i, length in enumerate(range(batch_start, batch_start + 5))]
            for batch_start in range(1, 50, 5)
        ]  # Creates 4 batches with 5 items each
        self.buffer_size = 10

    def test_initial_buffer_fill(self):
        """Test if buffer is filled up to (or slightly over) the buffer size on initialization."""
        stream = NumpyBufferedStream(iter(self.sample_stream), self.buffer_size)
        # Check that buffer is filled up to or slightly over the buffer size
        self.assertGreaterEqual(len(stream.buffer), self.buffer_size)
        self.assertLessEqual(len(stream.buffer), self.buffer_size + 5)  # Allowing for one batch overflow

    def test_get_lengths(self):
        """Test if get_lengths returns the correct lengths of each item in the buffer."""
        stream = NumpyBufferedStream(iter(self.sample_stream), self.buffer_size)
        expected_lengths = np.array([len(item) for item in stream.buffer])
        np.testing.assert_array_equal(stream.get_lengths(), expected_lengths)

    def test_pop_elements(self):
        """Test if pop correctly removes specified indices and refills the buffer with batched data."""
        stream = NumpyBufferedStream(iter(self.sample_stream), self.buffer_size)

        # Pop a few indices
        indices_to_pop = np.array([0, 2, 4])  # Example indices to pop
        expected = np.array([stream.buffer[i] for i in indices_to_pop], dtype=object)
        popped_elements = stream.pop(indices_to_pop)

        # Check that popped elements have the expected values
        self.assertEqual(len(popped_elements), len(expected))
        for actual, expected_case in zip(popped_elements, expected):
            np.testing.assert_array_equal(actual, expected_case)

        # Check buffer length after pop and refill
        self.assertGreaterEqual(len(stream.buffer), self.buffer_size)

        # Ensure popped elements are not in buffer
        for item in popped_elements:
            for case in stream.buffer:
                if len(case) != len(item):
                    continue
                self.assertTrue(np.any(case != item))

    def test_refill_buffer_end_of_stream(self):
        """Test refill_buffer when the stream is nearly exhausted."""
        stream = NumpyBufferedStream(iter(self.sample_stream), self.buffer_size)

        # Exhaust the stream manually
        for _ in range(15):  # 15 pops should exhaust the buffer and the stream
            stream.pop(np.array([0]))  # Pop one item repeatedly to drain buffer

        # Now refill buffer should not raise errors, even if stream is exhausted
        stream.refill_buffer()
        # Buffer length should be less than or equal to buffer_size
        self.assertLessEqual(len(stream.buffer), self.buffer_size)

    def test_stream_exhaustion(self):
        """Test that the buffer stops refilling once the stream is exhausted."""
        stream = NumpyBufferedStream(iter(self.sample_stream), self.buffer_size)

        # Exhaust the stream completely
        while len(stream.buffer) > 0:
            stream.pop(np.array([0]))  # Keep popping to exhaust the stream and buffer

        # After stream exhaustion, buffer should not refill
        stream.refill_buffer()
        self.assertEqual(len(stream.buffer), 0)

class MockDataset(Dataset):
    """A simple dataset to provide tokenized data of varying lengths."""
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):

        if isinstance(index, list):
            output = []
            for item in index:
                output.append(self.data[item])
        else:
            return self.data[index]

        return output

    def __len__(self):
        return len(self.data)

def collate_passthrough(x):
    return x
class TestBatchBufferedDataset(unittest.TestCase):

    def setUp(self):
        """Set up a mock dataset and an instance of BatchBufferedDataset for testing."""
        # Create a dataset with tokenized samples of varying lengths
        self.mock_data = [np.array([i] * length) for i, length in enumerate(range(1, 21))]
        self.dataset = MockDataset(self.mock_data)
        self.padding_id = 0
        self.buffer_size = 10
        self.batch_size = 5
        # Create an instance of BatchBufferedDataset

        data = [[i for i in range(j)] for j in range(1,24)]
        self.dataset = MockDataset(data)

    def make_loader(self):
        return make_buffered_pipeline(self.batch_size,
                                      num_workers=1,
                                      worker_rank=0,
                                      pretokenized_dataset=self.dataset,
                                      collate_fn=collate_passthrough
                                      )
    def test_select_batch_indices_using_clustoid(self):
        """Test the selection of batch indices based on clustering."""
        lengths = np.array([len(item) for item in self.mock_data[:self.buffer_size]])
        batch_indices = BufferedBatchSampler.select_batch_indices_using_clustoid(self.batch_size, lengths)

        # Check if correct number of indices is returned
        self.assertEqual(len(batch_indices), self.batch_size)

        # Check if the indices are within the bounds of the buffer
        self.assertTrue(all(0 <= idx < self.buffer_size for idx in batch_indices))

    def test_iter_batches(self):
        """Test the batching, padding, and yielding of batches from the dataset."""

        loader = self.make_loader()
        batch_iter = iter(loader)

        for _ in range(4):  # Assuming 20 items in mock_data / batch_size = 4 batches
            batch = next(batch_iter)

            # Check if the batch is of the correct shape
            self.assertEqual(len(batch), self.batch_size)



    def test_stream_exhaustion(self):
        """Test that the iterator stops yielding batches when the data is exhausted."""
        loader = self.make_loader()
        batch_iter = iter(loader)
        batch_count = 0

        # Count the number of batches yielded
        for batch in batch_iter:
            batch_count += 1

        # With 20 items in mock_data and batch_size of 5, expect 4 full batches
        self.assertEqual(batch_count, 4)
class MockLargeDataset(Dataset):
    """A mock dataset with sequences of random lengths between 500 and 40000."""
    def __init__(self, num_entries: int, min_len: int = 500, max_len: int = 40000):
        self.data = [np.random.randint(0, 100, size=np.random.randint(min_len, max_len)).tolist()
                     for _ in range(num_entries)]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class TestBatchBufferedDatasetPerformance(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up a large mock dataset for performance testing."""
        # Large dataset with entries between 500 and 40000 units
        cls.num_entries = 20000
        cls.min_len = 500
        cls.max_len = 40000
        cls.buffer_size = 4000
        cls.batch_size = 800
        cls.padding_id = 0

        # Initialize the large dataset
        cls.large_dataset = MockLargeDataset(cls.num_entries, cls.min_len, cls.max_len)


    def test_select_batch_indices_timing(self):
        """Test timing for select_batch_indices_using_clustoid with full buffer."""

        batch_buffered_dataset = BatchingBufferedDataset(
            batch_size=self.batch_size,
            num_workers=1,
            worker_rank=0,
            padding_id=self.padding_id,
            pretokenized_dataset=self.large_dataset,
            buffer_size=self.buffer_size,
            shuffle=False
        )

        # Fill buffer with data from the dataset
        buffer_data = [self.large_dataset[i] for i in range(self.buffer_size)]
        lengths = np.array([len(item) for item in buffer_data])
        num_runs = 10
        # Define the function to time
        def select_batch_func():
            batch_buffered_dataset.select_batch_indices_using_clustoid(self.batch_size, lengths)

        # Time the function
        select_batch_time = timeit.timeit(select_batch_func, number=num_runs)
        average_batch_time = select_batch_time/num_runs

        print(f"select_batch_indices_using_clustoid execution time (10 runs): {average_batch_time:.4f} seconds")

    def test_full_iteration_timing(self):
        """Test timing for iterating through the dataset in batches."""\


        batch_buffered_dataset = BatchingBufferedDataset(
            batch_size=self.batch_size,
            num_workers=1,
            worker_rank=0,
            padding_id=self.padding_id,
            pretokenized_dataset=self.large_dataset,
            buffer_size=self.buffer_size,
            shuffle=False
        )

        # Define the function to time
        num_iterations = 10

        def iterate_through_dataset():
            batch_iter = iter(batch_buffered_dataset)
            for _ in zip(batch_iter):
                break

        # Time the function
        full_iteration_time = timeit.timeit(iterate_through_dataset, number=num_iterations)

        average_time = full_iteration_time/num_iterations
        print(f"average iteration through dataset ({num_iterations} runs): {average_time:.4f} seconds")

