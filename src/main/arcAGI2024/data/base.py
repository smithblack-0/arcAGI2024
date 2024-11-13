import numpy as np
import torch
from torch.utils import data
from torch.utils.data import Dataset, IterableDataset
from typing import Iterable, List
from collections.abc import Sized
class NumpyBufferStream:
    """
    A specialized buffer that can perform
    a vectorized pop, push, and length process.

    It will express the length of the items in
    its buffer, automatically refill itself when
    possible if items are popped, and allows a vectorized
    pop process.
    """
    def pop(self, indices: np.ndarray) -> np.ndarray:
        """
        Pops the given indices out of the array,
        returning them in the specified order. Additional
        elements are then loaded into the buffer if
        possible to bring it up to capacity.

        :param indices: The indices to pop.
        :return: The popped array.
        """
        vector_mask = np.zeros_like(self.buffer.)


    def get_lengths(self)->np.ndarray:
        """
        Gets the length of each item in the buffer,
        and returns it as an array.
        """
        return np.array([len(item) for item in self.buffer])
    def refill_buffer(self):
        required_cases = self.buffer_size - len(self.buffer)
        try:
            update = []
            while len(update) < required_cases:
                update.append(next(self.stream))
        except StopIteration:
            pass

    def __init__(self,
                 stream: Iterable,
                 buffer_size: int
                 ):
        self.stream = stream
        self.buffer = np.array([])
        self.buffer_size = buffer_size
        self.refill_buffer()

class BatchBufferedDataset(data.IterableDataset):
    """
    An important prefetching and dataset tool.
    This class will finish tokenization, pad, and
    minimize necessary padding using prefetching.
    """

    @staticmethod
    def select_batch_indices_using_clustoid(batch_size: int, lengths: np.ndarray) -> np.ndarray:
        """
        Selects the most effective batch. This will be the
        one where the average distance between elements is lowest
        over batch_size.

        :param batch_size: The size of the batch to form.
        :param lengths: The lengths associated with each potential batch in the cache.
        :return: The indices associated with the batch.
        """
        differences = lengths[:, None] - lengths[None, :]
        distances = np.abs(differences)

        indexes = np.argsort(distances, axis=-1)[:, :batch_size]
        distances = np.take_along_axis(distances, indexes, axis=-1)
        batch_distances = np.sum(distances, axis=-1)
        best_centroid_index = np.argmin(batch_distances)

        return indexes[best_centroid_index]

    def load_data_into_buffer(self, data_iterator: Iterable):
        """
        Loads data into the buffer up to the buffer size if possible,
        handling the end of the data iterator gracefully.

        :param data_iterator: The data source to pull from.
        """
        try:
            while len(self.buffer) < self.buffer_size:
                self.buffer.append(next(data_iterator))
        except StopIteration:
            pass

    def pad_sequence(self, sequence: List[int], max_length: int) -> List[int]:
        """
        Pads a sequence to the specified max_length with zeros.

        :param sequence: The sequence to be padded.
        :param max_length: The desired length after padding.
        :return: The padded sequence.
        """
        padding_length = max_length - len(sequence)
        return sequence + [self.padding_id] * padding_length
    def __init__(self,
                 batch_size: int,
                 num_workers: int,
                 worker_rank: int,
                 padding_id: int,
                 pretokenized_dataset: Dataset,
                 buffer_size: int = 10000,
                 shuffle: bool = True):
        sampler = data.DistributedSampler(pretokenized_dataset,
                                          num_workers=num_workers,
                                          rank=worker_rank,
                                          shuffle=shuffle)

        self.unbatched_loader = data.DataLoader(pretokenized_dataset,
                                                shuffle=False,
                                                batch_size=None,
                                                sampler=sampler)

        self.padding_id = padding_id
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.buffer = []

    def __iter__(self):
        data_generator = iter(self.unbatched_loader)

        while True:
            # Load data into buffer until it reaches buffer_size
            self.load_data_into_buffer(data_generator)
            if len(self.buffer) == 0:
                break  # Exit if there are no more items to load

            # Get lengths for clustering
            lengths = np.array([len(tokenized_text) for tokenized_text in self.buffer])
            batch_indices = self.select_batch_indices_using_clustoid(self.batch_size, lengths)

            # Create the batch by popping selected items from the buffer
            batch = [self.buffer.pop(i) for i in batch_indices]

            # Pad the batch to make all sequences of equal length
            max_length = max(len(tokenized_text) for tokenized_text in batch)
            padded_batch = [self.pad_sequence(tokenized_text, max_length) for tokenized_text in batch]
            tensor_batch = torch.tensor(padded_batch, dtype=torch.long, pin_memory=True)
            yield tensor_batch

