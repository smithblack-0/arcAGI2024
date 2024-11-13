import numpy as np
import torch
from torch.nn import functional as F
from torch.utils import data
from torch.utils.data import Dataset, IterableDataset
from typing import Iterable, List
from collections.abc import Sized

class NumpyBufferedStream:
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
        chosen_mask = np.zeros(len(self.buffer), dtype=bool)
        chosen_mask[indices] = True
        output = self.buffer[chosen_mask]
        self.buffer = self.buffer[~chosen_mask]
        self.refill_buffer()
        return output

    def get_lengths(self) -> np.ndarray:
        """
        Gets the length of each item in the buffer,
        and returns it as an array.
        """
        return np.array([len(item) for item in self.buffer])

    def refill_buffer(self):
        """
        Refill the buffer by loading batches from the stream.
        """
        # Load a batch from the stream, appending it to the buffer
        try:
            while len(self.buffer) < self.buffer_size:
                batch = next(self.stream)
                # Flatten batch to 1D array if needed and append to the buffer
                batch = np.array(batch, dtype=object)
                self.buffer = np.concatenate([self.buffer, batch])
        except StopIteration:
            # Stop if the stream is exhausted
            pass

    def is_buffer_empty(self) -> bool:
        """Returns True if the buffer is empty."""
        return len(self.buffer) == 0

    def __init__(self,
                 stream: Iterable[Sized],
                 buffer_size: int):
        """
        Initialize the NumpyBufferedStream.

        :param stream: A batched iterable where each item is a batch of data.
        :param buffer_size: The maximum size of the buffer.
        """
        self.stream = iter(stream)
        self.buffer = np.array([], dtype=object)
        self.buffer_size = buffer_size
        self.refill_buffer()

class BatchingBufferedDataset(data.IterableDataset):
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

    def pad_sequence(self, sequence: List[int], max_length: int) -> List[int]:
        """
        Pads a sequence to the specified max_length with zeros.

        :param sequence: The sequence to be padded.
        :param max_length: The desired length after padding.
        :return: The padded sequence.
        """
        sequence = torch.tensor(sequence, dtype=torch.int64)
        padding_length = max_length - len(sequence)
        return F.pad(sequence, (0, padding_length))
    def __init__(self,
                 batch_size: int,
                 num_workers: int,
                 worker_rank: int,
                 padding_id: int,
                 pretokenized_dataset: Dataset,
                 buffer_size: int = 10000,
                 shuffle: bool = True):

        sampler = data.DistributedSampler(pretokenized_dataset,
                                          num_replicas=num_workers,
                                          rank=worker_rank,
                                          shuffle=shuffle)

        self.unbatched_loader = data.DataLoader(pretokenized_dataset,
                                                shuffle=False,
                                                collate_fn=lambda x : x,
                                                batch_size=batch_size,
                                                sampler=sampler,
                                                )

        self.padding_id = padding_id
        self.batch_size = batch_size
        self.buffer_size = buffer_size

    def __iter__(self):
        unbatched_tokens_buffer = NumpyBufferedStream(self.unbatched_loader, self.buffer_size)
        while not unbatched_tokens_buffer.is_buffer_empty():
            # Get lengths for clustering
            lengths = unbatched_tokens_buffer.get_lengths()
            batch_indices = self.select_batch_indices_using_clustoid(self.batch_size, lengths)

            # Create the batch by popping selected items from the buffer
            batch = unbatched_tokens_buffer.pop(batch_indices)

            # Pad the batch to make all sequences of equal length
            max_length = max(len(tokenized_text) for tokenized_text in batch)
            padded_tensors = [self.pad_sequence(tokenized_text, max_length) for tokenized_text in batch]
            batched_tensors = torch.stack(padded_tensors, dim=0)
            yield batched_tensors

