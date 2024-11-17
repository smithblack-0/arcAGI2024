import numpy as np
import torch
from torch.nn import functional as F
from torch.utils import data
from torch.utils.data import Dataset, IterableDataset
from typing import Iterable, List, Callable, Any
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


class BufferedBatchSampler(data.Sampler):
    """
    A Buffered batch sampler, responsible
    for sampling from an entire batch with
    a buffer for like sized collating
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

    def __init__(self,
                 batch_size: int,
                 stream_head: data.DataLoader,
                 num_batches_in_buffer: int = 20,
                 ):
        super().__init__(stream_head)
        self.buffer_stream_head = stream_head
        self.buffer_size = batch_size * num_batches_in_buffer
        self.batch_size = batch_size

    def __len__(self) -> int:
        return len(self.buffer_stream_head)

    def __iter__(self):
        buffer = NumpyBufferedStream(self.buffer_stream_head, self.buffer_size)
        while not buffer.is_buffer_empty():
            # Get lengths for clustering
            lengths = buffer.get_lengths()
            batch_indices = self.select_batch_indices_using_clustoid(self.batch_size, lengths)

            # Create the batch by popping selected items from the buffer
            batch = buffer.pop(batch_indices)
            yield batch


def make_buffered_pipeline(batch_size: int,
                           num_workers: int,
                           worker_rank: int,
                           collate_fn: Callable[[Any], torch.Tensor],
                           pretokenized_dataset: Dataset,
                           num_batches_in_buffer: int = 20,
                           shuffle: bool = True,
                           num_prefetch_threads: int = 4,
                           prefetch_factor: int = 2
                           ) -> data.DataLoader:
    """
    Creates a distributed buffered dataloader out of the
    given parameters including a pretokenized dataset
    returns: A setup dataloader.
    """

    # Create the primary loader. This will be responsible
    # for shuffling and accommodation of distribute processes,
    # and getting data off the hard drive.
    distributed_sampler = data.DistributedSampler(pretokenized_dataset,
                                                  num_replicas=num_workers,
                                                  rank=worker_rank,
                                                  shuffle=shuffle)

    stream_loader = data.DataLoader(pretokenized_dataset,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    collate_fn=lambda x : x,
                                    sampler=distributed_sampler)

    # Create the buffered loader. This will accept the primary
    # loader, and loads a certain number of the cases that it
    # keeps in a buffer. It chooses to batch cases that
    # have similar lengths

    batch_sampler = BufferedBatchSampler(batch_size, stream_loader, num_batches_in_buffer)

    return data.DataLoader(pretokenized_dataset,
                           batch_sampler=batch_sampler,
                           num_workers=num_prefetch_threads,
                           prefetch_factor=prefetch_factor,
                           collate_fn=collate_fn,
                           pin_memory=True)


