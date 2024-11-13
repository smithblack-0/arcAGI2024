import numpy as np



def select_batch(batch_size: int, lengths: np.ndarray) -> np.ndarray:
    """
    Selects the most effective batch. This will be the
    one where the average distance between elements is lowest
    over batch_size.

    :param batch_size: The size of the batch to form.
    :param lengths: The lengths associated with each potential batch in the cache. Shape (buffer_size)
    :return: The indices associated with the batch. Shape (batch_size)
    """

    # This is a very inefficient, brute force, clustering algorithm
    #
    # But since we only expect to be making clusters out of like 10,000 entries,
    # it will probably work.
    #
    # We compute the distance between each element. We sort. We keep only a batch
    # length. One of these will have the shortest sum

    differences = lengths[:, None] - lengths[None, :]  # (buffer_size, buffer_size)
    distances = np.absolute(differences)

    indexes = np.argsort(distances, axis=-1)  # Sort in ascending order
    indexes = indexes[:, :batch_size]  # Keep only the first batch size entries
    distances = distances[indexes]  # Get those distances
    distances = np.sum(distances, axis=-1)  # Get the total distance using a particular centroid
    best_distance = np.argmin(distances, axis=0)  # Find the best centroid

    # Sometimes, multiple centroids have equal distance. This case takes care of that
    if best_distance.shape[0] > 1:
        best_distance = best_distance[0]

    # Return results
    batch_indexes = indexes[best_distance, :]  # Get the batch indexes

    return batch_indexes

test_data = np.array([1.0, 2.0, 10.0, 13.0])
print(select_batch(2, test_data))