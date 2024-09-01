import numpy as np
import heapq
from typing import List, Callable, Tuple, Dict


def initialize_centroids_kmeans_pp(data, k):
    """ Initialize centroids using k-means++ method """
    n_samples = data.shape[0]
    centroids = np.empty((k, data.shape[1]))
    centroids[0] = data[np.random.randint(0, n_samples)]

    for i in range(1, k):
        distances = np.min(np.linalg.norm(data[:, np.newaxis] - centroids[:i], axis=2), axis=1)
        probabilities = distances / distances.sum()
        centroids[i] = data[np.searchsorted(np.cumsum(probabilities), np.random.rand())]

    return centroids

def assign_best_to_centroid(datapoints: np.ndarray,
                            centroid: np.ndarray,
                            num_kept: int,
                            measure: Callable[[np.ndarray, np.ndarray], np.ndarray],
                            )->Dict[str, np.ndarray]:
    """
    Assigns the best 'N' datapoints to be part of the given centroids. Returns it
    as a cluster.

    :param datapoints: The datapoints to draw from
    :param centroid: The centroid to build around
    :param num_kept: How many datapoints can fit in the centroid
    :return: A cluster
    """

    heap = []
    distances = measure(datapoints, centroid)
    priorities = -distances
    for i in range(datapoints.shape[0]):
        # Get features
        priority = priorities[i]
        point = datapoints[i]

        # Push onto heap, then if needed shrink heap
        heapq.heappush(heap, (priority, point, i ))
        if len(heap) > num_kept:
            heapq.heappop(heap)

    # Reformat and return
    priorities, points, indices = zip(*heap)
    cluster = {"priorities" : np.ndarray(list(priorities)),
               "points" : np.ndarray(list(points)),
               "indices" : np.ndarray(list(indices))}
    return cluster


def assign_data_to_centroids(datapoint: np.ndarray,
                            data_index: int,
                            centroids: np.ndarray,
                            centroid_buckets: List[List[Tuple[float, np.ndarray, int]]],
                            measure: Callable[[np.ndarray, np.ndarray], np.ndarray],
                            max_bucket_length: int):
    """
    Assign a datapoint to a centroid's bucket, rearranging if necessary.
    """
    distances = measure(datapoint, centroids)
    priorities = -distances
    centroid_priorities = np.argsort(distances)

    for index in centroid_priorities:
        heap = centroid_buckets[index]
        priority = priorities[index]

        # Add the datapoint and its index to the priority queue
        heapq.heappush(heap, (priority, datapoint, data_index))

        # If the bucket exceeds the maximum length, rearrange
        if len(heap) > max_bucket_length:
            _, point, idx = heapq.heappop(heap)
            if point is datapoint:
                continue
            else:
                assign_data_to_centroids(point, idx, centroids, centroid_buckets, measure, max_bucket_length)
                return
        else:
            return
    raise RuntimeError("Not able to assign a datapoint.")


def constrained_kmeans(data: np.ndarray,
                       k: int,
                       max_cluster_size: int,
                       measure: Callable[[np.ndarray, np.ndarray], np.ndarray],
                       max_iters: int = 100,
                        ) -> Tuple[
                                    Tuple[np.ndarray, np.ndarray],
                                    Dict[int, Dict[str, np.ndarray]],
                                    ]:
    """
    Constrained k-means clustering using the assign_data_to_centroid function.
    """
    n_samples, n_features = data.shape
    centroids = initialize_centroids_kmeans_pp(data, k)
    centroid_buckets = {i: [] for i in range(k)}  # Make sure centroid buckets are always available.

    for iteration in range(max_iters):
        # Shuffle indices instead of data to avoid unnecessary data copying
        data_indices = np.arange(n_samples)
        np.random.shuffle(data_indices)

        centroid_buckets = {i: [] for i in range(k)}  # Reinitialize buckets for each iteration

        for index in data_indices:
            point = data[index]
            assign_data_to_centroids(point, index, centroids, centroid_buckets, measure, max_cluster_size)

        # Update centroids based on current clusters
        for i in range(k):
            if centroid_buckets[i]:
                points = np.array([item[1] for item in centroid_buckets[i]])
                centroids[i] = points.mean(axis=0)
            else:
                # Handle empty clusters by reinitializing the centroid
                centroids[i] = data[np.random.randint(0, n_samples)]

    # Final cluster assignments and computations of std
    clusters = {}
    for i in range(k):
        bucket = centroid_buckets[i]

        priorities = []
        points = []
        indices = []
        for j in range(len(bucket)):
            priority, point, index = bucket[j]
            priorities.append(priority)
            points.append(point)
            indices.append(index)

        priorities = np.array(priorities)
        points = np.array(points)
        indices = np.array(indices)

        clusters[i] = {"priorities" : priorities, "points" : points, "indices" : indices}
    std = np.array([np.std(clusters[key]["points"]) for key in clusters])
    return (centroids, std), clusters

