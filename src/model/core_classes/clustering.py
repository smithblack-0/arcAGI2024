from abc import ABC, abstractmethod

import numpy as np
import heapq
from typing import List, Callable, Tuple, Dict, Optional
from .types import LoggingCallback
import torch


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


class ClusteringStrategy(ABC):
    """
    The clustering strategy class definition

    A clustering strategy is exactly what it sounds like
    - a method of defining clusters. potential batch cases
    will have various important dimensions, and we can see
    these dimensions and act to minimize padding.
    """
    @abstractmethod
    def __call__(self,
                vitals: Dict[str, torch.Tensor],
                cluster_size: int,
                logging_callback: LoggingCallback,
                force_selection: Optional[str]
                )->List[str]:
        """
        Creates a list of items which should be placed in a batch together.

        :param vitals: A mapping of the ids of a case to the important dimension statistics for it.
        :param cluster_size: The size of the cluster to form
        :param logging_callback: The logging callback function
        :param force_selection: This id must be included in the returned cluster.
        :return: A list of ids. They represent a cluster to form.
        """


class QueueClusteringStrategy(ClusteringStrategy):
    """
    A very simple clustering strategy based on a queue.
    It completely ignores the dimensions information, and instead
    just returns the oldest content first.
    """
    def __init__(self):
        super().__init__()


    def extract_forced(self,
                       vitals: Dict[str, torch.Tensor],
                       extract: str,
                       )->Dict[str, torch.Tensor]:
        """
        Extracts from a vital dictionary elements which
        may be forced to be included. Note this modifies the
        original dictionary.

        :param vitals: The vitals dictionary
        :param extract: The element to extract
        :return: The extracted elements
        :effect: Modifies vitals dictionary
        """
        output = {}
        for item in extract:
            output[item] = vitals.pop(item)
        return output

    def __call__(self,
                vitals: Dict[str, torch.Tensor],
                cluster_size: int,
                logging_callback: LoggingCallback,
                force_inclusion: Optional[str],
                ) -> List[str]:
        """
        Creates a list of items which should be placed in a batch together.
        It uses the oldest items first, except in the case that we are dealing with
        forced inclusion. In which case, we include what we need, then everything else
        is built out of the oldest items first.

        :param vitals: A mapping of the ids of a case to the important dimension statistics for it.
        :param cluster_size: The size of the cluster to form
        :param logging_callback: The logging callback function
        :param force_inclusion: These ids must be included in the returned cluster, if they exist
        :return: A list of ids. They represent a cluster to form.
        """

        forced_vitals = self.extract_forced(vitals, force_inclusion)
        output = list(forced_vitals.keys())
        keys = list(vitals.keys())
        while len(output) < cluster_size:
            output.append(keys.pop(0))
        return output


class ConstrainedKmeansStrategy(ClusteringStrategy):
    """
    Uses a constrained Kmeans clustering strategy.

    This uses a form of kmeans in which we constrain
    the maximum size each cluster can go. We may rarely
    return batch sizes that are smaller than the target

    It supports several measures

    ---- measures ----

    L2: Simple l2 distance.
    L1: The L1 distance. No eucledian space for you
    StatVolDiff:
        The difference between the volume needed to hold both shapes,
        and the portion of the volume that is common.
    """
    def __init__(self,
                 measure: str,
                 num_iterations: int = 6,
                 ):

        super().__init__()
        assert measure in ("L2", "L1", "StatVolDiff")
        self.measure = measure
        self.num_iterations = num_iterations

    def get_measure(self)->Callable[[np.ndarray, np.ndarray], np.ndarray]:
        if self.measure == "L2":
            return lambda points, centroid : np.linalg.norm(centroid - points)
        elif self.measure == "L1":
            return lambda points, centroid : (centroid-points).abs().sum()
        elif self.measure == "StatVolDiff":
            def outcome(points: np.ndarray, centroid: np.ndarray)->np.ndarray:
                # Broadcast centroid to match points.
                centroid = centroid + np.zeros_like(points)

                # Figure out minimum common dims, and bounding shape
                min_bounding_shape = np.minimum(points, centroid)
                bounding_shape = np.maximum(points, centroid)

                # Take the difference of their product
                return np.prod(bounding_shape, dim=-1) - np.prod(min_bounding_shape, dim=-1)
            return outcome
        else:
            raise RuntimeError("Illegal measure defined")
    def select_ids_from_indices(self,
                                vitals: Dict[str, torch.Tensor],
                                indices: np.ndarray,
                                )->List[str]:
        """
        Selects, based on a collection of indices, what ids
        need to be associated with them.

        :param vitals: The core vitals we are working with
        :param indices: A 1d array holding the indices to select
        :return: A list with the selected indices
        """
        assert indices.ndim == 1
        output = []
        keys = list(vitals.keys())
        for index in indices:
            output.append(keys[index])
        return output

    def kmeans_cluster_selection(self,
                                 array_vitals: np.ndarray,
                                 cluster_size: int,
                                 num_clusters: int,
                                 )->np.ndarray:
        """
        K means cluster selections. We make num_cluster by kmeans all of
        cluster size. Then we choose the one with the lowest variance.

        :param array_vitals: The vital statistics to cluster
        :param cluster_size: Size of each cluster
        :param num_clusters: The number of clusters to make
        :return: The selected indices
        """
        (_, std), clusters = constrained_kmeans(array_vitals,
                                                cluster_size,
                                                num_clusters,
                                                self.num_iterations,
                                                self.get_measure())
        best_cluster_index = np.argmin(std)
        best_cluster = clusters[best_cluster_index]
        return best_cluster["indices"]

    def forced_cluster_selection(self,
                                 array_vitals: np.ndarray,
                                 cluster_size: int,
                                 forced_index: int,
                                 )->np.ndarray:
        """
        Selects the best collection of points to build
        a batch from given the constraints that

        * it must include forced selection
        * it should be at maximum cluster size.

        :param array_vitals: The vital statistics indicating the index and their associated content
        :param cluster_size: The maximum size a cluster can reach
        :param logging_callback: The logging callback.
        :param forced_index: The index to forcibly include.
        :return: A cluster
        """
        # We operate basically by setting our centroid to be
        # the forced index, then building a cluster around it.

        centroid = array_vitals[forced_index]
        cluster = assign_best_to_centroid(array_vitals,
                                          centroid,
                                          cluster_size,
                                          self.get_measure())
        return cluster["indices"]
    def __call__(self,
                vitals: Dict[str, torch.Tensor],
                cluster_size: int,
                logging_callback: LoggingCallback,
                force_selection: Optional[str]
                )->List[str]:
        """
        Creates a list of items which should be placed in a batch together.

        :param vitals: A mapping of the ids of a case to the important dimension statistics for it.
        :param cluster_size: The size of the cluster to form
        :param logging_callback: The logging callback function
        :param force_selection: This id must be included in the returned cluster.
        :return: A list of ids. They represent a cluster to form.
        """

        array_vitals = np.stack([vitals[key].numpy() for key in vitals.keys()], dim=0)
        if force_selection is not None:
            # Handle forced construction of batches.
            #
            # This may sometimes result in batches that are too short.
            target = list(vitals.keys()).index(force_selection)
            indices = self.forced_cluster_selection(array_vitals,
                                                    cluster_size,
                                                    target,
                                                    )
            if indices.shape[0] < cluster_size:
                msg = f"Batch could not be made of size {cluster_size} when forcing target {force_selection}"
                logging_callback(msg, 2)
            msg = f"Batch construction was forced with target id '{force_selection}"
            logging_callback(msg, 3)
        else:
            # Handle clustering by kmeans
            #
            # We do a few brief rounds of constricted kmeans with the vital
            # statistics to select batches that have close dimensions. Then
            # we select those
            if not len(vitals) % cluster_size == 0:
                msg = f"None-forced batch construction requires buffer threshold to be divisible by batch size"
                raise ValueError(msg)
            num_clusters = len(vitals) // cluster_size
            indices = self.kmeans_cluster_selection(array_vitals, cluster_size, num_clusters)

        # Convert indices back to names and return
        output = self.select_ids_from_indices(vitals, indices)
        return output
