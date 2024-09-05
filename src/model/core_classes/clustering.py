from abc import ABC, abstractmethod

import numpy as np
import heapq
from typing import List, Callable, Tuple, Dict, Optional
from .types import LoggingCallback
import torch
class Measure(ABC):
    """
    Implements a measure of some sort, usable for clustering purposes.
    Will provide distances of points from centroids. Centroids and
    points may both be broadcastable.
    """
    @abstractmethod
    def __call__(self,
                 points: np.ndarray,
                 centroids: np.ndarray,
                 ) -> np.ndarray:
        """
        :param points: The points. Shape (n_samples, n_features)
        :param centroids: The centroids. Shape (n_centroids, n_features)
        :return:
            - Distances according to the measure
            - Shape (n_centroids, n_samples)
        """
        pass


class L1Distance(Measure):
    """
    A Computation of the L1 (Manhattan) distance measure.

    This assumes you can only move at right angles.
    """
    def __call__(self,
                 points: np.ndarray,
                 centroids: np.ndarray,
                 ) -> np.ndarray:
        """
        :param points: The points. Shape (n_samples, n_features)
        :param centroids: The centroids. Shape (n_centroids, n_features)
        :return: Distances from centroids according to L1 measure.
        """

        # Step 1: Prepare for broadcasting.
        # Ensure centroids ends up on the first dimension and points on the second.
        points = points[None, ...]
        centroids = centroids[:, None, :]

        # Step 2: Compute and return the measure
        difference = np.abs(points - centroids)
        distance = difference.sum(axis=-1)
        return distance


class L2Distance(Measure):
    """
    A measure appropriate for computing an L2 (Euclidean) distance.

    This is your standard Euclidean distance.
    """
    def __call__(self,
                 points: np.ndarray,
                 centroids: np.ndarray,
                 ) -> np.ndarray:
        """
        :param points: The points. Shape (n_samples, n_features)
        :param centroids: The centroids. Shape (n_centroids, n_features)
        :return: Distances from centroids according to L2 measure.
        """
        # Step 1: Prepare for broadcasting.
        # Ensure centroids end up on the first dimension and points on the second.
        points = points[None, ...]
        centroids = centroids[:, None, :]

        # Step 2: Compute and return the measure
        difference = points - centroids
        return np.linalg.norm(difference, axis=-1)


class BoundingOverlapDifference(Measure):
    """
    A specialized kind of measure useful when dealing
    with quantities that need to be ranked by both
    volume and shape.

    Consider two shapes with bounding boxes around them.
    These bounding boxes need not be 2D. For the following comparison
    computations, we consider these bounding boxes to be placed at the
    origin, overlapping, and with the same orientation.

    This metric calculates the difference between the overlapping volume
    of the bounding boxes and the volume of the bounding box required to
    contain both provided bounding box specifications.

    This is particularly useful when padding multidimensional arrays together,
    as it gives some indication of how much extra padding is needed. A larger
    result means a more significant mismatch. If the boxes fit exactly, then
    the result is zero.
    """
    def __call__(self,
                 points: np.ndarray,
                 centroids: np.ndarray,
                 ) -> np.ndarray:
        """
        :param points: The points. Shape (n_samples, n_features)
        :param centroids: The centroids. Shape (n_centroids, n_features)
        :return: The difference between the bounding box volumes.
        """
        # Step 1: Prepare for broadcasting.
        # Ensure centroids end up on the first dimension and points on the second.
        points = points[None, ...]
        centroids = centroids[:, None, :]

        # Step 2: Compute the minimum overlapping bounding box and the shared bounding box
        shared_minimum_dims = np.minimum(points, centroids)
        shared_maximum_dims = np.maximum(points, centroids)

        # Step 3: Compute the metric and return
        return np.prod(shared_maximum_dims, axis=-1) - np.prod(shared_minimum_dims, axis=-1)


def initialize_centroids_kmeans_pp(data: np.ndarray,
                                   k: int) -> np.ndarray:
    """
    Initialize centroids using the k-means++ method.

    The k-means++ algorithm improves the k-means initialization process by choosing
    initial centroids in a way that speeds up convergence and reduces the likelihood
    of poor clustering results. The key idea is to select the centroids in a way
    that spreads them out across the data points, increasing the chances of good
    clustering from the start.

    Steps:
    1. The first centroid is selected randomly from the data points.
    2. The remaining centroids are chosen from the data points with a probability
       proportional to the square of the distance from the nearest existing centroid.
    3. This process is repeated until k centroids are selected.

    :param data: np.ndarray
        The dataset as a 2D NumPy array with shape (n_samples, n_features).
    :param k: int
        The number of centroids to initialize.

    :return: np.ndarray
        An array of shape (k, n_features) containing the initialized centroids.
    """
    n_samples = data.shape[0]
    centroids = np.empty((k, data.shape[1]))

    # Step 1: Randomly select the first centroid from the data points
    centroids[0] = data[np.random.randint(0, n_samples)]

    for i in range(1, k):
        # Step 2: Compute the distance from each point to its nearest centroid
        distances= np.min(np.linalg.norm(data[:, np.newaxis] - centroids[:i], axis=2), axis=1)

        # Step 3: Square the distances and then calculate the probability of selection
        distances= distances + 1e-10 # Prevent division by zero.
        probabilities: np.ndarray = distances**2 / np.sum(distances**2)

        # Step 4: Use np.random.choice for multinomial selection from the data points
        centroids[i] = data[np.random.choice(n_samples, p=probabilities)]

    return centroids


def assign_best_to_centroids(datapoints: np.ndarray,
                             centroids: np.ndarray,
                             section_lengths: np.ndarray,
                             measure: Measure,
                             n: int,
                            )->Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    A function to use when assigning data to centroids based on
    the idea of keeping the best 'n' cases.

    An additional complication is the allowance of processing of
    corrolated centroid/datapoint sets using the same measure.
    In such circumstances, the different points and centroids can be concatenated
    together and processed by specifying the section length for each region
    - the regions are then separated, and run independently with their combined
    scores used to make decisions

    Steps:
        - 1) Break up datapoints and centroids into corrolated arrays using np.split
        - 2) Compute the distance to the centroids using the measure for all arrays, and add distances together
        - 3) Choose the topk lowest distances, get their indices.
        - 4) Select those datapoints. Return d

    :param datapoints:
        - The datapoint elements to draw information from.
        - Shape: (n_samples, n_features_compound)
        - Keep in mind n_features_compound may contain concatenation of multiple points
    :param centroids:
        -  The centroids to match to.
        - Shape: (n_features_compound)
        - Keep in mind n_features_compound may contain concatenation of multiple points
    :param section_lengths:
        - How many elements are used to specify a particular datapoint or centroid
        - Directly controls  a numpy.split. Keep this in mind.
    :param n:
        - How many data points to keep.
        - Keep in mind that we promise to return no more than this, NOT will return at least this.
    :param measure:
        - The measure to use when evaluating our cases.
        - Common ones are L2 distance or L1 distance.
    :return:
        - datapoints:
            - The selected datapoints
            - Note the datapoints are returned without being broken up again.
        - indices:
            - The indices these solutions were found at.
            - Shape (n), or shorter depending on amount of provided data.
        - distances:
            - The distance of each winner from the centroid.
    """

    # Step one - split into sections
    split_indices = np.cumsum(section_lengths)[:-1]  # Convert lengths to split indices
    point_arrays = np.split(datapoints, split_indices, axis=-1)
    centroid_arrays=  np.split(centroids, split_indices, axis=-1)

    # Step two - find combined lengths
    distances = np.zeros([datapoints.shape[0]])
    for point_array, centroid_array in zip(point_arrays, centroid_arrays):
        centroid_array = centroid_array[None, :] # metric expects a batch
        distances += measure(point_array, centroid_array).squeeze(axis=0)

    # Step three: select the 'n' lowest distances

    indices = np.argsort(distances)[:n]

    # Step four: get information and return results
    selected_datapoints = datapoints[indices, :]
    selected_distances = distances[indices]
    return selected_datapoints, indices, selected_distances


def stake_claims(claims: np.ndarray,
                 distances: np.ndarray,
                 n: int,
                 ):
    """
    A support function for constrained match. This section performs the "stake claims"
    process, where each centroid gets the opportunity to claim a number of unclaimed
    datapoints or reinforce their claim on points they already hold. Centroids can claim
    up to `n` points.

    Steps:
        1) Generate a mask (options_mask) that marks which points a centroid can claim.
           A centroid can either:
           - Claim unclaimed points, or
           - Retain its existing claims.
        2) Apply the mask to the distance matrix, setting distances to infinity for
           points that cannot be claimed. This ensures the sorting algorithm prioritizes
           valid options.
        3) Use `np.argsort` to find the top `n` points with the smallest distances
           that the centroid can claim.
        4) Update the claims matrix to reflect new claims. Centroids that attempt to
           claim the same point will "fight" for the best option in a subsequent step.

    :param claims:
        - A boolean matrix tracking which centroid is claiming each datapoint.
        - Shape: (centroids, datapoints).
        - True means that centroid is claiming that datapoint.

    :param distances:
        - The distances matrix. Indicates the distance between each centroid and
          each datapoint. The algorithm uses this matrix to determine which points
          each centroid should claim (based on proximity).
        - Shape: (centroids, datapoints).

    :param n:
        - The constraint: A centroid can claim at most `n` points. This ensures
          no centroid claims too many points.
    """

    ##
    # Step 1: Create a mask that indicates which points can be claimed by each centroid.
    # The mask (options_mask) will be True if the point is either:
    #   - Unclaimed by any centroid, or
    #   - Already claimed by this centroid.
    #
    # This ensures centroids cannot steal each other's claims but can retain their
    # current claims or make new claims on unclaimed points.
    ##

    unclaimed_mask = ~np.any(claims, axis=0, keepdims=True)
    options_mask = claims | unclaimed_mask

    ##
    # Step 2: Mask the distance matrix to exclude points that cannot be claimed.
    # Points that cannot be claimed (i.e., points already claimed by other centroids)
    # are set to infinity (np.inf). This ensures they are effectively ignored during
    # the claim selection process.
    ##

    masked_distances = np.where(options_mask, distances, np.inf)

    ##
    # Step 3: Use np.argpartition to select the `n` smallest distances for each centroid.
    # This is more efficient than sorting the entire array, as we only need the top `n`.
    # np.argpartition ensures the `n` smallest values are returned, but not fully sorted.
    # This step gives us the indices of the top `n` closest points for each centroid.
    ##

    top_indices = np.argsort(masked_distances,axis=-1)[:, :n]

    ##
    # Step 4: Update the claims matrix to reflect the new claims.
    #
    # Create a mask (stakes_mask) based on the top `n` selected indices. This mask
    # is True for points that the centroid wants to claim (either because they are
    # new claims or reinforcing existing ones).
    #
    # Since centroids cannot claim points already held by another centroid, the
    # final mask (stakes_mask) is refined by using the options_mask.
    # The resulting stakes_mask is then used to update the claims matrix.
    ##

    centroid_vector_index = np.arange(distances.shape[0])
    centroid_vector_index = np.expand_dims(centroid_vector_index, axis=1).repeat(top_indices.shape[-1], axis=-1)

    stakes_mask = np.zeros_like(claims)
    stakes_mask[centroid_vector_index, top_indices] = True
    stakes_mask = stakes_mask & options_mask

    # Update the claims matrix with the new valid claims.
    claims[stakes_mask] = True
def resolve_fights(claims: np.ndarray,
                   distances: np.ndarray):
    """
    A support function used by constrained matches. `resolve_fights` resolves the fights
    that can occur when more than one centroid stakes a claim to a point. The point is used
    to cast the deciding vote regarding which centroid gets it, by associating it with
    the centroid that is closest.

    Steps:
        1) Detect fights: Detect any region where more than one centroid stakes a claim
           to a point.
        2) Extract fights: Extract the relevant distance sections and a view into
           the claims matrix. The view can later be used to update the results.
        3) Resolve fights: By sorting the extracted distances, figure out which
           centroid has the best claim.
        4) Commit results: Set all claims for the points in question to False, then
           assign the correct winning centroid claim to True.

    :param claims:
        - The entire claims matrix. Boolean.
        - A "fight" is detected when more than one centroid is claiming a datapoint.
        - Shape (centroids, datapoints).
    :param distances:
        - The distances matrix. Indicates the distance of points from centroids.
        - Shape (centroids, datapoints).
    :effect: Modifies claims in place, updating with resolved ownership of points.
    """

    # Step 1: Detect fights (datapoints claimed by more than one centroid)
    fight_mask = claims.sum(axis=0) > 1  # True where more than one centroid claims the point
    fight_indices = np.where(fight_mask)[0]  # Get the indices of points with fights

    if fight_indices.size == 0:
        return  # No fights, return early

    # Step 2: Extract fight distances and corresponding claim sections
    fight_distances = distances[:, fight_indices]  # Get distances for points with fights
    fight_claims = claims[:, fight_indices]  # Get claims matrix for the same points

    # Step 3: Resolve fights by choosing the centroid with the smallest distance for each point
    # Sort the distances and get the indices of the centroids with the smallest distance
    winners = np.argmin(fight_distances, axis=0)  # Best (smallest) distance per fight

    # Step 4: Commit results
    # Set all claims for fighting points to False
    claims[:, fight_indices] = False

    # Assign the winner for each fight by setting the correct claims to True
    claims[winners, fight_indices] = True

def constrained_match(distances: np.ndarray,
                      n: int,
                      ) -> np.ndarray:
    """
    Finds the best constrained match for the given distance table. This
    means finding the collection of points which are associated with each
    centroid, and for which each centroid may not have more than `n` points
    associated with them, that minimizes the total distance between centroids
    and points over all clusters.

    This function now only returns the claims matrix, which shows which
    centroid has claimed which point after the matching process has been completed.

    Steps:
    - Create a master "claims" array in which to track what has claimed what, alongside
      to detect when multiple centroids are claiming the same point.
    - Loop:
        - From the claims array, extract the claims present per centroid, and
          the unclaimed points.
        - Concatenate, and let the centroids claim more points. Place into claims array.
        - Resolve any fights in favor of the centroid with the lowest distance that
          is not full.
    - Return the final claims matrix.

    :param distances:
        The distance matrix between centroids and points. Shape (centroids, datapoints).
    :param n:
        The maximum number of points that each centroid can claim.
    :return:
        claims: The final claims matrix after resolving all fights.
                Shape: (centroids, datapoints), with True indicating a claim.
    """

    # Set up tracking: claims matrix to track centroid claims on points
    claims = np.zeros_like(distances, dtype=bool)
    all_points_claimed = lambda: np.all(np.any(claims, axis=-1))
    all_centroids_full = lambda: np.all(n <= claims.sum(axis=-1))

    # While points remain unclaimed and not all centroids are full,
    # stake claims per centroid, then resolve any fights.
    while not (all_points_claimed() or all_centroids_full()):
        stake_claims(claims, distances, n)
        resolve_fights(claims, distances)

    # Return the final claims matrix
    return claims



def constrained_kmeans(datapoints: np.ndarray,
                       section_lengths: np.ndarray,
                       k: int,
                       n: int,
                       measure: Measure,
                       max_iters: int = 100,
                       tolerance: float = 0.0001,
                       )->List[Dict[str, np.ndarray]]:
    """
    The constrained kmeans algorithm is similar in behavior to
    the standard kmeans, but contains a notable twist. This
    twist is that it cannot create clusters of a certain size.

    It pledges to find the optimal distances given these restrictions.

    :param datapoints:
        - The datapoint elements to draw information from.
        - Shape: (n_samples, n_features_compound)
        - Keep in mind n_features_compound may contain concatenation of multiple points
    :param section_lengths:
        - How many elements are used to specify a particular datapoint or centroid
        - Directly controls  a numpy.split. Keep this in mind.
    :param k:
        - The number of clusters to create
    :param n:
        - How many data points to keep at maximum per cluster
        - Keep in mind that we promise to return no more than this, NOT will return at least this.
    :param measure:
        - The measure to use when evaluating our cases.
        - Common ones are L2 distance or L1 distance.
    :param max_iters:
        - The number of iterations to go through, at maximum.
    :param tolerance:
        - The tolerance
        - When gains are under this, we end generation.
    """
    # Step one.
    #
    # This is setup. We split the points into their separate sections,
    # and initialize the centroids to decent default values. We also
    # go ahead and initialize the last_total_std used to check threshold
    # conditions

    split_indices = np.cumsum(section_lengths)[:-1]  # Convert lengths to split indices
    point_arrays = np.split(datapoints, split_indices, axis=-1)
    centroid_arrays = [initialize_centroids_kmeans_pp(point_array, k) for point_array in point_arrays]
    centroids = np.concatenate(centroid_arrays, axis=-1)

    last_total_distance = np.inf

    # Run constrained kmeans process
    for i in range(max_iters):

        ## Step 2:
        #  Compute the distance matrices for each array subset,
        #  and combine them together to get the overall distances

        distances = np.zeros([k, datapoints.shape[0]])
        centroid_arrays = np.split(centroids, split_indices, axis=-1)
        for point_array, centroid_array in zip(point_arrays, centroid_arrays):
            distances += measure(point_array, centroid_array)

        ##
        # Step 3:
        #
        # We get the claims matrix, then decode it into several
        # important quantities such as the optimal points index,
        # points mask, and unselected points index

        claims_matrix = constrained_match(distances, n)
        point_indices = np.argsort(claims_matrix, axis=1)[:, -n:]

        centroids_index = np.expand_dims(np.arange(claims_matrix.shape[0]), axis=-1)
        centroid_indices = np.broadcast_to(centroids_index, point_indices.shape)
        point_mask = claims_matrix[centroid_indices, point_indices]

        ##
        # Step 4:
        #
        # Based on the claims, recompute our total distances, and decide
        # whether it would be better to break here

        distance_clusters = distances[centroid_indices, point_indices]
        distance_per_centroid = np.sum(distance_clusters*point_mask, axis=-1)
        new_total_distance = np.sum(distance_per_centroid)
        if (last_total_distance - new_total_distance) < tolerance:
            break
        last_total_distance = new_total_distance

        # Step 5:
        #
        # Recompute the centroids as a mean of the selected datapoints,
        # while ignoring the masked entries.

        point_clusters = datapoints.take(point_indices, axis=0) # shape: centroid, points, data_dim
        expanded_mask = np.expand_dims(point_mask, axis=-1)
        centroids = np.sum(point_clusters*expanded_mask, axis=1)/(np.sum(expanded_mask, axis=1)+ 1e-9)

    # Step 6:
    #
    # Perform actual production of results. For efficiencies sake,
    # the kmeans process runs using masks rather than split lists or dicts,
    # but we split up the output into such.

    datapoint_clusters = datapoints.take(point_indices, axis=0)
    clusters = []
    for i in range(k):
        cluster = {}

        # Get and store the centroid associated with this
        # cluster, and the total distance for that centroid.

        cluster["centroid"] = centroids[i, :]
        cluster["total_distance"] = distance_per_centroid[i]

        # Get and store all datapoints that are assigned within the
        # cluster, and the indices of these datapoints. This will
        # include shortening them with a mask

        cluster_datapoints = datapoint_clusters[i, :]
        cluster_indices = point_indices[i, :]
        cluster_mask = point_mask[i, :]

        cluster["datapoints"] = cluster_datapoints[cluster_mask]
        cluster["indices"] = cluster_indices[cluster_mask]

        # Append
        clusters.append(cluster)
    return clusters


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
