"""
Define the core async batch processor classes we can utilize
"""

import asyncio
import torch
import uuid
import numpy as np
from torch import nn
from torch.nn import functional as F
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Callable, List, Optional
from .clustering import constrained_kmeans, assign_best_to_centroid
from ..data import ActionRequest

RequestBuffer = Dict[str, Tuple[asyncio.Future, ActionRequest]]
LoggingCallback = Callable[[str, int], None]

SHAPES_NAME = "shape"
TARGETS_NAME = "targets"
CONTEXT_NAME = "context"

###
# The clustering strategy collection are a few ways
# of figuring out what a more optimal batch might be like.
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

## Basic batch strategy classes
#
# These primarily have the responsibility of getting the statistics
# that the above clustering strategies need. At the moment, we base
# everything on position.
class BatchStrategy(ABC):
    """
    The abstract batch strategy class.

    The batch strategy class has the primary responsibility of
    extracting important statistics from items in the request
    buffer to allow the various clustering strategies to work.

    It is dependency-njected with a clustering strategy
    """
    def __init__(self,
                 clustering_strategy: ClusteringStrategy
                 ):
        self.clustering_strategy = clustering_strategy

    @abstractmethod
    def get_vital_statistics(self, requests: RequestBuffer)->torch.Tensor:
        """
        An extremely important method, this gives us vital information on
        the various items in the request buffer. It needs to be compatible
        with the clustering mechanism.

        Exact details may vary
        :return:
        """

    def forward(self,
                request_buffer: RequestBuffer,
                batch_size: int,
                logging_callback: LoggingCallback,
                force_selection: str,
                )->List[str]:
        vitals = self.get_vital_statistics(request_buffer)
        output = self.clustering_strategy(vitals, batch_size, logging_callback, force_selection)
        return output

class TransformerBatchStrategy(BatchStrategy):
    """
    Implements a batch strategy to handle content that
    is optimized to work with transformers.

    What this actually means is we look at the product
    of the shapes, since that is what content will flatten
    down to. We also look at the length of the context.

    We look to optimize the lengths so that when we
    flatten a tensor, each tensor has lengths as close
    together as possible.
    """
    def __init__(self,
                 clustering_strategy: ClusteringStrategy,
                 use_shape_info: bool,
                 use_context_info: bool = True
                 ):
        super().__init__(clustering_strategy)
        self.use_shape_info = use_shape_info
        self.use_context_info = use_context_info
    def get_vital_statistics(self, requests: RequestBuffer) ->torch.Tensor:
        vitals = {}
        for key, (_, request) in requests.items():
            statistics = []

            if self.use_shape_info:
                # Get the shape statistic. This will help us select
                # based on the generation target
                shape = request.subtask_details[SHAPES_NAME]
                statistics.append(float(torch.prod(shape)))

            if self.use_context_info:
                # We also need to consider how much padding
                # it is going to take to handle the context concatenation
                context = request.subtask_details[CONTEXT_NAME]
                statistics.append(context.shape[0])

            # Combine together. Then store
            statistics = torch.stack(statistics)
            vitals[key] = statistics
        return vitals

##
# Batch assembly and dissassembly classes.
#
# This actually gets the job of putting a batch together... or taking it back apart and responding.

class BatchAssembly(ABC):
    """
    The BatchAssembly class is an abstract class responsible for assembling
    the tensors in the request buffer into a coherent batch that can be processed
    by the model.

    This class defines the interface and contract for assembling a batch.
    Implementers of this class should handle any necessary padding and other
    batch-specific logic.
    """

    def __init__(self):
    @abstractmethod
    def __call__(self,
                 request_buffer: RequestBuffer,
                 uuids: List[str],
                 logging_callback: LoggingCallback
                 ) -> Tuple[Dict[str, torch.Tensor], List[Tuple[str, Callable]]]:
        """
        Assembles a batch of data from the selected requests.

        :param request_buffer: The Request Buffer containing pending requests.
        :param uuids: A list of UUIDs selected by the Batching Strategy.
        :param logging_callback: A callback function for logging, which accepts a
                                 message and a verbosity level.
        :return:
            - batch: A fully formed batch, which could be a tensor (or a set of tensors)
                     ready for processing. Implementers should handle any necessary padding.
            - metadata: A list of List[Tuple[UUID, Callable]], where each inner list
                        corresponds to an entry in the batch and associates the UUIDs
                        with their callbacks.
        """
        pass


class TransformerAssembly(BatchAssembly):
    """
    The transformer batch assembly mechanism.

    This class can be configured to provide context,
    shapes, and target data so long as the needed information
    is being passed along through the training pipeline.

    The shape and target information will be expected to be in the
    subtask_details dictionary of each ActionRequest. The context
    can be extracted from the state tracker feature.

    --- expected usage ---

    It is expected this will be used in a context-only configuration - for
    control flow - and in a context+shape config for eval, context+shape+targets
    for supervised training.
    """
    def __init__(self,
                 include_context: bool,
                 include_shapes: bool,
                 include_targets: bool
                 ):
        """
        :param include_context: Whether to include context info in the output dictionary
        :param include_shapes: Whether to include shape info in the output dictionary
        :param include_targets: Whether to include target info in the output dictionary.
        """
        super().__init__()
        self.include_context = include_context
        self.include_shape = include_shapes
        self.include_target = include_targets
