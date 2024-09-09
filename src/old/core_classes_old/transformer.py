"""
Transformer-based async batch processor support structures are located
within this file. This generally is meant to contain information
and mechanisms that will flatten flowing data for processing
by trasformers suchas ViT or just your average transforemr decoder

The models are assumed to be operational and flowing around with certain
particularities in their data flow. In detail, it is assumed we are using
block decoding, there is likely a context, and there may be targets.

"""
from typing import Dict, List, Tuple, Callable

import torch
from torch.nn import functional as F

from src.old.core_classes_old.async_processing import BatchStrategy, RequestBuffer, SHAPES_NAME, \
    CONTEXT_NAME, LoggingCallback, TARGETS_NAME
from src.old.core_classes_old.clustering import ClusteringStrategy
from src.old.core_classes_old.core_processer import BatchAssembly


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
    def get_vital_statistics(self, requests: RequestBuffer) ->Dict[str, torch.Tensor]:
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
            statistics = torch.tensor(statistics)
            vitals[key] = statistics
        return vitals


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
                 include_targets: bool,
                 num_channel_dim: int
                 ):
        """
        :param include_context: Whether to include context info in the output dictionary
        :param include_shapes: Whether to include shape info in the output dictionary
        :param include_targets: Whether to include target info in the output dictionary.
        :param num_channel_dim: The number of channel dimensions.
                                  Everything before that will be flattened.

        """
        super().__init__()

        self.include_context = include_context
        self.include_shape = include_shapes
        self.include_target = include_targets
        self.num_embedding_dim = num_channel_dim

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
        :effect: Modifies the request buffer to remove used
        """

        # Define content accumulators
        metadata = []
        shapes = [] if self.include_shape else None
        targets = [] if self.include_target else None
        contexts = [] if self.include_context else None

        # Define max length trackers. These will be used in the padding
        # step.

        targets_max_length = 0 if self.include_target else None
        context_max_length = 0 if self.include_context else None

        # Go and get all the information.
        #
        # Track as well the padding targets, to whatever degree is relevant
        for uuid in uuids:
            future, action_request = request_buffer.pop(uuid)
            metadata.append((uuid, future))

            if self.include_context:
                # Handle context extraction. This includes updating the padding targets
                # and getting the actual information
                context = action_request.state_tracker.context
                context_max_length = max(context_max_length, context.shape[0])
                contexts.append(context)

            if self.include_shape:
                # Handle shape extraction. Shapes also are used to track target
                # lengths
                shape = action_request.subtask_details[SHAPES_NAME]
                shapes.append(shape)

            if self.include_target:
                target = action_request.subtask_details[TARGETS_NAME]
                target = target.flatten(0, -(self.num_embedding_dim + 1))
                targets_max_length = max(targets_max_length, target.shape[0])

        batched = {}
        if self.include_shape:
            shapes = torch.stack(shapes, dim=0)
            batched["shape"] = shapes

        if self.include_context:
            contexts = [F.pad(context, (0, context_max_length - context.shape[0])) for context in contexts]
            contexts = torch.stack(contexts, dim=0)
            batched["context"] = contexts

        if self.include_target:
            targets = [F.pad(target, (0, targets_max_length-target.shape[0])) for target in targets]
            targets = torch.stack(targets, dim=0)
            batched["targets"] = targets

        return batched, metadata
