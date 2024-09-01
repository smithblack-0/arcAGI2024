import torch
from torch import nn
from typing import Tuple, Dict
from abc import ABC, abstractmethod
from ..adapters.io_adapters import IOAdapter


from ..adapters.distribution_adapters import DistributionAdapter
class BlockDecoderAdapter(ABC, nn.Module):
    """
    Responsible for decoding a particular preprocessed
    block of content into either it's supervised results
    or it's sampled results.-
    """

    @abstractmethod
    def supervised(self,
                shapes: torch.Tensor,
                context: Tuple[torch.Tensor, torch.Tensor],
                targets: Tuple[torch.Tensor, torch.Tensor],
                metrics: Dict[str, torch.Tensor],
                *extras
                )-> Tuple[torch.Tensor, torch.Tensor]:
        """
        The supervised block decoder mechanism is expected
        to, as it's namesake suggests, use supervised learning
        in order to learn how to generate an appropriate block.

        The mechanism is provided with the expected block shapes,
        the common context to generate from, and the generation targets.
        It is up to the decoder to decide the generation strategy,
        which could be everything from noise removal to next sequence
        prediction. There are not any firm restrictions.

        :param shapes:
            * The batched shapes specification. It indicates the shape of each block being processed.
            * It will have shape (batch, D), where D is the number of dimensions and each dimension is a
              positive integer
            * It is up to the model to make use of this, if needed. Not all models may need this information,
              - for instance a diffusion model can probably get by without it.
        :param context:
            * The common context to use while generating. This is where we supposed to get information on
              what to produce
            * Note that whatever generative model ends up being used, it had better be able to consume a
              transformer-like context to get the job done.
            * The first tensor is the actual context, the second one is the padding mask, and should be used
              to mask attention.
            * Shapes are (batch, items, common_embedding_dim) and (batch, items)
        :param targets:
            * The generation targets. Usually used for teacher forcing, but can also be used for other
              strategies such as denoising, etc.
            * Note that the generation targets have been flattened into one dimensional data. If
              you need to restore your dimensions, you can rebuild it using the shape tensor.
            * Two entries exist.
                * One defines the actual data, in a (batch, items, ...) format.
                * One defines the padding mask, in a (batch, items) format.
                * The padding mask is true if an item is NOT padding.
        :param metrics:
            One can append metrics into the provided dictionary, in the form of tensors.
            Be warned, however - those metrics must be defined per batch!
        :return: Two items must be returned. These must be the block and the loss
            * block:
                * The block will have the same shape of targets. The unmasked target
                  locations will be extracted and used as context embeddings
                * Shape: (batch, ..., embeddings)
            * loss:
                * The per-batch loss. This will be based on the target
                * Shape: (batch, ..., embeddings)
        """

    @abstractmethod
    def decode(self,
               shapes: torch.Tensor,
               context: Tuple[torch.Tensor, torch.Tensor],
               noise: Tuple[torch.Tensor, torch.Tensor],
               metrics: Dict[str, torch.Tensor],
               )->torch.Tensor:
        """
        The unsupervised decode mechanism, which should use
        your build-in sampling method.

        :param shapes:
            * The batched shapes specification. It indicates the shape of each block being processed.
            * It will have shape (batch, D), where D is the number of dimensions and each dimension is a
              positive integer
            * It is up to the model to make use of this, if needed. Not all models may need this information,
              - for instance a diffusion model can probably get by without it.
        :param context:
            * The common context to use while generating. This is where we supposed to get information on
              what to produce
            * Note that whatever generative model ends up being used, it had better be able to consume a
              transformer-like context to get the job done.
            * The first tensor is the actual context, the second one is the padding mask, and should be used
              to mask attention.
            * Shapes are (batch, items, common_embedding_dim) and (batch, items)
        :param noise:
            * This has the shape of targets, but contains normal noise. Use, or ignore, as you
              wish.
        :param metrics:
            One can append metrics into the provided dictionary, in the form of tensors.
            Be warned, however - those metrics must be defined per batch!
        :return: Two items must be returned. These must be the block and the loss
            * block:
                * The block will have the same shape of targets. The unmasked target
                  locations will be extracted and used as context embeddings
                * Shape: (batch, ..., embeddings)
            * loss:
                * The per-batch loss. This will be based on the target
                * Shape: (batch, ..., embeddings)
        """


class TransformerDecoder(BlockDecoderAdapter):
    """
    A decoder adapter specialized to use a transformer-based,
    flat model to process provided content.


    """
    def __init__(self,
                 io: IOAdapter,
                 distribution: DistributionAdapter,
                 patching: "PatchingAdapter"
                 ):
        """

        :param io:
        :param distribution:
        :param patching:
        """

