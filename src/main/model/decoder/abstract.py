"""
The abstract descriptor for the decoder interface.

T
"""
import torch
from torch import nn
from ..registry import InterfaceRegistry
class ComputationalCore(nn.Module):
    """
    The computational core of the decoder process,
    this processes information at the level of the
    bottleneck. It is an abstract interface that
    must be implemented.
    """

class VocabularyCore:
    """
    The vocabulary core is the interface the model directly
    has with the rest of the world. It can consist of vocabulary,
    logits, and embeddings stripped from other models.
    """

class ExternalInterface(nn.Module):
    """
    Defines the external interface between the text
    stream, tokenizer, etc and the internal computation
    core. Several core features are defined here, including
    the size of the vocabulary, the embeddings, and the logit
    used to interact with it.

    Also tied into this are some other core features, such
    as the virtual layer selector
    """