"""
Data converters are dedicated to converting
input leafs of their responsibility into a
tensor of embeddings. Or vice versa
"""
import torch
import numpy as np
from torch import nn
from typing import Any, List, Callable, Tuple, Generic, TypeVar, Dict
from dataclasses import dataclass

## Payload definitions and converters.

@dataclass
class PayloadObject:
    """
    A payload object consists of a type
    and a payload that is then processed.
    """
    type: str
    payload: Any
LatentRep = TypeVar("LatentRep")

@dataclass
class PayloadEmbedding:
    """
    An embedded payload.
    """
    embeddings: torch.Tensor # Shape (batch, N, embed_dim)
    mask: torch.Tensor # Shape (Batch, N). Bool tensor.


# Define helper functions.

def walk_and_apply(pytree: Any, predicates: List[Callable], funcs: List[Callable]):
    """
    Walks through a pytree (nested structure) and applies associated func when predicate
    is matched.

    Args:
        pytree: The nested structure (lists, tuples, dicts).
        predicate: A function that takes a node and returns True if the node matches the condition.
        func: A function to apply to nodes that match the predicate.

    Returns:
        A new pytree with func applied to matching nodes.
    """
    for predicate, func in zip(predicates, funcs):
        if predicate(pytree):
            return func(pytree)
    if isinstance(pytree, dict):
        return {k: walk_and_apply(v, predicate, func) for k, v in pytree.items()}
    elif isinstance(pytree, (list, tuple)):
        return type(pytree)(walk_and_apply(v, predicate, func) for v in pytree)
    else:
        raise TypeError("Unhandled type encountered")

def convert_data_to_payload_tree(pytree: Any)->Any:
    """
    Converts a native pytree structure into payload format.

    We look for dictionaries that consist of a "payload" and "type"
    object.
    :param pytree:
    :return:
    """
    def predicate(pytree):
        if not isinstance(pytree, dict):
            return False
        if "type" not in pytree or "payload" not in pytree:
            return False
        return True

    def operand(pytree):
        return PayloadObject(pytree["type"], pytree["payload"])
    return walk_and_apply(pytree, [predicate], [operand])

def convert_payload_tree_to_pytree(payload_tree: Any)->Any:
    """
    Converts a payload tree into pytree format.

    Each payload object is replaced with a {type, payload} object.

    :param payload_tree: the payload tree to convert
    :return: Thje converted tre
    """
    predicate = lambda x : isinstance(x, PayloadObject)
    operand = lambda x : {"type" : x.type, "payload" : x.payload}
    return walk_and_apply(payload_tree, [predicate], [operand])

# Encoder/Decoder contracts.

class AbstractPayloadEncoder(nn.Module):
    """
    The abstract encoder defines the contract that data encoders must
    face. The general promise is:
        - You are handed in a payload of some sort
        - You are expected to return a tensor of embeddings and a mask

    Be careful to encode any extra needed details so you can generate the correct output.
    Also, note we do not emit a latent representation.
    """
    def __init__(self, embedding_dim: int):
        """
        :param embedding_dim: The dimensionality of the embeddings
        """
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, payload: Any)->PayloadEmbedding:
        """
        :param payload: The payload to convert to embeddings. Shape batch x ragged..
        :return: The payload embeddings, shape (batch, N, embedding_dim)
        :return: A payload mask. Indicates what embeddings were active. True where active.
        """
        raise NotImplementedError("You need to implement this")


class AbstractPayloadDecoder(nn.Module, Generic[LatentRep]):

    """
    The abstract decoder is responsible for decoding a provided
    latent representation into the standard representation.

    The general promise is:
        - Give me the latent rep, a decoding key, and a payload type.
        - I give you back a payload object

    We also have predicate logits to handle as well:
        - When handed a key, one per batch
        - I give you the probability that that key corrolated with this type of payload.
    """
    def selection_logits(self, key: torch.Tensor)->torch.Tensor:
        """
        Produces a logit for selection purposes for all elements of a batch.

        :param key: a tensor, of probable shape (..., embedding).
        :return: A tensor of shape (batch) containing predicate logits.
        """
        raise NotImplementedError(" A subclass needs to implement predicate logits")

    def __init__(self,
                 embedding_dim: int
                 ):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self,
                key: torch.Tensor,
                latent_rep: LatentRep,
                payload_type: str)->PayloadObject:
        """
        :param key: The key to use for decoding
        :param latent_rep: The latent representation we will be using. Generic
        :param payload_type: A string. Put in payload slot
        :return: A payload object
        """
        raise NotImplementedError("A subclass needs to implement forward.")

# Data converter framework

class ConverterPayloadRegistry(nn.Module, Generic[LatentRep]):
    """
    The actual data converter, this is responsible
    for performing static data conversion and satisfying dependencies.
    """
    def __init__(self):
        super().__init__()
        self.encoders: nn.ModuleDict() = nn.ModuleDict()
        self.decoders: nn.ModuleDict() = nn.ModuleDict()

    def register(self,
                 type: str,
                 encoder: AbstractPayloadEncoder,
                 decoder: AbstractPayloadDecoder):
        """
        Registers a new converter pair.

        :param type: A string, the type of the conversion
        :param encoder: The encoder to register
        :param decoder: The decode to register
        """
        assert encoder.embedding_dim == decoder.embedding_dim, "Embedding dims should be similar"
        self.encoders[type] = encoder
        self.decoders[type] = decoder

    def encode(self, payload: PayloadObject)->PayloadEmbedding:
        """
        Finds the right encoder, and encodes the payload.
        :param payload: The payload to encode
        :return:
        """
        return self.encoders[payload.type](payload.payload)

    def decode(self, key: torch.Tensor, payload_type: str, encoding: LatentRep)->PayloadObject:
        """
        Decodes based on a key and an encoding back to the original payload object.

        :param key: A key uniquely associated with the payload object
        :param payload_type: the payload type to process
        :param encoding: The encoding to process
        :return: An original payload object.
        """
        return self.decoders[payload_type](key, encoding, payload_type)

    def select_decoders(self,
                        key: torch.Tensor,
                        temperature: float = 1.0)->Tuple[np.ndarray, torch.Tensor]:
        """
        A batch method, this will use a key collection to return a list representing
        which decoder it is thought would be optimal to choose.

        :param key: A key to generate predicates from. (..., embedding) in shape
        :return: The selected decoders. A NUMPY string tensor of shape (...)
        :return: The decoder logits. (batch, num_decoders) in shape.
        """

        # Create the logits
        predicates = [decoder.predicate_logits for decoder in self.decoders.values()]
        logits = torch.stack([predicate(key) for predicate in predicates], dim=-1)

        # Perform sampling.
        if temperature == 0:
            # Handle the static case
            outcome = torch.argmax(logits, dim=-1)
        else:
            # Handle the temperature case
            scaled_logits = logits / temperature
            probabilities = torch.softmax(scaled_logits, dim=-1)
            outcome = torch.multinomial(probabilities, num_samples=1, replacement=True)

        # Encode grid
        map_vocab = dict(zip(range(len(self.decoders)), self.decoders.keys()))
        vectorized_map = np.vectorize(lambda x : map_vocab[x])
        outcome = vectorized_map(outcome.numpy())
        return outcome, logits

registry = ConverterPayloadRegistry()

## The whole converter shebang.

class LatentConverter:
    """
    Responsible for converting an entire data block into
    latent representation format, the DataConverter will
    convert the input.
    """

    def __init__(self,
                 secondary_encoder: nn.Module,
                 registry: ConverterPayloadRegistry | None,
                 ):

