import os
import textwrap
from typing import Dict, Union, Tuple

import torch
from torch import nn
from transformers import (PreTrainedModel, AutoTokenizer, PreTrainedTokenizer,
                          AutoModelForCausalLM, PreTrainedTokenizerFast)
from tokenizers import processors
from enum import Enum

class SpecialTokens(Enum):
    bos_token = "<BOS>"
    eos_token = "<EOS>"
    pad_token = "<PAD>"

class AdditionalSpecialTokens(Enum):
    prompt_token = "<PROMPT>" # Prompt tokens are tokens the model cannot generate itself
    beginning_of_response_token = "<RESPONSE>" # These the model are now responsible for generating
class Vocabulary(nn.Module):
    """
    A centralized place in which vocabulary,
    embeddings, and logits can be kept and
    otherwise managed.
    """
    def save_pretrained_vocabulary(self, directory: Union[str, os.PathLike]):
        """
        Saves the vocabulary in its current configuration
        :param directory: The directory to save in.
        """
        self.tokenizer.save_pretrained(directory)
        torch.save(self.embeddings, os.path.join(directory, "embeddings.pt"))
        torch.save(self.logit_projector, os.path.join(directory, "logit_projector.pt"))

    @classmethod
    def load_pretrained_vocabulary(cls, directory: Union[str, os.PathLike]) -> 'Vocabulary':
        """
        Loads a pretrained vocabulary from the indicated save directory
        :param directory: The directory to save in
        :return: The pretrained vocabulary struct
        """
        tokenizer = AutoTokenizer.from_pretrained(directory)
        embeddings = torch.load(os.path.join(directory, "embeddings.pt"))
        logit_projector = torch.load(os.path.join(directory, "logit_projector.pt"))
        return cls(embeddings, tokenizer, logit_projector)

    def __reduce__(self):
        msg = """
        Attempt was made to invoke torch.save or pickle directly. This 
        is not supported. Instead, use built in save methods on the classes.
        """
        msg = textwrap.dedent(msg)
        raise NotImplementedError(msg)

    def __init__(self,
                 embeddings: nn.Embedding,
                 tokenizer: PreTrainedTokenizer,
                 logit: nn.Linear,
                 ):
        super().__init__()

        assert embeddings.weight.shape[0] == logit.weight.shape[0], "Embeddings and logits did not have same model dim"

        self.d_model = embeddings.weight.shape[-1]
        self.embeddings = embeddings
        self.tokenizer = tokenizer
        self.logit_projector = logit

TokenizerAlias = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
def _customize_tokenizer(tokenizer: TokenizerAlias)->Tuple[TokenizerAlias, int]:
    """
    Customizes the provided pretrained tokenizer to include, possibly, extra special
    tokens for various situations. This will include injecting additional special
    tokens for the prompt-response logic, eos, bos, and pad conditions. It also will
    include inserting

    :param tokenizer: The tokenizer to customize
    :return:
    - The tokenizer
    - The number of new tokens
    """
    # Generate the update dictionary, and integrate it
    # into the tokenizer
    original_size = tokenizer.vocab_size

    # Setup special token update. Integrate them
    special_tokens = {case.name: case.value for case in SpecialTokens}
    special_tokens["additional_special_tokens"] = [case.value for case in AdditionalSpecialTokens]
    num_added_tokens = tokenizer.add_special_tokens(special_tokens, replace_additional_special_tokens=False)

    special_token_ids = special_tokens.copy()
    additional_special_tokens = special_token_ids.pop("additional_special_tokens")
    special_token_ids = [(value, tokenizer.convert_tokens_to_ids(value)) for value in special_token_ids.values()]
    special_token_ids += [(value, tokenizer.convert_tokens_to_ids(value)) for value in additional_special_tokens]

    # Define the postprocessor.
    #
    # The post processor is configured to either
    # inject SOS and EOS in single configuration,
    # or that PLUS prompt, response singles for
    # pair configuration.

    single_directive = [SpecialTokens.bos_token.value, "$A", SpecialTokens.eos_token.value]
    pair_directive = [SpecialTokens.bos_token.value,
                      AdditionalSpecialTokens.prompt_token.value,
                      "$A",
                      AdditionalSpecialTokens.beginning_of_response_token.value,
                      "$B",
                       SpecialTokens.eos_token.value
                       ]

    single_directive = " ".join(single_directive)
    pair_directive = " ".join(pair_directive)

    postprocessor = processors.TemplateProcessing(
        single =single_directive,
        pair=pair_directive,
        special_tokens=special_token_ids
    )
    if isinstance(tokenizer, PreTrainedTokenizerFast):
        tokenizer._tokenizer.post_processor = postprocessor
    elif isinstance(tokenizer, PreTrainedTokenizer):
        tokenizer.post_processor = postprocessor
    else:
        raise TypeError()

    tokenizer.true_vocab_size = original_size + num_added_tokens
    return tokenizer, num_added_tokens
def _load_tokenizer_from_huggingface(name: str, kwargs) -> Tuple[TokenizerAlias, int]:
    """
    Loads, and customizes as needed, the tokenizer vocabulary to be compatible
    with the model architecture. This consists of

    - Updating any missing special tokens to be present within the vocabulary
    - Updating the post processor to handle said tokens as needed


    :param name: The name of the model to load the head off of.
    :return:
    - The number of additional tokens that had to be injected
    - The tokenizer. Ready to tokenize. Note that some addition
    """
    # Ready the tokenizer, by inserting custom
    # vocabulary
    tokenizer = AutoTokenizer.from_pretrained(name, **kwargs)
    tokenizer, num_new_vocab_elements = _customize_tokenizer(tokenizer)
    return tokenizer, num_new_vocab_elements

def _load_embeddings_for_tokenizer(model: nn.Module, num_added_tokens: int)->nn.Embedding:
    """
    Loads embeddings off of a module, and modifies them to be
    compatible with the tokenizer.
    :param model: Huggingface causal model to load off
    :param num_tokens_to_expand: The number of tokens to expand by
    :return: The setup embeddings.
    """
    with torch.no_grad():
        embeddings = model.get_input_embeddings()
        assert embeddings is not None

        old_num_embeddings, embedding_dim = embeddings.weight.size()
        new_embeddings = nn.Embedding(old_num_embeddings + num_added_tokens, embedding_dim)
        new_embeddings.weight[:old_num_embeddings] = embeddings.weight
    return new_embeddings

def _load_logits_for_tokenizer(model: nn.Module, num_added_tokens: int)->nn.Linear:
    """
    Loads the logits for the model, and expands them to be compatible
    based on the given number of additional tokens

    :param model: The huggingface model to load from
    :param num_added_tokens: The additional tokens
    :return: The logit layer
    """
    # Expand the logits to be able to predict the extra dimensions.
    with torch.no_grad():
        logits = model.get_output_embeddings()
        assert logits is not None

        num_logits, d_embedding = logits.weight.size()
        if logits.bias is None:
            new_logits = nn.Linear(d_embedding, num_logits + num_added_tokens, bias=False)
            new_logits.weight[:num_logits] = logits.weight
        else:
            new_logits = nn.Linear(d_embedding, num_logits)
            new_logits.weight[:num_logits] = logits.weight
            new_logits.bias[:num_logits] = logits.bias
    return new_logits
def load_vocabulary_off_huggingface_model(name: str, **kwargs) -> 'Vocabulary':
    """
    Loads, then customizes, a causal lm vocabulary for (hopefully)
    compatibility with the model.

    ---- requirements ----

    To construct a vocabulary struct, we need to get three things.
    These are.

    - A functional tokenizer with the right customizations.
    - An embedding that can accept the provided tokens
    - A logit projector that can predict the tokens

    --- General logic---

    Basically, we fetch the

    In particular, we fetch three
    things. These are

    - A tokenizer.

    :param name: The model to load from. We are going to
    :return: The vocabulary, configured for use.
    """

    tokenizer, num_new_vocab_elements = _load_tokenizer_from_huggingface(name, kwargs)
    donor_model = AutoModelForCausalLM.from_pretrained(name)
    embeddings = _load_embeddings_for_tokenizer(donor_model, num_new_vocab_elements)
    logits = _load_logits_for_tokenizer(donor_model, num_new_vocab_elements)
    return Vocabulary(embeddings, tokenizer, logits)