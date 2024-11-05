import os
import textwrap
from typing import Dict, Union

import torch
from torch import nn
from transformers import PreTrainedModel, AutoTokenizer, PreTrainedTokenizer, AutoModelForCausalLM
from enum import Enum


class SpecialTokens(Enum):
    bos_token = "<BOS>"
    eos_token = "<EOS>"
    pad_token = "<PAD>"

class AdditionalSpecialTokens(Enum):
    read_token = "<READ>"
    response_token = "<RESPONSE>"
    start_grid_data = "<STARTGRID>"
    end_grid_data = "<ENDGRID>"
    grid_line = "<GRIDLINE>"

class VocabularyStruct(nn.Module):
    """
    A centralized place in which vocabulary,
    embeddings, and logits can be kept and
    otherwise managed.
    """
    @classmethod
    def auto_load_from_pretrained(cls, name: str) -> 'VocabularyStruct':
        """
        Loads the vocabulary and corrosponding tokenizer
        directly from huggingface.
        :param name: The name to load from
        :return: The setup vocabulary struct
        """
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name)
        return cls.load_from_pretrained(tokenizer, model)

    @classmethod
    def load_from_pretrained(cls,
                             tokenizer: PreTrainedTokenizer,
                             model: PreTrainedModel
                             ) -> 'VocabularyStruct':
        """
        Loads a vocabulary struct from a pretrained language model.
        This allows us to use a premade embedding and manipulation
        system, saving some time.

        Be aware that this will fetch the logit endpoint too. Make
        sure it is correctly configured!
        :param tokenizer: The tokenizer associated with the model
        :param model: The model to fetch off of. Should have
            - Input embedding system
            - Output logit system
        :return: The setup vocabulary struct
        """
        embeddings = model.get_input_embeddings()
        logits = model.get_output_embeddings()
        assert logits is not None, "attempted to load model without predictive capacities"
        assert embeddings is not None, "attempted to load model without embeddings"
        return cls(embeddings, tokenizer, logits)

    def customize_vocabulary(self):
        """
        Customizes the provided collection of pretrained tokenizer,
        embeddings, and logits in order to support the declared
        required special tokens
        """
        # Generate the update dictionary, and integrate it
        # into the tokenizer
        original_size = self.tokenizer.vocab_size
        special_tokens = {case.name : case.value for case in SpecialTokens}
        special_tokens["additional_special_tokens"] = [case.value for case in AdditionalSpecialTokens]
        num_added_tokens = self.tokenizer.add_special_tokens(special_tokens, replace_additional_special_tokens=False)

        # Modify the tokenizer defaults slightly
        self.tokenizer.padding_side = "Right"
        self.tokenizer.true_vocab_size = original_size + num_added_tokens

        # Expand the embeddings. We take the current embeddings, extend it
        # a bit, and initialize the extension.
        with torch.no_grad():
            old_num_embeddings, embedding_dim = self.embeddings.weight.size()
            new_embeddings = nn.Embedding(old_num_embeddings + num_added_tokens, embedding_dim)
            new_embeddings.weight[:old_num_embeddings] = self.embeddings.weight

        self.embeddings = new_embeddings

        # Expand the logits to be able to predict the extra dimensions.
        with torch.no_grad():
            num_logits, d_embedding = self.logit_projector.weight.size()
            if self.logit_projector.bias is None:
                new_logits = nn.Linear(d_embedding, num_logits + num_added_tokens, bias=False)
                new_logits.weight[:num_logits] = self.logit_projector.weight
            else:
                new_logits = nn.Linear(d_embedding, num_logits)
                new_logits.weight[:num_logits] = self.logit_projector.weight
                new_logits.bias[:num_logits] = self.logit_projector.bias
        self.logit_projector = new_logits
    def save_pretrained_vocabulary(self, directory: Union[str, os.PathLike]):
        """
        Saves the vocabulary in its current configuration
        :param directory: The directory to save in.
        """
        self.tokenizer.save_pretrained(directory)
        torch.save(self.embeddings, os.path.join(directory, "embeddings.pt"))
        torch.save(self.logit_projector, os.path.join(directory, "logit_projector.pt"))

    @classmethod
    def load_pretrained_vocabulary(cls, directory: Union[str, os.PathLike])->'VocabularyStruct':
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
        self.customize_vocabulary()
