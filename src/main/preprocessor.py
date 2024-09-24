"""
An implementation of the preprocessors and other needed
mechanisms required to allow the main to work. Preprocessing operates
in essentially two phases in this main.

1) One of them is tokenization/gridint conversion,
   in which raw blocks of data content are converted into flattened tensors
   with significant meaning. These blocks can then be concatenated together
2) Then, there is the internal embedding processes that are used to convert content
   into vectors the main can understand.

"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any
from transformers.models.perceiver.modeling_perceiver import PreprocessorType, PreprocessorOutputType

class AbstractModePreprocessor(ABC):
    """
    An abstract, mode-associated preprocessor
    class which is defined sufficiently to ensure
    it can be associated with a particular mode
    of content.
    """
class TextPreprocessor:
    """

    """

class