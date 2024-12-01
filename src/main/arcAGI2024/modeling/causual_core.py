"""
The causal model cores. These are constructed by
config entries, and savable/loadable
"""

from ..base import SavableConfig
from dataclasses import dataclass
class ModelCoreConfig(SavableConfig):
    """

    """
    # Tokenizer
    # Logits
    # Embedding
    # Model spec.