"""
Some forms of content - particularly image-based content,
- may end up being "patched" in which multiple pixels
end up lying within the same encoding. This sector
contains mechanisms for this behavior.

The point behind the actual classes is to put, in one
location, all the things that will be needed to make
patching training work, such as downsampling, upsampling,
and block shape conversion.
"""

import warnings
import torch
import inspect
import helper_utils
from abc import ABC, abstractmethod
from torch import nn
from typing import List, Dict, Callable, Any, Type, Tuple

class PatchingAdapterRegistry:
    """
    The PatchingAdapter registry keeps track of the various PatchingAdapter that exist,
    and their setup requirements. Each PatchingAdapter has to be manually
    registered after the class is created.

    Registration should include information on what keywords need
    to be passed in on setup, and what types they map to.
    """

    @property
    def names(self) -> List[str]:
        return list(self.setup_registry.keys())

    def validate_patching_adapter_registered(self, name: str):
        """
        Validates whether a particular PatchingAdapter has
        been registered. Raises an error if not available.
        :param name: The name of the adapter to look for
        :raises: KeyError, if the adapter has not been registered.
        """
        if name not in self.model_registry:
            raise KeyError(f"PatchingAdapter of name '{name}' has not been registered and cannot be manipulated.")

    def __init__(self):
        self.model_registry: Dict[str, PatchingAdapter] = {}
        self.setup_registry: Dict[str, Type[Any]] = {}

    def register(self,
                 name: str,
                 config_spec: Dict[str, Type],
                 adapter: Type["PatchingAdapter"],
                 ):
        """
        Registers a PatchingAdapter to be associated with a particular
        mode of operation.

        :param name: The name of the registry.
        :param config_spec: A dictionary that maps keywords
               to types. Used to validate config dictionaries, and even
               tell us what those dictionaries should contain.
        """
        assert isinstance(config_spec, dict), f"Item is not a config dict, '{config_spec}'"
        assert issubclass(adapter, PatchingAdapter), f"Item is not a PatchingAdapter '{adapter}'"

        if name in self.model_registry:
            warnings.warn(f"Warning! Overwriting PatchingAdapter of name '{name}'")

        self.model_registry[name] = adapter
        self.setup_registry[name] = config_spec

    def register_decorator(self,
                           name: str,
                           config_spec: Dict[str, Type]
                           ) -> Callable[[Type["PatchingAdapter"]], None]:
        """
        Creates a decorator for registering a PatchingAdapter. You
        first specify the setup specification in terms of the
        keywords and the type.

        Then you get a callback to use

        :param config_spec: A mapping of keywords to their required types
        :return: A decorator capable of being called with and registering a
                 PatchingAdapter.
        """
        def decorator(adapter: Type[PatchingAdapter]):
            self.register(name, config_spec, adapter)
            return adapter
        return decorator

    def setup(self,
              name: str,
              config: Dict[str, Type]
              ) -> "PatchingAdapter":
        """
        Set up a new instance of a particular type of adapter
        based on the provided config.

        :param name: The given name of the adapter
        :param config: The config for the name
        :return: A PatchingAdapter instance that has been setup
        """

        self.validate_patching_adapter_registered(name)
        expected_config = self.setup_registry[name]
        helper_utils.validate_config(config, expected_config)
        return self.model_registry[name].setup(**config)

    def get_structure(self, name) -> Dict[str, Type]:
        """
        Gets the structure associated with a particular
        PatchingAdapter. Smart use of this will allow the construction
        of an automated wizard.

        :param name: The name of the given adapter
        :return: The config spec dictionary.
        """
        self.validate_patching_adapter_registered(name)
        return self.setup_registry[name]

    def get_documentation(self, name: str) -> Tuple[str, str]:
        """
        Gets the documentation that was declared on the PatchingAdapter
        with a given name. This consists of the class documentation,
        and the setup documentation.

        :param name: The name of the PatchingAdapter to get documentation about
        :return: The class docstring
        :return: The setup docstring
        """
        self.validate_patching_adapter_registered(name)
        subclass = self.model_registry[name]
        class_docstring = inspect.getdoc(subclass)
        setup_docstring = inspect.getdoc(subclass.setup)
        return class_docstring, setup_docstring

class PatchingAdapter(ABC, nn.Module):
    """
    A patching adapter helps to support patching - something which
    commonly occurs with some image models between embedding and
    training/generation.

    Patching associates a collection of pixels into a single bigger
    collection, which is usually then combined together during embedding
    or afterwords. The result is that multiple pixels can end up being associated
    with one value.

    Patching adapters are required whether dealing with patching or not,
    and in the case of a lack of use will just do nothing.
    """
    @abstractmethod
    def patch(self,
              shapes: torch.Tensor,
              embeddings: torch.Tensor,
              )->Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        The main patching mechanism, used primarily during training.
        It processes embeddings into reduced-dimensionality patches.
        It also updates the shapes tensor based on how we are patching.

        :param shapes:
            * The shapes tensor specifies the shapes of any active regions within the embedding tensor per batch
            * It will always have a shape like (batch, D) where D are the number of dimensions.
        :param embeddings:
            * The embeddings to be patched. Will have some sort of
              variation on the shape (batch, ...., embeddings)
            * The ellipses component, ..., should have as many dimensions as is D
        :return shapes:
            * The shapes of the patches active region. This should be
              reduced in size in association with the patches. Note that
              what is returned here is what the model will learn to predict
              in terms of block size.
        :return patched_embeddings:
            * The patched embeddings. The dimensionality can be reduced here among the ...
              dimensions, and embedding may change size.
            * Shape: (batch, ...othershapes, other_embedding_dim)
        """
        pass

    @abstractmethod
    def unpatch(self,
                embeddings: torch.Tensor,
                ):
        """
        The unpatching function. This function should take in
        a collection of currently patched embeddings that are
        usually generated by a model, and turn them into their
        unpatched format.

        :param embeddings:
            * The embeddings to be unpatched.
            * Will have some sort of shape like (batch, ..., embeddings)
        """

