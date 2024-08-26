"""
distribution adapters are tied to a particular distribution, and are capable
of sampling from that distributions. Due to project details, this includes
differentiable approximations of the distributions.
"""
import warnings
import textwrap
import torch
import inspect
from torch import nn
from torch.nn import functional as F
from typing import Dict, List, Any, Type, Tuple, Callable
from abc import ABC, abstractmethod
from .io_adapters import IORegistry
from .io_adapters import registry as io_adapters_registry
class DistributionAdapterRegistry:
    """
    The distribution registry mechanism. DistributionAdapters are
    capable of sampling or computing loss against particular distributions,
    and this registry keeps track of them.

    Notably, distributionAdapters have to be registered against one or more
    io adapters so we know what distributions can be handled by what adapters.

    Note that when subclassing an additional distribution adapter that is registered,
    it will be assumed that the config spec and such are similar for the different
    cases.
    """
    @property
    def names(self)->List[str]:
        return list(self.model_registry.keys())

    def __init__(self, io_adapter_registry: IORegistry):
        self.io_registry = io_adapter_registry
        self.association_registry: Dict[str, List[str]] = {}
        self.model_registry: Dict[str, "DistributionAdapter"] = {}
        self.setup_registry: Dict[str, Dict[str, Type[Any]]] = {}

    def validate_io_name_in_registry(self, name: str):
        if name not in self.io_registry.names:
            msg = f"""
            Attempted to register a sampler to nonexistant io adapter 
            named {name}
            """
            msg = textwrap.dedent(msg)
            raise ValueError(msg)

    def validate_sampler_in_registry(self, name: str):
        if name not in self.model_registry:
            msg = f"""
            Attempt was made to access a sampler named '{name}', however this
            sampler was never registered!
            """
            msg = textwrap.dedent(msg)
            raise ValueError(msg)

    def validate_config_compatible(self,
                                   config: Dict[str, Any],
                                   config_spec: Dict[str, Type[Any]]
                                   ):
        if len(config) != len(config_spec):
            msg = f"""
            Config passed to create sampler was not compatible. 
            
            The sample expected a config dictionary of length {len(config_spec)},
            however it got a config dictionary of length {len(config)}.
            """
            msg = textwrap.dedent(msg)
            raise ValueError(msg)
        if set(config.keys()) != set(config_spec.keys()):
            msg = f"""
            Config passed to create sampler was not compatible.
            
            The config did not contain the same keys as the config spec
            
            config: {config.keys()}
            config_spec: {config.keys()}
            """
            msg = textwrap.dedent(msg)
            raise ValueError(msg)
        for key in config_spec.keys():
            if not isinstance(config[key], config_spec[key]):
                msg = f"""
                Config spec specified the key '{key}' be of 
                type {config_spec[key]}. However, we got 
                {config[key]}
                """
                msg = textwrap.dedent(msg)
                raise ValueError(msg)

    def register(self,
                 name: str,
                 association: str,
                 config_spec: Dict[str, Type[Any]],
                 sampler: "DistributionAdapter"
                 ):
        """
        Performs the initial registration of a sampler adapter.

        This includes associating it with at least one io adapter,
        which must exist, and a config spec plus a sampler.

        :param name: What to call the sampler adapter
        :param association: The io adapter to associate with, at least to begin with.
        :param config_spec: The config specification to associate setup with
        :param sampler: The sampler to register
        """

        if name in self.model_registry:
            msg = f"""
            Warning: Overwriting existing sampling registry named {name}. If you
            intend to add another io adapter association, use .associate instead.
            """
            msg = textwrap.dedent(msg)
            warnings.warn(msg)

        self.validate_io_name_in_registry(association)
        self.model_registry[name] = sampler
        self.setup_registry[name] = config_spec
        self.association_registry[name] = [association]

    def register_association(self, name: str, io_name: str):
        """
        Registers another IO association with a particular
        sampler. This would let the sampler sample from
        distributions produced by that IO Adapter.

        :param name: The name of the sampler to modify
        :param io_name: The new association to encode.
        """
        self.validate_io_name_in_registry(io_name)
        self.validate_sampler_in_registry(name)
        if io_name not in self.association_registry[name]:
            self.association_registry[name].append(io_name)

    def get_associations(self, name: str)->List[str]:
        """
        Gets the allowed associations for a particular sampler.

        :param name: The name of the sampler to inquire about
        :return: A list of the associated io adapters
        """
        self.validate_sampler_in_registry(name)
        return self.association_registry[name]

    def get_structure(self, name: str)->Dict[str, Type[Any]]:
        """
        Gets a dictionary which specifies the name and types
        we expect to be passed in setup to setup the specified
        type of sampler.

        :param name: The name of the sampler to get structure about
        :return: A config spec dictionary, mapping keywords to type
        """
        self.validate_sampler_in_registry(name)
        return self.setup_registry[name]

    def get_documentation(self, name: str)->Tuple[str, str]:
        """
        Gets documentation off the sampler regarding first the
        sampler's class docstring and second the setup method
        docstring. This can give insight into how to intepret
        the structure.
        :param name: The name of the sampler to get documentation from
        :return: The class docstring
        :return: The docstring on "setup"
        """
        self.validate_sampler_in_registry(name)
        subclass = self.model_registry[name]
        class_docstring = inspect.getdoc(subclass)
        setup_docstring = inspect.getdoc(subclass.setup)
        return class_docstring, setup_docstring

    def registry_decorator(self,
                           name: str,
                           association: str,
                           config_spec: Dict[str, Type[Any]]
                           )->Callable[[Type["DistributionAdapter"]], Type["DistributionAdapter"]]:
        """
        A decorator for making it easier to register items. Same format as
        native register.

        :param name: The name to call the sampler by
        :param association: The initial io association
        :param config_spec: The specification regarding what setup needs.
        :return: A callable that will decorate a class correctly.
        """

        def decorator(sampler: Type[DistributionAdapter])->DistributionAdapter:
            self.register(name, association, config_spec, sampler)
            return sampler
        return decorator

    def setup(self, name: str, config: Dict[str, Any])-> "DistributionAdapter":
        """
        Sets up a sampler adapter using the provided configuration, and
        returns the setup adapter.

        :param name: The name of the adapter
        :param config: The config to generate it with
        :return: A subclass of the specified instance
        """
        self.validate_sampler_in_registry(name)
        self.validate_config_compatible(config, self.setup_registry[name])
        return self.model_registry[name].setup(**config)

class DistributionAdapter(ABC, nn.Module):
    """
    An abstract distribution adapter class.

    This class needs to implement support for sampling
    and loss with a particular distribution representation.
    """

    @classmethod
    @abstractmethod
    def setup(cls, **config: Dict[str, Any]) -> 'DistributionAdapter':
        """
        Creates and configures a new DistributionAdapter instance based on the given config.

        :param config: A dictionary containing configuration parameters.
        :return: An instance of the subclassed IOAdapter, configured for the specified mode.
        """
        pass

    @abstractmethod
    def gradient_sample(self,
                              distribution: torch.Tensor | Tuple[torch.Tensor, ...],
                              *controls: Any
                              )->Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a hard, but differentiable, sample from the distribution. What
        this means will depend in large part on the exact distribution involved,
        but we will generally get out a differentiable selection that can be used
        to establish a loss. In fact, the sample must be directly compatible
        with the loss method.

        :param distribution: The distribution to sample from
        :param controls: Any controls
        :return: The sampled distribution in whatever format makes the most sense
        :return: The actual samples, usable for review purposes or whatever other mechanisms are required.
        """
        pass
    @abstractmethod
    def sample(self,
               distribution: torch.Tensor | Tuple[torch.Tensor, ...],
               *controls: Any
               )->torch.Tensor:
        """
        Performs sampling from the distribution. Should
        :param distribution:
            The tensor distribution to sample from. Was produced by an IO adapter.
            Has a common shape of (batch, ...
        :param controls:
            Any additional parameters we might want to pass, such as temperature or beam search
            width.

        :return:
            The sampled tensors. This should generally be as reduced as possible - for instance,
            return indices rather than a one-hot probability vector. This need not be differentiable
        """
        pass

    def loss(self,
             distribution: torch.Tensor | Tuple[torch.Tensor, ...],
             targets: torch.Tensor | Tuple[torch.Tensor, ...],
             *controls,
             ):
        """
        Provides a loss mechanism by which to train against by considering the
        generated distribution and the targets. The controls feature allows
        additional information to be passed in, such as label smoothing targets.

        :param distribution: The distribution to take a loss with
        :param targets: The targets to use for the loss
        :param controls: Any additional parameters we need
        :return: A loss scalar
        """
        pass

class VocabularyDistributionAdapter(DistributionAdapter):
    """
    A distribution adapter designed to be interacting
    with a vocabulary distribution. In this format,
    each entry of a vocabulary is represented by an integer,
    and we expect to see logits matching the vocabulary.
    """

    @classmethod
    def setup(cls, hard: bool)->"VocabularyDistributionAdapter":
        """

        :param hard: Whether or not to use a hard gumbel-softmax during training
        :return: A new vocbaulary distribution adapter instance.
        """
        return cls(hard)
    def __init__(self, hard: bool):
        super().__init__()
        self.hard = hard

    def gradient_sample(self,
                        distribution: torch.Tensor,
                        temperature: float | torch.Tensor
                        )->torch:
        """
        :param logits: Unnormalized log probabilities (e.g., output of a linear layer).
        :param temperature: Controls the smoothness of the distribution.
        :param hard: If True, returns a one-hot encoded vector.
        :return: Tensor of the same shape as logits, representing the sampled probabilities.
        """
        # Sample from Gumbel(0, 1)
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(distribution)))
        # Add Gumbel noise to the logits and apply softmax
        y = F.softmax((distribution + gumbel_noise) / temperature, dim=-1)

        if self.hard:
            # Get the one-hot encoded version by taking the argmax
            y_hard = torch.zeros_like(y).scatter_(-1, y.argmax(dim=-1, keepdim=True), 1.0)
            # In the backward pass, retain the gradients from the soft sample
            y = (y_hard - y).detach() + y

        return y
    def sample(self,
               distribution: torch.Tensor,
               temperature: float | torch.Tensor) -> torch.Tensor:
        """
        :param distribution:
            The logit distribution we want to sample from.
            We will assume the last dimension is associated with the probabilities.
            Common shape of around (batch, ..., logits)
        :param temperature: The generation temperature
        :return: A int tensor indicating the sampled vocabulary elements
            Shape is (batch, ...)
        """
        # Apply softmax with temperature scaling to get the probability distribution
        probs = F.softmax(distribution / temperature, dim=-1)

        # Sample from the categorical distribution created by probs
        sampled_indices = torch.multinomial(probs, num_samples=1)
        return sampled_indices.squeeze(-1)

