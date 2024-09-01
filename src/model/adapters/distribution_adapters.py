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
from .io_adapters import registry as io_registry
from .helper_utils import validate_config

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
    def names(self) -> List[str]:
        return list(self.model_registry.keys())

    def __init__(self, io_adapter_registry: IORegistry):
        self.io_registry = io_adapter_registry
        self.association_registry: Dict[str, List[str]] = {}
        self.model_registry: Dict[str, "DistributionAdapter"] = {}
        self.setup_registry: Dict[str, Dict[str, Type[Any]]] = {}

    def validate_io_name_in_registry(self, name: str):
        if name not in self.io_registry.names:
            msg = f"""
            Attempted to register a distribution adapter to non-existent io adapter 
            named {name}
            """
            msg = textwrap.dedent(msg)
            raise ValueError(msg)

    def validate_distribution_adapter_in_registry(self, name: str):
        if name not in self.model_registry:
            msg = f"""
            Attempt was made to access a distribution adapter named '{name}', however this
            distribution adapter was never registered!
            """
            msg = textwrap.dedent(msg)
            raise ValueError(msg)

    def register(self,
                 name: str,
                 association: str,
                 config_spec: Dict[str, Type[Any]],
                 distribution_adapter: "DistributionAdapter"
                 ):
        """
        Performs the initial registration of a distribution adapter.

        This includes associating it with at least one io adapter,
        which must exist, and a config spec plus a distribution adapter.

        :param name: What to call the distribution adapter
        :param association: The io adapter to associate with, at least to begin with.
        :param config_spec: The config specification to associate setup with
        :param distribution_adapter: The distribution adapter to register
        """

        assert issubclass(distribution_adapter, DistributionAdapter)

        if name in self.model_registry:
            msg = f"""
            Warning: Overwriting existing distribution adapter registry named {name}. If you
            intend to add another io adapter association, use .associate instead.
            """
            msg = textwrap.dedent(msg)
            warnings.warn(msg)

        self.validate_io_name_in_registry(association)
        self.model_registry[name] = distribution_adapter
        self.setup_registry[name] = config_spec
        self.association_registry[name] = [association]

    def register_association(self, name: str, io_name: str):
        """
        Registers another IO association with a particular
        distribution adapter. This would let the distribution sample from
        distributions produced by that IO Adapter.

        :param name: The name of the distribution adapter to modify
        :param io_name: The new association to encode.
        """
        self.validate_io_name_in_registry(io_name)
        self.validate_distribution_adapter_in_registry(name)
        if io_name not in self.association_registry[name]:
            self.association_registry[name].append(io_name)

    def get_associations(self, name: str) -> List[str]:
        """
        Gets the allowed associations for a particular distribution adapter.

        :param name: The name of the distribution adapter to inquire about
        :return: A list of the associated io adapters
        """
        self.validate_distribution_adapter_in_registry(name)
        return self.association_registry[name]

    def get_config_spec(self, name: str) -> Dict[str, Type[Any]]:
        """
        Gets a dictionary which specifies the name and types
        we expect to be passed in setup to setup the specified
        type of distribution adapter.

        :param name: The name of the distribution adapter to get structure about
        :return: A config spec dictionary, mapping keywords to type
        """
        self.validate_distribution_adapter_in_registry(name)
        return self.setup_registry[name]

    def get_documentation(self, name: str) -> Tuple[str, str]:
        """
        Gets documentation off the distribution regarding first the
        distribution adapter's class docstring and second the setup method
        docstring. This can give insight into how to interpret
        the structure.
        :param name: The name of the distribution adapter to get documentation from
        :return: The class docstring
        :return: The docstring on "setup"
        """
        self.validate_distribution_adapter_in_registry(name)
        subclass = self.model_registry[name]
        class_docstring = inspect.getdoc(subclass)
        setup_docstring = inspect.getdoc(subclass.setup)
        return class_docstring, setup_docstring

    def registry_decorator(self,
                           name: str,
                           association: str,
                           config_spec: Dict[str, Type[Any]]
                           ) -> Callable[[Type["DistributionAdapter"]], Type["DistributionAdapter"]]:
        """
        A decorator for making it easier to register items. Same format as
        native register.

        :param name: The name to call the distribution adapter by
        :param association: The initial io association
        :param config_spec: The specification regarding what setup needs.
        :return: A callable that will decorate a class correctly.
        """

        def decorator(distribution_adapter: Type[DistributionAdapter]) -> DistributionAdapter:
            self.register(name, association, config_spec, distribution_adapter)
            return distribution_adapter
        return decorator

    def setup(self, name: str, config: Dict[str, Any]) -> "DistributionAdapter":
        """
        Sets up a distribution adapter using the provided configuration, and
        returns the setup adapter.

        :param name: The name of the adapter
        :param config: The config to generate it with
        :return: A subclass of the specified instance
        """
        self.validate_distribution_adapter_in_registry(name)
        validate_config(config, self.setup_registry[name])
        return self.model_registry[name].setup(**config)
registry = DistributionAdapterRegistry(io_registry)

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
    def sample(self,
               distribution: torch.Tensor | Tuple[torch.Tensor, ...],
               mask: torch.Tensor,
               *controls: Any
               )->torch.Tensor:
        """
        Performs sampling from the distribution. Should
        :param distribution:
            The tensor distribution to sample from. Was produced by an IO adapter.
            Has a common shape of (batch, ...
        :param mask:
            A mask against which to sample from.
            The ones we want to sample should be "true"
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
             mask: torch.Tensor,
             *controls,
             ):
        """
        Provides a loss mechanism by which to train against by considering the
        generated distribution and the targets. The controls feature allows
        additional information to be passed in, such as label smoothing targets.

        :param distribution: The distribution to take a loss with
        :param targets: The targets to use for the loss
        :param mask: A mask for forming the loss.
        :param controls: Any additional parameters we need
        :return: A loss scalar
        """
        pass

setup_spec = {"label_smoothing_rates" : List[float]}
@registry.registry_decorator("vocab_distribution",
                             "vocabulary_adapter",
                             setup_spec
                             )
class VocabularyDistributionAdapter(DistributionAdapter):
    """
    A distribution adapter designed to be interacting
    with a vocabulary distribution. In this format, we
    represent each element of the vocabulary with a logit,
    which can be associated with a probability.

    Sampling can be performed based on this probability distribution.
    Additionally, loss can be performed as well. Loss can be assigned
    to be performed with one of several label smoothing values, and
    the value to use can be defined per-target.

    This is important for controlling exploration.
    """

    @classmethod
    def setup(cls,
              label_smoothing_rates: List[float],
              )->"VocabularyDistributionAdapter":
        """
        Sets up a vocabulary distribution adapter, ready to be used
        for sampling and loss. The primary thing that needs to be
        defined is the label smoothing rates for loss.

        :param label_smoothing_rates:
            A list which specifies an association between an integer value
            and a label smoothing rate. Each target must be assigned to one of these
            categories
        """

        assert isinstance(label_smoothing_rates, list)
        assert all([isinstance(rate, float) for rate in label_smoothing_rates]), "Not all label smoothing rates were floats"
        assert all([rate >= 0 for rate in label_smoothing_rates]), "Not all label smoothing rates were >= to 0"
        assert all([rate <= 1 for rate in label_smoothing_rates]), "Not all label smoothing rates were <= to 1"

        rates = torch.tensor(label_smoothing_rates)
        return cls(label_smoothing_rates)

    def __init__(self,
                 smoothing_rates: torch.Tensor
                 ):
        super().__init__()
        self.smoothing_rates = smoothing_rates

    def sample(self,
               distribution: torch.Tensor,
               mask: torch.Tensor,
               temperature: float)-> torch.Tensor:
        """
        :param distribution:
            The logit distribution we want to sample from.
            We will assume the last dimension is associated with the probabilities.
            Common shape of around (batch, ..., logits)
        :param mask:
            Indicates any logit elements we wish to exclude from sampling. True means
            include during sampling, false means ignore.
        :param temperature: The generation temperature
        :return: A int tensor indicating the sampled vocabulary elements
            Shape is (batch, ...)
        """
        assert temperature >= 0, "temperature must be greater than or equal to zero"

        # We apply a large negative fill to anything that is going to be masked
        distribution = distribution.masked_fill(~mask, -1e9)

        # Apply softmax with temperature scaling to get the probability distribution
        probs = F.softmax(distribution / temperature, dim=-1)

        # Sample from the categorical distribution created by probs
        sampled_indices = torch.multinomial(probs, num_samples=1)
        return sampled_indices.squeeze(-1)

    def loss(self,
             distribution: torch.Tensor,
             targets: torch.Tensor,
             mask: torch.Tensor,
             smoothing_association: torch.Tensor
             ) -> torch.Tensor:
        """
        Computes the loss with the given label smoothing. Reduces down to
        one loss per batch. We assume there is only one batch dimension.

        :param distribution: The logit distribution we intend to sample from
            Shaped something like (batch, items, classes)
        :param targets: The targets we intend to compute the loss with. Ints
            Shaped something like (batch, items)
        :param mask: A mask to apply when taking a loss. True indicates keep the loss
            Shaped something like (batch, items)
        :param batch_dims: The batch dimension we intend to keep around
        :param smoothing_association: The label smoothing association, indicating
               which label smoothing rate to associate with.
        :return: The loss.
            * Will have distribution.shape[:batch_dim] shape.
            * Reduction is mean.
        """

        # Getting separate label smoothing values, and
        # seperate batch losses, working is kind of tricky
        #
        # Basically, what we are going to do is use torch's
        # cross entropy, but run it N times, where N is the
        # number of label smoothing catagories. Each time, we
        # only compute the loss for the associated smoothing
        # rate. We accumulate.
        #
        # Also, we do not use the built-in reduction for cross
        # entropy, so we can sum each batch separately.

        assert targets.shape == mask.shape
        assert smoothing_association.shape == mask.shape



        # We begin by calculating the losses. We get an unreduced
        # loss per target.
        ignore_target = -100
        targets = targets.masked_fill(~mask, ignore_target)
        losses = torch.zeros(targets.shape, dtype=distribution.dtype)
        for i, smoothing_rate in enumerate(self.smoothing_rates):
            selected_indices = smoothing_association == i
            subtargets = targets.masked_fill(~selected_indices, ignore_target)
            losses += F.cross_entropy(input = distribution.movedim(-1, 1),
                                      target=subtargets.long(),
                                      reduction = "none",
                                      ignore_index=ignore_target,
                                      label_smoothing=smoothing_rate
                                      )

        # We reduce down to only the batch dimensions. We sum up how many active elements are
        # part of each batch. We sum up the batch losses, and normalize.

        mask = mask.flatten(1)
        losses = losses.flatten(1)

        num_active = mask.sum(dim=-1).float()
        losses = losses.sum(dim=-1)
        loss = losses/num_active
        return loss
registry.register_association("vocab_distribution", "controller_adapter")
