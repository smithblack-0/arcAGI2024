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

    def validate_config_compatible(self,
                                   config: Dict[str, Any],
                                   config_spec: Dict[str, Type[Any]]
                                   ):
        if len(config) != len(config_spec):
            msg = f"""
            Config passed to create distribution adapter was not compatible. 
            
            The distribution adapter expected a config dictionary of length {len(config_spec)},
            however it got a config dictionary of length {len(config)}.
            """
            msg = textwrap.dedent(msg)
            raise ValueError(msg)
        if set(config.keys()) != set(config_spec.keys()):
            msg = f"""
            Config passed to create distribution adapter was not compatible.
            
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

    def get_structure(self, name: str) -> Dict[str, Type[Any]]:
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
        self.validate_config_compatible(config, self.setup_registry[name])
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
    def reinforcement_sample(self,
                             distribution: torch.Tensor | Tuple[torch.Tensor, ...],
                             *controls: Any
                             )->Tuple[torch.Tensor, torch.Tensor]:
        """
        The reinforcement sample method should return a mechanism by which a sample can
        be drawn that is suitable for reinforcement learning. This sampling mechanism's
        result should be natively interpretable by the io adapter embedding mechanism.
        It should also maintain gradients so that gradient descent has something to work
        with.

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


setup_spec = {"use_hard_samples" : bool,
               "use_embedding_bags" : bool,
               "num_resamples" : int,
               "top_k" : int
               }
@registry.registry_decorator("vocab_distribution",
                             "vocabulary_adapter",
                             setup_spec
                             )
class VocabularyDistributionAdapter(DistributionAdapter):
    """
    A distribution adapter designed to be interacting
    with a vocabulary distribution. In this format,
    each entry of a vocabulary is represented by an integer,
    and we expect to see logits matching the vocabulary.

    ---- logit subset sampling ----

    The reinforcement sampling mechanism is capable of using
    only a subset of the original logits when computing the next word.
    We call this logit subset sampling.

    In this mechanism, we restrict the logits in some way to a smaller
    subset of the original vocabulary - in our case, by sampling
    gumbel logits a bunch and keeping top-k. Then, we perform gumbel
    softmax on that subset, and only continue computation based on
    it.

    """

    @classmethod
    def setup(cls,
              use_hard_sample: bool,
              use_embedding_bags: bool,
              num_resamples: int,
              top_k: int,
              )->"VocabularyDistributionAdapter":
        """
        Sets up a vocabulary distribution adapter, ready to be used for
        supervised learning and reinforcement learning. Most passed
        parameters have to do with reinforcement learning - the following
        will have no effect when run in a supervised manner.

        :param use_hard_sample: Whether or not to perform hard sampling when
                                doing gumbel softmax sampling.
        :param use_embedding_bags:
            Whether or not to use the embedding bag mechanism and trim the distribution
            under consideration when doing reinforcement sampling. When false, the following
            parameters have no effect.
        :param num_resamples:
            The number of resamples to perform when doing embedding bag gumbel softmax.
        :param top_k:
            How many top-k entries to keep during each resample during embedding bag
            gumbel softmax.
        """

        return cls(use_hard_sample,
                   use_embedding_bags,
                   num_resamples,
                   top_k,
                   )

    def __init__(self,
                 use_hard_sample: bool,
                 use_embedding_bags: bool,
                 num_resamples: int,
                 top_k: int,
                 ):
        super().__init__()

        assert num_resamples > 0
        assert top_k > 0

        self.use_hard_sample = use_hard_sample
        self.use_embedding_bags = use_embedding_bags
        self.num_resamples = num_resamples
        self.top_k = top_k

    def reinforcement_sample(self,
                             distribution: torch.Tensor,
                             temperature: float | torch.Tensor
                             )->torch.Tensor | Tuple[torch.Tensor, ...]:
        """
        Performs reinforcement sampling from the distribution. The exact
        return varies depending on how we were configured.

        If use_embedding_bags was false, we simply do gumbel softmax sampling,
        harden it if needed, then return the distribution. The result is a single
        tensor of classes.

        If it is true, however, we begin by trimming the vocabulary to a certain
        randomly selected percentage of the logits, getting a vocabulary subset.
        Then, we inject gumbel noise and select the top-k and rand-k from the vocabulary
        subset. Finally, we sample from THIS distribution, get probabilities, then
        return the bags of probabilities. Note that for compatibility with

        :param distribution: Unnormalized log probabilities (e.g., output of a linear layer).
        :param temperature: Controls the smoothness of the distribution.
        :param hard: If True, probabilities are set to one-hot value of 1.
        :return: Tensor of the same shape as logits, representing the sampled probabilities.
        """

        assert temperature > 0, "temperature cannot become less that or equal to zero during reinforcement learning"

        # Handle simple gumbel softmax without any frills. We get
        # back a probability distribution.
        if not self.use_embedding_bags:
            output = F.gumbel_softmax(distribution, tau=temperature, hard=self.use_hard_sample)
            return output

        # Handle reduced gumbel softmax with resampling.
        #
        # This proceeds in two steps:
        #
        # 1): We get a collection of probable top-k logits to process
        # 2): We perform gumbel sampling against these logits.


        ## Step 1:
        #
        # Basically, we start from a set consisting of the entire
        # vocabulary index, then generate gumbel logits from the
        # set, then keep the top-k indices. These indices are then
        # removed from the canidate logits.
        #
        # This is repeated a number of times until we have drawn
        # N samples consisting of K logits each.
        #
        # The purpose is to get some idea about what kinds of logits
        # might matter in various scenarios.

        logits = distribution.clone()
        final_vocabulary_indices = []
        for _ in range(self.num_resamples):
            # Create gumbel logits based on the existing logit distribution
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(distribution)))
            gumbel_logits = (distribution + gumbel_noise) / temperature

            # Select the top k to keep.
            top = torch.topk(gumbel_logits, k=self.top_k)
            selected_indices = top.indices

            # Update the logits, and mask out anything that was
            # selected. We do this by setting it to a very large
            # negative value.

            logits.scatter_(-1, selected_indices, -1e+9)

            # Store
            final_vocabulary_indices.append(selected_indices)

        target_vocabulary = torch.cat(final_vocabulary_indices, dim=-1)

        ## Move onto step two.
        #
        # We perform gumbel softmax with the reduced set, then return the
        # embedding bags to target

        logit_subset = distribution.gather(dim=-1, index=target_vocabulary)
        y = F.gumbel_softmax(logit_subset, tau=temperature, hard=self.use_hard_sample)
        return target_vocabulary, y

    def sample(self,
               distribution: torch.Tensor,
               temperature: float)-> torch.Tensor:
        """
        :param distribution:
            The logit distribution we want to sample from.
            We will assume the last dimension is associated with the probabilities.
            Common shape of around (batch, ..., logits)
        :param temperature: The generation temperature
        :return: A int tensor indicating the sampled vocabulary elements
            Shape is (batch, ...)
        """
        assert torch.all(temperature >= 0), "temperature must be greater than or equal to zero"

        # Apply softmax with temperature scaling to get the probability distribution
        probs = F.softmax(distribution / temperature, dim=-1)

        # Sample from the categorical distribution created by probs
        sampled_indices = torch.multinomial(probs, num_samples=1)
        return sampled_indices.squeeze(-1)

    def loss(self,
             distribution: torch.Tensor,
             targets: torch.Tensor,) -> torch.Tensor:
        """

        :param distribution:
        :param targets:
        :return:
        """


