import inspect
import textwrap
import typeguard
import typing
import types
import torch
from torch import nn
from typing import (Type, Dict, Any, Optional, Generic, TypeVar, Union, Tuple,
                    get_type_hints,
                    get_origin, get_args, Callable)

RegisteredType = TypeVar('RegisteredType')


def get_method_type_hints(cls: Type[Any],
                          method_name: str,
                          expect_return: bool) -> Tuple[Dict[str, Any], Any]:
    """
    Gets the type hints for the parameters and return type of a provided method
    from the given class, excluding the first parameter (usually 'self').

    :param cls: The provided class whose method type hints are to be extracted.
    :param method_name: The name of the method whose type hints are being fetched.
    :param expect_return: Whether to throw an error if this does not have a return
    :return: A tuple containing:
        - A dictionary mapping method parameter names to their type hints.
        - The return type hint of the method.
    :raises ValueError: If the method does not have a return type annotation
                        or if any parameter is missing a type annotation.
    """
    # Get the method from the class
    method = getattr(cls, method_name, None)
    if method is None:
        raise ValueError(f"Class {cls.__name__} does not have a method named '{method_name}'.")

    # Get the method signature
    method_signature = inspect.signature(method)
    method_params = method_signature.parameters
    param_names = list(method_params.keys())

    # Exclude the first parameter (usually 'self')
    param_names = param_names[1:]

    # Build a dictionary of parameter names to their type hints
    parameter_types = {}
    for name in param_names:
        if method_params[name].annotation == inspect.Parameter.empty:
            raise TypeError(f"Parameter '{name}' in method '{method_name}' of class '{cls.__name__}' does not have a type annotation.")
        parameter_types[name] = method_params[name].annotation

    # Get the return type hint and raise an error if it is missing
    return_type = method_signature.return_annotation
    if return_type == inspect.Signature.empty:
        if expect_return:
            raise TypeError(f"Method '{method_name}' in class '{cls.__name__}' does not have a return type annotation.")
        else:
            return_type = None
    return parameter_types, return_type

def get_constructor_type_hints(cls: Type[Any]) -> Dict[str, Any]:
    """
    Gets the type hints from the constructor (__init__) of a provided class,
    excluding the first parameter (usually 'self').

    :param cls: The layer to get the type hints from
    :return: The type hint params, and return.
    """
    # Get the constructor type hint by fetching off init
    output, _ = get_method_type_hints(cls, '__init__', expect_return=False)
    return output

def get_forward_type_hints(cls: Type[Any]) -> Tuple[Dict[str, Any], Any]:
    """
    Gets the type hints from the forward method of a provided torch layer

    :param cls: The provided torch layer
    :return: The parameters and their type hints, then the return type hint.
    """
    # Get the forward type hint by fetching off forward
    return get_method_type_hints(cls, 'forward', expect_return=True)

def is_type_hint(obj: Any) -> bool:
    """
    Checks if the passed object is a valid type hint.
    :param: obj: The object to be checked.
    :return: True if the object is a valid type hint, False otherwise.
    """
    # Check if it's a generic type (List[int], Dict[str, int], etc.)
    if get_origin(obj) is not None:
        return True

    # Check if it's a special typing construct (Union, Optional, etc.)
    if isinstance(obj, typing._SpecialForm):  # type: ignore
        return True

    # Check if it's a concrete class (int, str, etc.)
    if isinstance(obj, type):
        return True

    # If none of the conditions match, it's not a valid type hint
    return False


def is_same_type_hint(required_type_hint: Any,
                      proposed_type_hint: Any) -> bool:
    """
    Checks if type hints are the same
    :param required_type_hint: The type hint of the required parameter.
    :param proposed_type_hint: The type hint of the proposed parameter.
    :return: The result of the check.
    """
    # Get the origin of both type hints (e.g., List, Tuple, Dict)
    origin1 = get_origin(required_type_hint)
    origin2 = get_origin(proposed_type_hint)

    # In a special case, NoneType and None mean the same thing. Exchange
    if required_type_hint is types.NoneType:
        required_type_hint = None
    if proposed_type_hint is types.NoneType:
        proposed_type_hint = None


    # When both origins are not the same, this is not the same type hint
    if origin1 != origin2:
        return False

    # When both origins are none, they are concrete classes. They had better be
    # the same concrete class
    if origin1 is None:
        return required_type_hint == proposed_type_hint

    # They were not concrete classes. That means they were generics
    # We must now process the generics recursively. Get the arguments off
    # of them.

    args1 = get_args(required_type_hint)
    args2 = get_args(proposed_type_hint)

    # If the number of arguments differs, the type hints are different
    if len(args1) != len(args2):
        return False

    # Recursively compare the type arguments
    for arg1, arg2 in zip(args1, args2):
        if not is_same_type_hint(arg1, arg2):
            return False
    return True


def is_sub_type_hint(origin_hint: Any,
                     subhint: Any
                     ) -> bool:
    """
    Checks if the subhint type is equal to or a subtype of the origin
    type hint. It allows for the elegant handling of "type reduction"
    :param origin_hint: The origin type hint to compare to
    :param subhint: The subhint we are currently busy comparing ourselves to
    :return: Whether or not the two type hints are compatible.
    """

    # The Any type is a wildcard and will match anything
    if origin_hint is Any:
        return True

    # If the two type hints are the same, then they are matching
    # sub type hints. Return true.
    if is_same_type_hint(origin_hint, subhint):
        return True

    # They were not the same type hints. However, the origin could be made up of
    # a Union, which then matches somewhere when taken apart.
    if get_origin(origin_hint) is Union:
        for item in get_args(origin_hint):
            if is_sub_type_hint(item, subhint):
                return True
        return False


    # It was not a union. But it COULD still be a generic, with unions inside of them.
    # So we take apart the generics and process them. If the taken apart generics are all
    # true, then some union node deeper in resolved to a working value

    origin = get_origin(origin_hint)
    suborigin = get_origin(subhint)

    if origin != suborigin:
        # There is no way for them to be compatible if the generics
        # are of different types at this location
        return False

    if origin is None and origin_hint != subhint:
        # These were concrete. They did not match. False
        return False

    args1 = get_args(origin_hint)
    args2 = get_args(subhint)

    # We have to process tuples separately if they have variable length args
    if len(args1) != len(args2):
        if origin is not tuple:
            # Tuple can have variable length type hints, but nothing else can.
            # This is because tuple can use the ellipse feature. Technically so can
            # set, but we will not support that.
            return False
        elif args1[-1] != Ellipsis:
            # The ONLY variable length configuration that is allowed is ellipse
            return False
        else:
            # Everything up to the ellipse region must match
            standard_length = len(args1) - 1
            for arg1, arg2 in zip(args1[:standard_length], args2[:standard_length]):
                if not is_sub_type_hint(arg1, arg2):
                    return False

            # Then, the extensions must match the last provided solid type
            for extra in args2[standard_length:]:
                if not is_sub_type_hint(extra, args1[standard_length-1]):
                    return False
            return True

    # Handle every other generic.
    for arg1, arg2 in zip(args1, args2):
        if not is_sub_type_hint(arg1, arg2):
            return False
    return True

class TorchLayerRegistry(Generic[RegisteredType]):
    """
    A specialized registry mechanism designed specifically to provide an
    interface for handling the creation of torch layers as part of setting up
    a model. It is a generic class.

    ---- Purpose ----

    The `TorchLayerRegistry` allows developers to set up and swap various components
    in a torch model by defining clear interfaces between them. This facilitates
    the trial of different torch layers with minimal code changes by registering
    and validating these layers against the defined interface.

    The registry supports the dynamic registration of torch layers by name,
    while ensuring that the registered classes implement the required constructor
    parameters, forward method parameters, and return types. The registry enforces
    type safety through validation mechanisms.

    ---- Methods ----

    * __init__: Initializes the registry with specified interface requirements
                It also supports registry indirection to handle more complex
                layer-building logic.

    * register_class: Registers a torch layer by validating its constructor and
                      forward method against the predefined interface. Throws
                      errors for any mismatches in expected types.

    * register: A decorator that allows the use of `register_class` in a decorator
                style, offering more convenience when registering classes.

    * build: Builds a registered layer by passing in the necessary parameters and
             running any sub-builders if required.

    ---- Abstract Interface ----

    In order to use the class, you are supposed to define an abstract interface,
    which is then plumbed for the typing details which can be utilized to manage
    the implementation. These must be

    * torch nn.Module layers

    And they must implement

    * __init__: With full parameter type hints
    * forward: With full parameter and return type hints

    This will then be examined with inspect, so beware of monkey patching, as
    that may not show up correctly!

    ---- Builder Indirection ----

    The **Builder Indirection** feature is essential for constructing more complex models
    by allowing a registry to include other builders as part of its constructor.

    Example: Consider building a transformer model that consists of multiple components
    such as feedforward layers, self-attention layers, and cross-attention layers.
    Instead of manually creating each layer, you can nest builders into the transformer
    registry.

    * Builder Indirection works by allowing a constructor argument to be another builder.
    * If a required constructor parameter is not explicitly provided when calling `build`,
      the builder will automatically invoke the corresponding sub-builder to generate the
      missing layer. This can happen recursively, allowing you to build deeply nested models
      with minimal manual intervention.

    In contrast, otherwise we would have to create inflexible layers that could not be
    parameter injected without this additional functionality.

    This enables modularity and flexibility by allowing complex networks to be built from
    simple, reusable building blocks.

    ---- Examples ----

    *** Simple linear***

    As an example, we could setup a registry for the linear layer action something like follows,
    then register the layer, and then build. It is also worth noting that each implementation
    may have extra specific details that must be required. For instance, below bias was never
    defined as a required parameter, but I can still pass it in to this implementation!

    ```python

    # Create abstract linear

    class AbstractLinear(nn.Module):

        def __init__(self, in_features: int, out_features: int):
            ...

        def forward(self, tensor: torch.Tensor)->torch.Tensor:
            ....

    linear = TorchLayerRegistry[AbstractLinear]("Linear", AbstractLinear)

    # Stores torch's linear
    linear.register_class("normal",nn.Linear)

    # Builds a copy
    instance = linear.build("normal", in_features=3, out_features=4, bias=False)
    ```

    It should also be noted that keywords must match exactly. For instance, the following
    would NOT be able to register torch's linear successfully

    ```python

    # Implement abstract linear. Note the keywords will not line up

    class AbstractLinear(nn.Module):

        def __init__(self, d_in: int, d_out: int, *params, **kwargs):
            ...

        def forward(self, tensor: torch.Tensor)->torch.Tensor:
            ....

    # Create a registry. We did not name in feature correct
    linear = TorchLayerRegistry[AbstractLinear]("Linear", AbstractLinear)

    # Fails to store linear. in != in_feature
    linear.register_class("incorrect",nn.Linear)
    ```

    Finally, it should be noted that the builder mechanism is designed to expect
    it is being passed in an entire config setting, and will only pull the entries it
    actually needs. For instance, the following would work fine

    ```python

    # Creates a registry
    linear = TorchLayerRegistry[AbstractLinear]("Linear", AbstractLinear)

    # Stores torch's linear
    linear.register_class("normal",nn.Linear)

    # Builds an instance. Notice how we are passing in features
    # that are not needed? Well, presumably something else needs them, and we ignore it!
    instance = linear.build("normal", in_features=3, out_features=4, attention_type="normal")
    ```

    This means it is quite easy to couple parameters, like say d_model, when using builder
    indirection

    *** Builder Indirection ***

    Builder indirection is where the registry class comes into its
    own. You see, constructor arguments may be specified in terms of keyword, typehint.
    OR in terms of keyword, builder. In the case of specifying by the builder, we
    expect to initialize with that builder and place it in that location.

    Lets see some examples

    ```python
    # Define abstract classes
    class abstract_attention(nn.Module):
        ...

    class abstract_feedforward(nn.Module):
        ...

    # Define a builder for an attention layer and for feedforward
    attention_builder = TorchLayerRegistry[abstract_attention]("Attention", abstract_attention)
    feedforward_builder = TorchLayerRegistry[abstract_feedforward]("Feedforward", abstract_feedforward)

    # Implement them somehow.

    ....irrelevant_code
    attention_builder.register_class("standard_attention", attention_implementation)
    feedforward_builder.register_class("standard_feedforward", feedforward_implementation)

    # Setup indirection for a transformer
    class abstract_transformer(nn.Module):
        ...



    indirection = {"self_attention" : attention_builder, "feedforward" : feedforward_builder}
    transformer_builder = TorchLayerRegistry[abstract_transformer]("Transformer", indirection)

    # Register an implementation

    ....irrelevant_code
    transformer_builder.register_class("transformer", transformer_implementation)

    # Construct a transformer. Notice the use of the names of the registries,
    # that is of 'Attention' and 'Feedforward',
    # in order to specify what kind of attention and feedforward to use

    transformer = transformer_builder.build("transformer",
                                             d_model=128,
                                             d_hidden=512,
                                             heads=4,
                                             Attention="standard_attention"
                                             Feedforward="standard_feedforward"

    """
    @property
    def layer_abstract_type(self)->Type[Any]:
        return self.__orig_class__

    @property
    def forward_parameter_schema(self)->Dict[str, Any]:
        return self.forward_parameters

    @property
    def forward_return_schema(self)->Any:
        return self.forward_returns

    @property
    def constructor_schema(self)->Dict[str, Any]:
        return self.constructor_required_schema

    def __init__(self,
                 registry_name: str,
                 abstract_layer: nn.Module,
                 **registry_indirections: Dict[str, 'TorchLayerRegistry']
                 ):
        """
        Initialize the registry builder. If desired, the user
        can provide certain interface parameters that the attached
        implementations MUST provide to be bound to.

        :param registry_name: The name of the registry. Keywords at higher level registries with this
                     name will be redirected into this builder during build
        :param abstract_layer: The abstract interface layer associated with this builder.
                               It will have forward and the constructor searched for type hints.
        :param registry_indirections: Any registry indirections to perform. When you provide keywords
                                     here that collide with constructor keywords from the interface,
                                     it is possible to build that parameter by indirection.
        """

        # Get the interfaces for the constructor and for forward

        constructor_interface = get_constructor_type_hints(abstract_layer)
        forward_params, forward_returns = get_forward_type_hints(abstract_layer)

        # Now, go over the registry indirections and make sure they make sense
        for name, registry in registry_indirections.items():
            if not isinstance(registry, TorchLayerRegistry):
                msg= f"""
                Issue on registry indirection named '{name}'
                The indirection value was not a TorchLayerRegistry.
                """
                msg = textwrap.dedent(msg)
                raise TypeError(msg)

            if name not in constructor_interface:
                msg = f"""
                Indirection registry of name '{name}' was not found among
                constructor arguments
                """
                msg = textwrap.dedent(msg)
                raise TypeError(msg)
            if not is_sub_type_hint(constructor_interface[name], registry.constructor_schema):
                msg = f"""
                Abstract interface and indirection registry named {registry.registry_name} were
                found to be incompatible
                """
                msg = textwrap.dedent(msg)
                raise TypeError(msg)

        # Setup and otherwise store intake details.
        self.forward_parameters = forward_params
        self.forward_returns = forward_returns

        self.registry_name = registry_name
        self.constructor_required_schema = constructor_interface
        self.registry_indirections = registry_indirections

        self.registry: Dict[str, Dict[str, Any]] = {}

    def validate_forward_schema(self, cls: Type[RegisteredType]):
        """
        Validates that the class has a forward schema that is compatible
        with the registry.
        :param cls: The class to examine
        :raises TypeError: If the class has a forward schema that is not compatible
        """
        forward_parameters, forward_returns = get_forward_type_hints(cls)

        # If the implementation differs, throw.
        if forward_parameters.keys() != self.forward_parameters.keys():
            msg = f"""
            The forward interface expected to see, in order, features: 
            
            {self.forward_parameters.keys()}
            
            However, actually implemented was:
            
            {forward_parameters.keys()}
            """
            msg = textwrap.dedent(msg)
            raise TypeError(msg)

        # Check the actual types match
        for name in forward_parameters.keys():
            if not is_sub_type_hint(self.forward_parameters[name], forward_parameters[name]):
                msg = f"""
                Issue found at parameter {name}
                The forward interface expected to see a type of: {self.forward_parameters[name]}.
                However, actual implementation was: {forward_parameters[name]}.
                These are not compatible.
                """
                msg = textwrap.dedent(msg)
                raise TypeError(msg)

        # Check the types of the return match
        if not is_sub_type_hint(self.forward_returns, forward_returns):
            msg = f"""
            Forward interface's return different from implementation.
            
            The forward interface was expected to match type {self.forward_returns}.
            However, actually observed: {forward_returns}.
            """
            msg = textwrap.dedent(msg)
            raise TypeError(msg)

    def validate_constructor_schema(self, cls: Type[RegisteredType]):
        """
        Validates the constructor schema is valid.
        :param cls: The class to validate the constructor schema on
        :raises TypeError: If the class has a constructor schema that is not compatible
        """

        constructor_schema = get_constructor_type_hints(cls)

        # Loop over any required types. Verify existence. Verify they
        # have the same type hints.
        for required_name, required_type in self.constructor_required_schema.items():
            if required_name not in constructor_schema:
                msg = f"""
                The registry expects registered classes to accept a constructor
                keyword of name {required_name}. That was not found in class
                of name '{cls.__name__}'
                """
                msg = textwrap.dedent(msg)
                raise TypeError(msg)
            if not is_sub_type_hint(required_type, constructor_schema[required_name]):
                msg = f"""
                The registry expects the registered class to have a parameter typehint
                for '{required_name}' of '{required_type}'. However, class named {cls.__name__}
                implements it as {constructor_schema[required_name]}
                """
                msg = textwrap.dedent(msg)
                raise TypeError(msg)

    def register_class(self,
                       name: str,
                       class_type: Type[RegisteredType]
                       ):
        """
        Registers a class in the registry by its name, type, and constructor parameters.

        Args:
            name (str): The name under which to register the class.
            class_type (Type[T]): The class type to be registered,
                                  constrained by the generic type T.

                                  This class must have a constructor with fully
                                  expressed type hints.
        """
        # Get the required schema for the constructor
        constructor_schema = get_constructor_type_hints(class_type)

        # Validate
        self.validate_forward_schema(class_type)
        self.validate_constructor_schema(class_type)

        # Everything passed validation. Store
        self.registry[name] = {
            "type": class_type,
            "params": constructor_schema
        }

    def is_optional_type(self, type_hint: Type) -> bool:
        """
        Determines if a type hint is an Optional type.

        Args:
            type_hint (Type): The type hint to check.

        Returns:
            bool: True if the type is Optional, otherwise False.
        """
        origin = getattr(type_hint, '__origin__', None)
        args = getattr(type_hint, '__args__', ())
        return origin is Union and type(None) in args

    def build(self, name: str, **params: Dict[str, Any]) -> RegisteredType:
        """
        Instantiates a registered class using the provided parameters.

        Args:
            name (str): The name of the registered class to instantiate.
            params (Dict[str, Any]): A dictionary containing the parameters to pass
                to the constructor of the class.

        Returns:
            T: An instance of the registered class of the expected generic type.

        Raises:
            ValueError: If the class is not registered, or if a required parameter is missing or has an invalid type.
        """
        if name not in self.registry:
            raise ValueError(f"Class '{name}' is not registered.")

        class_info = self.registry[name]
        class_type = class_info["type"]
        required_params = class_info["params"]

        constructor_args = {}
        for param_name, param_type in required_params.items():
            # Check if the parameter exists in the passed params
            if param_name not in params:
                # There are several reasons why param names may be missing, and we can recover
                # from some of them

                if param_name in self.registry_indirections:
                    # If the name was missing, but found in the builders group, we are likely
                    # being asked to recurrently run a child builder. Lets see if we can do that

                    builder = self.registry_indirections[param_name]
                    if builder.registry_name not in params:
                        msg = f"""
                        Attempt was made to run a subbuilder. However, keyword '{builder.registry_name}'
                        was not found, preventing this from happening
                        """
                        msg = textwrap.dedent(msg)
                        raise ValueError(msg)

                    build_class = params[builder.registry_name]
                    constructor_args[param_name] = builder.build(build_class, **params)

                if self.is_optional_type(param_type):
                    # Optional means none
                    constructor_args[param_name] = None
                else:
                    # Not recoverable.
                    raise ValueError(f"Missing required parameter '{param_name}' for class '{name}'.")
            else:
                value = params[param_name]
                # Use typeguard to check the type of the parameter
                try:
                    typeguard.check_type(value, param_type)
                    constructor_args[param_name] = value
                except typeguard.TypeCheckError as e:
                    raise TypeError(f"Parameter '{param_name}' has an invalid type: {e}") from e

        # Instantiate the class with the collected arguments
        return class_type(**constructor_args)

    def register(self,
                 name: str
                 ) -> Callable[[Type[RegisteredType]], Type[RegisteredType]]:
        """
        Decorator register to use with registry builder
        :param name: The name of the feature to register
        :return: The registered class
        """

        def register(class_type: Type[RegisteredType]) -> Type[RegisteredType]:
            self.register_class(name, class_type)
            return class_type

        return register



