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
        # These were concrete. They did not match. We still need
        # to check if they are subclasses
        if subhint is None:
            return False
        if issubclass(subhint, origin_hint):
            return True
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

def get_signature_type_hints(method_name: str,
                             class_name: str,
                            signature: inspect.Signature,
                            expect_return: bool
                            ) -> Tuple[Dict[str, Any], Any]:
    """
    Gets the type hints for the parameters and return type of a provided method
    from the given class, excluding the first parameter (usually 'self'). Notably,
    it does not distinguish between varidacs and normal typing parameters, but can
    still fetch types off varidacs.

    :param signature: The provided signature.
    :return: A tuple containing:
        - A dictionary mapping method parameter names to their type hints.
        - The return type hint of the method.
    :raises ValueError: If the method does not have a return type annotation
                        or if any parameter is missing a type annotation.
    """
    # Get the method signature
    method_params = signature.parameters
    param_names = list(method_params.keys())

    # Exclude the first parameter (usually 'self')
    param_names = param_names[1:]

    # Build a dictionary of parameter names to their type hints
    parameter_types = {}
    for name in param_names:
        if method_params[name].annotation == inspect.Parameter.empty:
            raise TypeError(f"Parameter '{name}' in method '{method_name}' of class '{class_name}' does not have a type annotation.")
        parameter_types[name] = method_params[name].annotation

    # Get the return type hint and raise an error if it is missing
    return_type = signature.return_annotation
    if return_type == inspect.Signature.empty:
        if expect_return:
            raise TypeError(f"Method '{method_name}' in class '{cls.__name__}' does not have a return type annotation.")
        else:
            return_type = None
    return parameter_types, return_type

def get_constructor_spec(abstract_cls: Type[nn.Module]) -> inspect.Signature:
    """
    Gets constructor data. This includes the signature, and the typing on each parameter.
    """
    signature = inspect.signature(abstract_cls.__init__)
    return signature

def get_abstract_spec(abstract_cls: Type[nn.Module]) -> Dict[str, inspect.Signature]:
    """
    Collects abstract method specs from a given abstract class.
    :param abstract_cls: The class to inspect.
    :return: A dictionary with method names as keys and tuples of (signature, parameter types, return type) as values.
    """
    abstract_specs = {}
    for name, method in abstract_cls.__dict__.items():
        if inspect.isfunction(method) and hasattr(method, "__isabstractmethod__"):
            signature = inspect.signature(method)
            abstract_specs[name] = signature
    return abstract_specs

def get_concrete_specs(abstract_specs: Dict[str, inspect.Signature],
                       concrete_cls: Type[Any]
                       )->Dict[str, inspect.Signature]:
    concrete_specs = {}
    for name in abstract_specs.keys():
        concrete_specs[name] = inspect.signature(getattr(concrete_cls, name))
    return concrete_specs


class AbstractClassSchemaEnforcer:
    """
    `AbstractClassSchemaEnforcer` ensures that implementations of an abstract class match
    the specified interfaces for all abstract methods, including but not limited to `__init__` and `forward`.
    This enforces consistency across implementations within a registry.
    """

    def verify_abstract_schema(self,
                               signature: inspect.Signature,
                               method: str,
                               expect_return: bool
                               ):
        """
        Verify that a method signature is sane
        :param signature: The signature to verify
        :param method: The method it is called
        :param expect_return: Whether it must have a return type annotation
        """
        has_self_been_skipped = False
        for name, parameter in signature.parameters.items():
            if not has_self_been_skipped:
                has_self_been_skipped = True
                continue

            if parameter.annotation == inspect.Parameter.empty:
                msg = f"""
                Issue with method '{method}'
                All parameters were expected to have a type annotation.
                However, parameter called {name} had none
                """
                msg = textwrap.dedent(msg)
                raise TypeError(msg)
        if signature.return_annotation == inspect.Signature.empty and expect_return:
            msg = f"""
            Issue with method '{method}'
            The return was expected to have a type annotation, but 
            none was detected
            """
            msg = textwrap.dedent(msg)
            raise TypeError(msg)



    def __init__(self, cls_abstract: Type[nn.Module]):
        """
        Initializes the schema enforcer with the abstract class's specifications.
        """
        self.name = cls_abstract.__name__
        self.cls_interface = cls_abstract

        contructor_signature = get_constructor_spec(cls_abstract)
        methods_signature = get_abstract_spec(cls_abstract)

        self.verify_abstract_schema(contructor_signature, "__init__", False)
        for name, signature in methods_signature.items():
            self.verify_abstract_schema(signature, name, True)

        self.constructor_signature = contructor_signature
        self.abstract_methods_signature = methods_signature

    def verify_signatures_compatible(self,
                                     method_name: str,
                                     abstract_signature: inspect.Signature,
                                     concrete_signature: inspect.Signature,
                                     ):

        # If the abstract signature is the same as the concrete signature
        # no further checks are needed
        if abstract_signature == concrete_signature:
            return

        if abstract_signature.parameters != concrete_signature.parameters:
            # This is possibly because we had a kwarg or arg and possibly because
            # we had a bad specification. To narrow this down, we try to bind the concrete
            # signature paramters onto the abstract parameter specification. Then we verify
            # all types were properly reducable
            try:
                args_routing = (inspect.Parameter.POSITIONAL_ONLY,
                                inspect.Parameter.VAR_POSITIONAL,
                                inspect.Parameter.POSITIONAL_OR_KEYWORD)
                kwargs_routing = (inspect.Parameter.KEYWORD_ONLY,
                                  inspect.Parameter.VAR_KEYWORD)

                args = [arg for arg in concrete_signature.parameters.values() if arg.kind in args_routing]
                kwargs = {key: value for key, value in concrete_signature.parameters.items()
                          if value.kind in kwargs_routing}
                bound_parameters = abstract_signature.bind(*args, **kwargs)
            except Exception as e:
                msg = """
                  Could not bind concrete implementation to abstract sequence: {e}
                  """
                msg = textwrap.dedent(msg)
                raise TypeError(msg) from e

            has_self_been_skipped = False
            for name, abstract_parameter in abstract_signature.parameters.items():

                # Skip this if we are dealing
                if not has_self_been_skipped:
                    has_self_been_skipped = True
                    continue
                # Skip variable args if nothing was bound to it
                if (abstract_parameter.kind == inspect.Parameter.VAR_POSITIONAL
                        and name not in bound_parameters.arguments):
                    continue

                # Skip variable kwargs if nothing was bound to it.
                if (abstract_parameter.kind == inspect.Parameter.VAR_KEYWORD
                        and name not in bound_parameters.arguments):
                    continue


                # Get the concrete cases. This is either a single parameter,
                # or a list or dict of them if we are dealing with a varidac
                concrete_cases = bound_parameters.arguments[name]
                if isinstance(concrete_cases, inspect.Parameter) and \
                        concrete_cases.annotation != abstract_parameter.annotation:
                    # This is a normal parameter. Just verify it.
                    if not is_sub_type_hint(abstract_parameter.annotation, concrete_cases.annotation):
                        msg = f"""
                        Issue detected with implementation of '{method_name}'.
                        
                        In parameter {name} of the abstract implementation, we were 
                        given type of '{abstract_parameter.annotation}'. However, implemented
                        '{concrete_cases.annotation} which was not compatible.
                        """
                        msg = textwrap.dedent(msg)
                        raise TypeError(msg)
                if isinstance(concrete_cases, tuple):
                    # Variable args need to be looped over and checked
                    for concrete_case in concrete_cases:
                        if not is_sub_type_hint(abstract_parameter.annotation, concrete_case.annotation):
                            msg = f"""
                            Issue detected with implementation of '{method_name}'.
                            The abstract type had variable args named {name}, with
                            a type of '{abstract_parameter.annotation}'. However, one
                            of the arguments had type '{concrete_case.annotation}
                            """
                            msg = textwrap.dedent(msg)
                            raise TypeError(msg)
                elif isinstance(concrete_cases, dict):
                    # Keyword arguments need to be looped over and checked
                    # as well
                    for kwarg_name, concrete_case in concrete_cases.items():
                        if not is_sub_type_hint(abstract_parameter.annotation, concrete_case.annotation):
                            msg = f"""
                            Issue detected with implementation of '{method_name}'.
                            The abstract type was a variable kwargs feature named {name},
                            with attached type of '{abstract_parameter.annotation}'. However,
                            kwarg with name '{kwarg_name}' in the concrete implementation
                            had type of '{concrete_case.annotation}'
                            """
                            msg = textwrap.dedent(msg)
                            raise TypeError(msg)

    @property
    def constructor_schema(self) -> inspect.Signature:
        return self.constructor_signature

    @property
    def abstract_schemas(self) -> Dict[str, inspect.Signature]:
        return self.abstract_methods_signature

    def validate_constructor_specification(self, cls_implementation: Type[nn.Module]):
        """
        Validates that the constructor of an implementation matches the abstract class's constructor schema.
        """
        constructor_signature = get_constructor_spec(cls_implementation)
        expected_signature = self.constructor_signature


    def validate_method_specification(self, cls_implementation: Type[nn.Module], method_name: str):
        """
        Validates that a specific method in the implementation matches the abstract method specification.
        """
        if method_name not in self.abstract_methods_signature:
            raise ValueError(f"Method '{method_name}' is not an abstract method in '{self.name}'.")

        # Fetching the specs to compare
        expected_signature, expected_params, expected_return = self.abstract_methods_signature[method_name]
        actual_signature = inspect.signature(getattr(cls_implementation, method_name))
        actual_params, actual_return = get_signature_type_hints(method_name, cls_implementation.__name__, actual_signature, expect_return=True)

        # Parameter checks
        if expected_params.keys() != actual_params.keys():
            msg = f"""
            Method '{method_name}' parameters mismatch.
            Expected: {expected_params.keys()}
            Found: {actual_params.keys()}
            """
            raise TypeError(textwrap.dedent(msg))

        for name in expected_params.keys():
            if not is_sub_type_hint(expected_params[name], actual_params[name]):
                msg = f"""
                Parameter '{name}' in method '{method_name}' does not match.
                Expected: {expected_params[name]}
                Found: {actual_params[name]}
                """
                raise TypeError(textwrap.dedent(msg))

        # Return type check
        if not is_sub_type_hint(expected_return, actual_return):
            msg = f"""
            Return type of method '{method_name}' does not match.
            Expected: {expected_return}
            Found: {actual_return}
            """
            raise TypeError(textwrap.dedent(msg))

    def __call__(self, cls_implementation: Type[nn.Module]):
        """
        Validates the class implementation against the abstract class schema for all methods and constructor.
        """
        if not issubclass(cls_implementation, self.cls_interface):
            msg = f"""
            Implementation '{cls_implementation.__name__}' is not a subclass of abstract interface '{self.cls_interface.__name__}'.
            """
            raise TypeError(textwrap.dedent(msg))

        # Validate constructor
        self.validate_constructor_specification(cls_implementation)

        # Validate each abstract method
        concrete_signatures = get_concrete_specs(self.abstract_methods_signature, cls_implementation)
        for name in self.abstract_methods_signature.keys():
            abstract_signature = self.abstract_methods_signature[name]
            concrete_signature = concrete_signatures[name]
            self.verify_signatures_compatible(name, abstract_signature, concrete_signature)
class ImplementationBuilder:
    """
    A helper class responsible for validating constructor arguments against an implementation's
    expected type hints and for building instances of torch layers.

    The `ImplementationBuilder` ensures that any parameters provided for initializing a class
    match the expected types as specified by the class's constructor. It validates arguments,
    handles variable-length arguments (varargs and kwargs), and checks compatibility with
    expected type hints using `typeguard`. Upon successful validation, it builds and returns
    an instance of the class.

    ---- Purpose ----

    This class provides a mechanism for:
    * Ensuring that arguments passed for constructing a layer match the type hints provided
      in the implementation.
    * Handling complex cases, such as variable-length arguments or keyword arguments.
    * Providing a unified interface for validating and instantiating torch layer implementations.

    ---- Properties ----

    * `constructor_types`: A dictionary mapping constructor parameter names to their expected types.
    * `constructor_signature`: The `inspect.Signature` of the constructor, used to bind arguments and validate them.

    ---- Methods ----

    * `__call__`: Attempts to validate and instantiate the implementation with the provided arguments.
    * `check_spec_matches`: Ensures that the provided parameter matches the expected type, raising a `TypeError` if not.
    * `is_varidac_keywords`: Determines if the given parameter name is a `**kwargs` argument in the constructor.
    * `is_varidac_args`: Determines if the given parameter name is a `*args` argument in the constructor.
    """
    @property
    def constructor_types(self)->Dict[str, Any]:
        signature = self.constructor_spec
        constructor_params = {}
        has_skipped_self = False
        for name, param in signature.parameters.items():
            if not has_skipped_self:
                has_skipped_self = True
                continue
            constructor_params[name] = param.annotation
        return constructor_params

    @property
    def constructor_signature(self)->inspect.Signature:
        return self.constructor_spec

    def is_varidac_keywords(self, parameter: str)->bool:
        return self.constructor_signature.parameters[parameter].kind == inspect.Parameter.VAR_KEYWORD

    def is_varidac_args(self, parameter: str)->bool:
        return self.constructor_signature.parameters[parameter].kind == inspect.Parameter.VAR_POSITIONAL

    def __init__(self,
                 implementation: Type[nn.Module],
                 ):

        # Store away the spec
        self.constructor_spec = get_constructor_spec(implementation)
        self.implementation = implementation
        self.indirections = {}

    def check_spec_matches(self, parameter: Any, type: Type):
        """
        Checks if the spec typing matches
        :param parameter: The thing we are trying to use as a parameter
        :param type: The type it must match
        :raises TypeError: If they do not match
        """
        try:
            typeguard.check_type(parameter, type)
        except typeguard.TypeCheckError as e:
            raise TypeError(f"Parameter '{parameter}' has an invalid type: {e}") from e
    def __call__(self, *args, **kwargs):
        """
        Runs the validation.
        :param args: The args to attempt initialization with
        :param kwargs: The kwargs to attempt initialization with
        :raises TypeError: If a variety of issues occur
        """
        # Get data to use

        signature = self.constructor_signature
        param_types = self.constructor_types

        # Try to bind the arguments to the signature. This now
        # ensures we can associate them downstream with the correct type
        bindings = signature.bind("self",*args, **kwargs)

        # Go over each binding slot. Ensure all in each slot has correct type
        # Note that the bindings feature can now have direct feature, and arg,
        # kwargs dictionaries

        for bound_name, bound_value in bindings.arguments.items():
            if bound_name == "self":
                continue

            # Get the type we must match
            bound_type = param_types[bound_name]

            # the varidac args case requires special handling. In particular,
            # we need to figure out the type, then assert is is applied across
            # all the matches
            if self.is_varidac_args(bound_name):
                for arg in bound_value:
                    self.check_spec_matches(arg, bound_type)

            # The varidac keywords has a similar requirement
            if self.is_varidac_keywords(bound_name):
                for arg in bound_value._values():
                    self.check_spec_matches(arg, bound_type)

            # And if neither, it is just normal binding
            self.check_spec_matches(bound_value, bound_type)

        # We succeeded in validation. Init

        return self.implementation(*args, **kwargs)





class InterfaceRegistry(Generic[RegisteredType]):
    """
    A specialized registry mechanism designed specifically to provide an
    interface for handling the creation of torch layers as part of setting up
    a argAGI2024. It is a generic class.

    ---- Purpose ----

    The `InterfaceRegistry` allows developers to set up and swap various components
    in a torch argAGI2024 by defining clear interfaces between them. This facilitates
    the trial of different torch layers with minimal code changes by registering
    and validating these layers against the defined interface.

    The registry supports the dynamic registration of torch layers by name,
    while ensuring that the registered classes implement the required constructor
    parameters, forward method parameters, and return types. The registry enforces
    type safety through validation mechanisms.

    ---- Methods ----

    * __init__: Initializes the registry with specified interface requirements.
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

    To define an interface, use `@abstractmethod` on methods and properties within
    the abstract class, which will then be enforced by the `InterfaceRegistry` on
    implementations. Methods defined with `@abstractmethod`, such as `__init__`
    or `forward`, become part of the interface that all implementations registered
    in this registry must match.

    Abstract classes must define:

    * __init__ with full parameter type hints
    * Additional methods, like forward, with complete parameter and return type hints

    The registry examines abstract methods and enforces compliance with their
    signatures and typing, ensuring all registered classes conform to the specified
    abstract interface.

    ---- Builder Indirection ----

    The **Builder Indirection** feature is essential for constructing more complex models
    by allowing a registry to include other builders as part of its constructor.

    Example: Consider building a transformer argAGI2024 that consists of multiple components
    such as feedforward layers, self-attention layers, and cross-attention layers.
    Instead of manually creating each layer, you can nest builders into the transformer
    registry.

    * Builder Indirection works by allowing a constructor argument to be another builder.
    * If a required constructor parameter is not explicitly provided when calling `build`,
      the builder will automatically invoke the corresponding sub-builder to generate the
      missing layer. This can happen recursively, allowing you to build deeply nested models
      with minimal manual intervention.

    This enables modularity and flexibility by allowing complex networks to be built from
    simple, reusable building blocks.

    ---- Examples ----

    *** Simple Linear Layer ***

    An example of setting up a registry for a linear layer interface, registering the layer,
    and building an instance. Abstract methods are required to be defined using `@abstractmethod`.

    ```python
    from abc import abstractmethod

    # Define abstract linear interface
    class AbstractLinear(nn.Module):
        def __init__(self, in_features: int, out_features: int):
            ...

        @abstractmethod
        def forward(self, tensor: torch.Tensor) -> torch.Tensor:
            ...

    linear_registry = TorchLayerRegistry[AbstractLinear](AbstractLinear)

    # Register PyTorch's Linear implementation
    linear_registry.register_class("standard", nn.Linear)

    # Build an instance with the required parameters
    instance = linear_registry.build("standard", in_features=3, out_features=4, bias=False)
    ```

    *** Builder Indirection ***

    Builder indirection enables component nesting in the registry.

    ```python
    from abc import abstractmethod

    # Define abstract attention and feedforward classes
    class AbstractAttention(nn.Module):
        @abstractmethod
        def forward(self, x):
            ...

    class AbstractFeedforward(nn.Module):
        @abstractmethod
        def forward(self, x):
            ...

    # Define individual registries
    attention_registry = TorchLayerRegistry[AbstractAttention]("AttentionRegistry", AbstractAttention)
    feedforward_registry = TorchLayerRegistry[AbstractFeedforward]("FeedforwardRegistry", AbstractFeedforward)

    # Define a transformer interface that uses attention and feedforward layers
    class AbstractTransformer(nn.Module):
        def __init__(self, self_attention: AbstractAttention, feedforward: AbstractFeedforward):
            ...

        @abstractmethod
        def forward(self, x):
            ...

    # Define transformer registry with indirection
    transformer_registry = TorchLayerRegistry(
        "TransformerRegistry", AbstractTransformer,
        self_attention=attention_registry,
        feedforward=feedforward_registry
    )

    # Register and build transformer with specified attention and feedforward implementations
    transformer_registry.register_class("transformer_impl", TransformerImplementation)
    transformer_instance = transformer_registry.build(
        "transformer_impl",
        d_model=128, d_hidden=512, heads=4,
        Attention="standard_attention",
        Feedforward="standard_feedforward"
    )
    """
    @property
    def layer_abstract_type(self) -> Type[Any]:
        return self.layer_type

    @property
    def forward_parameter_schema(self) -> Dict[str, Any]:
        return self.forward_parameters

    @property
    def forward_return_schema(self) -> Any:
        return self.forward_returns

    @property
    def constructor_schema(self) -> Dict[str, Any]:
        return self.constructor_required_schema

    def __init__(self,
                 registry_name: str,
                 abstract_layer: nn.Module,
                 **registry_indirections: 'InterfaceRegistry'
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
        # Basic type checking
        if not isinstance(registry_name, str):
            raise TypeError("Registry name must be a string.")
        if not isinstance(abstract_layer, type):
            raise TypeError("abstract_layer must be a class, not an instance.")
        if not issubclass(abstract_layer, nn.Module):
            raise TypeError("abstract_layer must be a flavor of nn.Module")

        # Build the schema enforcer
        schema_enforcer = AbstractClassSchemaEnforcer(abstract_layer)

        # Validate registry indirections
        for name, registry in registry_indirections.items():
            if not isinstance(registry, InterfaceRegistry):
                msg = f"Issue on registry indirection named '{name}'. The indirection value was not a TorchLayerRegistry."
                raise TypeError(msg)


        # Setup and otherwise store intake details
        self.layer_type = abstract_layer
        self.interface_validation = AbstractClassSchemaEnforcer(abstract_layer)
        self.registry: Dict[str, ImplementationBuilder] = {}
        self.indirections = registry_indirections
        self.name = registry_name

    def register_class(self,
                       name: str,
                       class_type: Type[RegisteredType]
                       ):
        """
        Registers a class in the registry by name, and implementation

        Args:
            name (str): The name under which to register the class.
            class_type (Type[T]): The class type to be registered,
                                  constrained by the generic type T.
                                  This class must have a constructor with fully
                                  expressed type hints.
        """

        # Validate
        if not isinstance(class_type, type):
            raise TypeError("class_type must be a class, not an instance.")
        self.interface_validation(class_type)

        # Everything passed validation. Store
        self.registry[name] = ImplementationBuilder(class_type)

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

    def build(self, name: str = "Default", **params: Any) -> RegisteredType:
        """
        Instantiates a registered class using the provided parameters. If a parameter is missing
        and marked as optional, it will be set to `None`. If the parameter requires indirection
        through another registered builder, it will be built automatically if the implementation
        is provided

        :param name: The name of the implementation to build
        :param **params: The various pieces that can be used while building.
        :return: A setup torch layer.
        :raises KeyError: If the implementation was never registered
        :raises RuntimeError: If builder indirection fails
        :raises TypeError: If params have wrong type.
        """
        if name not in self.registry:
            msg = f"""
            Class of name '{name}' is not registered, and thus cannot be 
            built. Current options are:
            
            {self.registry.keys()}
            """
            msg = textwrap.dedent(msg)
            raise KeyError(msg)

        implementation_builder = self.registry[name]
        required_params = implementation_builder.constructor_types

        constructor_args = {}

        # Perform standardization. Handle any builder indirection or
        # optional cases.
        for param_name, param_type in required_params.items():

            # If dealing with something marked as optional, it may be missing. That is
            # okay. It may also be an indirection marked as default. That is also okay.
            if param_name not in params and param_name not in self.indirections:
                if self.is_optional_type(param_type):
                    # Optional means none. We can insert that
                    constructor_args[param_name] = None
                else:
                    # Not recoverable.
                    msg = f"""
                    Missing required parameter '{param_name}' for class '{name}' on
                    registry named '{self.name}'
                    """
                    msg = textwrap.dedent(msg)
                    raise ValueError(msg)

            # Sometimes, we might be dealing with builder indirection. When
            # it is detected, we can run the indirection.
            elif param_name in self.indirections:
                subbuilder = self.indirections[param_name]
                lookup_object_maybe_is_instance = "Default" if param_name not in params else params[param_name]
                if not isinstance(lookup_object_maybe_is_instance, subbuilder.layer_abstract_type):
                    build_directive = lookup_object_maybe_is_instance
                    try:
                        # It was not an instance. Try running
                        constructor_args[param_name] = subbuilder.build(build_directive, **params)
                    except Exception as e:
                        raise RuntimeError(f"Error while building with indirection for {param_name}") from e
                else:
                    constructor_args[param_name] = params[param_name]

            # It was present, not a builder. Just insert
            else:
                constructor_args[param_name] = params[param_name]

        # Instantiate the class with the collected arguments
        return implementation_builder(**constructor_args)

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



