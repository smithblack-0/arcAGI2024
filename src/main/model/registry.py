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

def get_constructor_spec(abstract_cls: Type[nn.Module]
                         )->Tuple[inspect.Signature, Dict[str, Any]]:
    """
    Gets constructor data. This includes the signature, and the typing on each parameter
    :return:
        - Signature
        - Typing for each parameter
    """
    signature = inspect.signature(abstract_cls.__init__)
    params, _ = get_signature_type_hints("__init__", abstract_cls.__name__,
                                         signature, expect_return=False)
    return signature, params

def get_forward_spec(abstract_cls: Type[nn.Module]
                     )->Tuple[inspect.Signature, Dict[str, Any], Any]:
    """
    Get the forward spec, including the return
    :param abstract_cls: The class to work with
    :return:
        - Signature
        - Typing for each parameter
        - Return typing
    """
    if abstract_cls.forward.__name__ != "forward":
        raise TypeError(f"Class '{abstract_cls.__name__}' still has default forward method.")

    signature = inspect.signature(abstract_cls.forward)
    params, return_type = get_signature_type_hints("forward", abstract_cls.__name__,
                                         signature, expect_return=True)
    return signature, params, return_type
class AbstractClassSchemaEnforcer:
    """
    `AbstractClassSchemaEnforcer` ensures that any implementation of an abstract
    class continues to match its interface. This class is used to verify that the
    constructor and forward method of an implementation match the expected typing
    from the abstract class.

    This ensures compatibility when swapping different layers into the same registry.
    """
    @property
    def constructor_schema(self)->Dict[str, Any]:
        return self.constructor_spec[1]

    @property
    def forward_schema(self)->Dict[str, Any]:
        return self.forward_spec[1]

    @property
    def return_schema(self)->Any:
        return self.forward_spec[2]
    def __init__(self,
                 cls_abstract: Type[nn.Module],
                 ):

        # Inspect it.
        self.name = cls_abstract.__name__
        self.cls_interface = cls_abstract
        self.constructor_spec = get_constructor_spec(cls_abstract)
        self.forward_spec = get_forward_spec(cls_abstract)

    def validate_constructor_specification(self, cls_implementation: Type[nn.Module]):
        """
        Validates that the class constructor specification is sane. We throw if not
        :param cls_implementation: The implementation to validate
        :raises TypeError: If various issues happen
        """
        # Fetch the data to compare.
        constructor_signature, constructor_params = get_constructor_spec(cls_implementation)
        expected_signature, expected_params = self.constructor_spec

        # Constructors are considered compatible if and only if they
        # implement the entire expected signature, however there can be
        # more required features in the implementation constructor than the
        # interface one

        for name, type in expected_params.items():
            if name not in constructor_params:
                raise TypeError(f"Implementation constructor missing parameter {name}")
            if not is_sub_type_hint(type, constructor_params[name]):
                msg = f"""
                Types did not match for parameter '{name}'
                AbstractInterface: '{type}'
                Implementation: {constructor_params[name]}
                """
                msg = textwrap.dedent(msg)
                raise TypeError(msg)

    def validate_forward_specification(self, cls_implementation: Type[nn.Module]):
        """
        Validates the forward implementation is sane.
        :param cls_implementation:
        :return:
        """
        # Get data to compare
        signature, param_types, return_type = get_forward_spec(cls_implementation)
        expected_signature, expected_params, expected_return = self.forward_spec

        # If the implementation differs, throw.
        if param_types.keys() != expected_params.keys():
            msg = f"""
            The forward interface expected to see, in order, features: 

            {expected_params.keys()}

            However, actually implemented was:

            {param_types.keys()}
            """
            msg = textwrap.dedent(msg)
            raise TypeError(msg)

        # Check the actual types match
        for name in param_types.keys():
            if not is_sub_type_hint(expected_params[name], param_types[name]):
                msg = f"""
                Issue found at parameter {name}
                The forward interface expected to see a type of: {expected_params[name]}.
                However, actual implementation was: {param_types[name]}.
                These are not compatible.
                """
                msg = textwrap.dedent(msg)
                raise TypeError(msg)

        # Check the types of the return match
        if not is_sub_type_hint(expected_return, return_type):
            msg = f"""
            Forward interface's return different from implementation.

            The forward interface was expected to match type {expected_return}.
            However, actually observed: {return_type}.
            """
            msg = textwrap.dedent(msg)
            raise TypeError(msg)

    def __call__(self, cls_implentation: Type[nn.Module]):
        """
        Perform validation. Includes inspection actions
        :param cls_implentation: The class to verify matches the schema
        :raises TypeError: If they do not match. Various error messages
        """

        if not issubclass(cls_implentation, self.cls_interface):
            msg = f"""
            Issue encountered. Attempted to register a class that was not a 
            subclass of the interface. 
            Class: {cls_implentation}
            Interface: {self.cls_interface}
            """
            msg = textwrap.dedent(msg)
            raise TypeError(msg)

        self.validate_constructor_specification(cls_implentation)
        self.validate_forward_specification(cls_implentation)

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
        return self.constructor_spec[1]

    @property
    def constructor_signature(self)->inspect.Signature:
        return self.constructor_spec[0]

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

        signature, param_types = self.constructor_spec

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
    such as feedforward layers, self-long_term_memories layers, and cross-long_term_memories layers.
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

    linear_registry = TorchLayerRegistry[AbstractLinear](AbstractLinear)

    # Stores torch's linear
    linear_registry.register_class("normal",nn.Linear)

    # Builds a copy
    instance = linear_registry.build("normal", in_features=3, out_features=4, bias=False)
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
    linear = TorchLayerRegistry[AbstractLinear](AbstractLinear)

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

    # Define a builder for an long_term_memories layer and for feedforward
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
    # in order to specify what kind of long_term_memories and feedforward to use

    transformer = transformer_builder.build("transformer",
                                             d_model=128,
                                             d_hidden=512,
                                             heads=4,
                                             Attention="standard_attention"
                                             Feedforward="standard_feedforward"

    """
    @property
    def layer_abstract_type(self)->Type[Any]:
        return self.layer_type

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
                 **registry_indirections: 'TorchLayerRegistry'
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

        # Build the schema enforce

        schema_enforcer = AbstractClassSchemaEnforcer(abstract_layer)


        # Now, go over the registry indirections and make sure they make sense
        for name, registry in registry_indirections.items():
            if not isinstance(registry, TorchLayerRegistry):
                msg= f"""
                Issue on registry indirection named '{name}'
                The indirection value was not a TorchLayerRegistry.
                """
                msg = textwrap.dedent(msg)
                raise TypeError(msg)

            if name not in schema_enforcer.constructor_schema:
                msg = f"""
                Indirection registry of name '{name}' was not found among
                constructor arguments
                """
                msg = textwrap.dedent(msg)
                raise TypeError(msg)
            if not issubclass(schema_enforcer.constructor_schema[name], registry.layer_abstract_type):
                msg = f"""
                Abstract interface and indirection registry named {registry.registry_name} were
                found to be incompatible
                """
                msg = textwrap.dedent(msg)
                raise TypeError(msg)

        # Setup and otherwise store intake details.
        self.layer_type = abstract_layer
        self.interface_validation = AbstractClassSchemaEnforcer(abstract_layer)
        self.registry: Dict[str, ImplementationBuilder] = {}
        self.indirections = registry_indirections

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
            raise TypeError("class_type must be a class, not a instance.")
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

    def build(self, name: str, **params: Dict[str, Any]) -> RegisteredType:
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
            raise KeyError(f"Class '{name}' is not registered.")

        implementation_builder = self.registry[name]
        required_params = implementation_builder.constructor_types

        # Perform standardization. Handle any builder indirection or
        # optional cases.
        constructor_args = {}
        for param_name, param_type in required_params.items():

            # If dealing with something marked as optional, it may be missing. That is
            # okay
            if param_name not in params:
                if self.is_optional_type(param_type):
                    # Optional means none. We can insert that
                    constructor_args[param_name] = None
                else:
                    # Not recoverable.
                    raise ValueError(f"Missing required parameter '{param_name}' for class '{name}'.")

            # Sometimes, we might be dealing with builder indirection. When
            # it is detected, we can run the indirection.
            elif param_name in self.indirections:
                subbuilder = self.indirections[param_name]
                if not isinstance(params[param_name], subbuilder.layer_abstract_type):
                    try:
                        constructor_args[param_name] = subbuilder.build(params[param_name], **params)
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



