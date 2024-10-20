import inspect
import textwrap

import typeguard

from typing import (Type, Dict, Any, Optional,
                    Generic, TypeVar, Union, get_type_hints, get_origin, get_args, Callable)

RegisteredType = TypeVar('RegisteredType')


def get_constructor_type_hints(cls: Type[Any]) -> Dict[str, Any]:
    """
    Gets the type hints from the constructor (__init__) of a provided class,
    excluding the first parameter (usually 'self').

    Args:
        cls (Type[Any]): The class whose constructor type hints are to be extracted.

    Returns:
        Dict[str, Any]: A dictionary mapping constructor parameter names to their type hints.
    """
    # Get the constructor (__init__) of the class
    constructor = getattr(cls, '__init__', None)
    if constructor is None:
        raise ValueError(f"Class {cls.__name__} does not have a constructor (__init__) method.")

    # Get the list of parameter names from the constructor signature
    # Also, get the list of parameters whose type hints will be required
    # in some way or another. We exclude the first one, which is always
    # 'self' on a constructor

    constructor_params = inspect.signature(constructor).parameters
    required_typehints = list(constructor_params.keys())[1:]

    # Build a datastructure matching parameter names to their types.
    constructor_types = {}
    for name in required_typehints:
        if constructor_params[name].annotation == inspect.Parameter.empty:
            raise ValueError(f"Parameter of name {name} did not have a type annotation")
        constructor_types[name] = constructor_params[name].annotation
    return constructor_types


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
                      proposed_type_hint: Any)->bool:
    """
    Checks if type hints are the same
    :param required_type_hint: The type hint of the required parameter.
    :param proposed_type_hint: The type hint of the proposed parameter.
    :return: The result of the check.
    """

    # Check if both type hints are the same concrete class type (e.g., int, str, MyClass)
    if required_type_hint == proposed_type_hint:
        return True

    # Get the origin of both type hints (e.g., List, Tuple, Dict)
    origin1 = get_origin(required_type_hint)
    origin2 = get_origin(proposed_type_hint)

    # If the origins do not match, the type hints are different
    if origin1 != origin2:
        return False

    # Get the type arguments (e.g., the type within List[int])
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

class RegistryBuilder(Generic[RegisteredType]):
    """
    A generic class to register and manage the instantiation of other classes.

    This class allows the dynamic registration of classes by name,
    along with the expected constructor parameters. It supports generic
    types and handles optional parameters seamlessly.

    Attributes:
        registry (Dict[str, Dict[str, Any]]): A dictionary that stores the
            registered class information, including the class type and
            expected constructor parameters.
    """

    def __init__(self,
                 **parameters: Dict[str, Any]
                 ):
        """
        Initialize the registry builder. If desired, the user
        can provide certain interface parameters that the attached
        implementations MUST provide to be bound to.

        parameters: A name, type hint matching of parameters to the expected
                    types of those parameters.
        """
        # Do some quick type checking
        for name, type in parameters.items():
            if not is_type_hint(type):
                msg = """
                Registry should have been initialized with required defaults and 
                their type hints. However, for '{name}', did not receive type hint.
                """
                msg = textwrap.dedent(msg)
                raise TypeError(msg)

        # Setup and otherwise store
        self.required_parameters = parameters
        self.registry: Dict[str, Dict[str, Any]] = {}

    def register_class(self, name: str,
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
        schema = get_constructor_type_hints(class_type)

        # Loop over any required types. Verify existence. Verify they
        # have the same type hints.
        for required_name, required_type in self.required_parameters.items():
            if required_name not in schema:
                msg = f"""
                The registry expects registered classes to accept a constructor
                keyword of name {required_name}. That was not found in class
                of name '{class_type.__name__}'
                """
                msg = textwrap.dedent(msg)
                raise ValueError(msg)
            if not is_same_type_hint(required_type, schema[required_name]):
                msg = f"""
                The registry expects the registered class to have a parameter typehint
                for '{required_name}' of '{required_type}'. However, class named {class_type.__name__}
                implements it as {schema[required_name]}
                """
                msg = textwrap.dedent(msg)
                raise ValueError(msg)

        # Everything passed validation. Store
        self.registry[name] = {
            "type": class_type,
            "params": schema
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

    def build(self, name: str, params: Dict[str, Any]) -> RegisteredType:
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
                if self.is_optional_type(param_type):
                    constructor_args[param_name] = None
                else:
                    raise ValueError(f"Missing required parameter '{param_name}' for class '{name}'.")
            else:
                value = params[param_name]
                # Use typeguard to check the type of the parameter
                try:
                    typeguard.check_type(value, param_type)
                    constructor_args[param_name] = value
                except TypeError as e:
                    raise ValueError(f"Parameter '{param_name}' has an invalid type: {e}") from e

        # Instantiate the class with the collected arguments
        return class_type(**constructor_args)

    def register(self,
                 name: str
                 )->Callable[[Type[RegisteredType]], Type[RegisteredType]]:
        """
        Decorator register to use with registry builder
        :param name: The name of the feature to register
        :return: The registered class
        """
        def register(class_type: Type[RegisteredType]) -> Type[RegisteredType]:
            self.register_class(name, class_type)
            return class_type
        return register
