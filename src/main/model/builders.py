from typing import Type, Dict, Any, Optional, Union
from typeguard import check_type


class RegistryBuilder:
    """
    A class to register and manage the instantiation of other classes.

    This class allows the dynamic registration of classes by name,
    along with the expected constructor parameters. It supports generic
    types and handles optional parameters seamlessly.

    Attributes:
        registry (Dict[str, Dict[str, Any]]): A dictionary that stores the
            registered class information, including the class type and
            expected constructor parameters.
    """

    def __init__(self):
        """
        Initializes a new RegistryBuilder with an empty registry.
        """
        self.registry: Dict[str, Dict[str, Any]] = {}

    def register_class(self, name: str, class_type: Type, **constructor_params: Optional[Any]) -> None:
        """
        Registers a class in the registry by its name, type, and constructor parameters.

        Args:
            name (str): The name under which to register the class.
            class_type (Type): The class type to be registered.
            constructor_params (Optional[Any]): Constructor parameters with their expected types.
        """
        self.registry[name] = {
            "type": class_type,
            "params": constructor_params
        }

    def build(self, name: str, params: Dict[str, Any]) -> Any:
        """
        Instantiates a registered class using the provided parameters.

        Args:
            name (str): The name of the registered class to instantiate.
            params (Dict[str, Any]): A dictionary containing the parameters to pass
                to the constructor of the class.

        Returns:
            Any: An instance of the registered class.

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
                    check_type(param_name, value, param_type)
                    constructor_args[param_name] = value
                except TypeError as e:
                    raise ValueError(f"Parameter '{param_name}' has an invalid type: {e}")

        # Instantiate the class with the collected arguments
        return class_type(**constructor_args)

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
