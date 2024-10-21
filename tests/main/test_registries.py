import unittest
import torch
from torch import nn
from typing import Optional, List, Dict, Any, Tuple, Union
from src.main.model.registry import (is_type_hint, is_same_type_hint, is_sub_type_hint,
                                     get_constructor_type_hints, TorchLayerRegistry)


class TestHelperFunctions(unittest.TestCase):
    """
    Test some very important registry helper functions
    responsible in the module for handling typing.
    """
    def test_is_type_hint_valid(self):
        # Test valid type hints
        self.assertTrue(is_type_hint(int))
        self.assertTrue(is_type_hint(Optional[int]))
        self.assertTrue(is_type_hint(List[int]))
        self.assertTrue(is_type_hint(Dict[str, int]))

    def test_is_type_hint_invalid(self):
        # Test invalid type hints
        self.assertFalse(is_type_hint(123))
        self.assertFalse(is_type_hint("hello"))
        self.assertFalse(is_type_hint([1, 2, 3]))

    def test_is_same_type_hint(self):
        # Test same type hints
        self.assertTrue(is_same_type_hint(int, int))
        self.assertTrue(is_same_type_hint(Optional[int], Optional[int]))
        self.assertTrue(is_same_type_hint(List[int], List[int]))
        self.assertTrue(is_same_type_hint(Dict[str, int], Dict[str, int]))

        # Test different type hints
        self.assertFalse(is_same_type_hint(int, str))
        self.assertFalse(is_same_type_hint(Optional[int], Optional[str]))
        self.assertFalse(is_same_type_hint(List[int], List[str]))
        self.assertFalse(is_same_type_hint(Dict[str, int], Dict[int, str]))

    def test_get_constructor_type_hints(self):
        class MyClass:
            def __init__(self, a: int, b: Optional[str] = None):
                self.a = a
                self.b = b

        expected_hints = {"a": int, "b": Optional[str]}
        self.assertEqual(get_constructor_type_hints(MyClass), expected_hints)
    def test_is_sub_type_hint(self):
        # Test Any compatibility (Any matches anything)
        self.assertTrue(is_sub_type_hint(Any, int))
        self.assertTrue(is_sub_type_hint(Any, List[int]))
        self.assertTrue(is_sub_type_hint(Any, Dict[str, int]))

        # Test exact matches
        self.assertTrue(is_sub_type_hint(int, int))
        self.assertTrue(is_sub_type_hint(Optional[int], Optional[int]))
        self.assertTrue(is_sub_type_hint(List[int], List[int]))
        self.assertTrue(is_sub_type_hint(Dict[str, int], Dict[str, int]))

        # Test unions (Optional is Union[T, None])
        self.assertTrue(is_sub_type_hint(Union[int, str], int))  # int is part of Union[int, str]
        self.assertTrue(is_sub_type_hint(Union[int, str], str))  # str is part of Union[int, str]
        self.assertFalse(is_sub_type_hint(Union[int, str], float))  # float is not part of Union[int, str]

        # Test Optional (which is Union[T, None])
        self.assertTrue(is_sub_type_hint(Optional[int], None))  # None is valid for Optional[int]
        self.assertTrue(is_sub_type_hint(Optional[int], int))  # int is valid for Optional[int]
        self.assertFalse(is_sub_type_hint(Optional[int], str))  # str is not valid for Optional[int]

        # Test generic types (List, Dict)
        self.assertTrue(is_sub_type_hint(List[Any], List[int]))  # List[Any] is broader than List[int]
        self.assertFalse(is_sub_type_hint(List[int], List[Any]))  # List[int] is not broader than List[Any]
        self.assertTrue(is_sub_type_hint(Dict[str, Any], Dict[str, int]))  # Dict[str, Any] is broader than Dict[str, int]
        self.assertFalse(is_sub_type_hint(Dict[str, int], Dict[str, Any]))  # Dict[str, int] is not broader than Dict[str, Any]

        # Test nested generics with unions
        self.assertTrue(is_sub_type_hint(List[Union[int, str]], List[int]))  # List[int] is compatible with List[Union[int, str]]
        self.assertFalse(is_sub_type_hint(List[int], List[Union[int, str]]))  # List[Union[int, str]] allows more than List[int]

        # Test tuple with ellipsis (variable length)
        self.assertTrue(is_sub_type_hint(Tuple[int, ...], Tuple[int, int, int]))  # Tuple[int, ...] is compatible with Tuple[int, int, int]
        self.assertFalse(is_sub_type_hint(Tuple[int, ...], Tuple[str, str]))  # Tuple[int, ...] is not compatible with Tuple[str, str]
        self.assertTrue(is_sub_type_hint(Tuple[int, str], Tuple[int, str]))  # Exact tuple match

        # Test incompatible tuple lengths
        self.assertFalse(is_sub_type_hint(Tuple[int, str], Tuple[int]))  # Tuple[int, str] is not compatible with Tuple[int]
        self.assertFalse(is_sub_type_hint(Tuple[int], Tuple[int, int]))  # Tuple[int] is not compatible with Tuple[int, int]


class TestRegistryBuilder(unittest.TestCase):
    """
    Test the registry, and associated build, method.
    """
    def test_basic_class(self):
        """Test registry of a basic class."""


        class MyClass:
            def __init__(self, a: int, b: Optional[str]):
                self.a = a
                self.b = b

            def forward(self, x: torch.Tensor)-> torch.Tensor:
                pass

        # Registering MyClass in RegistryBuilder[MyClass]
        registry = TorchLayerRegistry[MyClass]("test_classes", MyClass)
        registry.register_class("my_class", MyClass)
        self.assertIn("my_class", registry.registry)

        # Instance it

        instance = registry.build("my_class", a= 3, b="potato")

    def test_constructor_validation(self):
        """Test we catch when registering insane requirements or classes"""

        # Try to setup a registry that does not implement forward
        class MyClass:
            def __init__(self, a: int, b: Optional[str]):
                self.a = a
                self.b = b

        with self.assertRaises(ValueError):
            registry = TorchLayerRegistry[MyClass]("test_registry", MyClass)

        # Try to setup a class with missing constructor type hints

        class MyClass:
            def __init__(self, a: int, b):
                self.a = a
                self.b = b

            def forward(self)->torch.Tensor:
                pass

        with self.assertRaises(TypeError):
            registry = TorchLayerRegistry[MyClass]("test_registry", MyClass)

        # Try to setup a class with missing forward type hints
        class MyClass:
            def __init__(self, a: int, b: int):
                self.a = a
                self.b = b

            def forward(self, x)->torch.Tensor:
                pass

        with self.assertRaises(TypeError):
            registry = TorchLayerRegistry[MyClass]("test_registry", MyClass)

        # Try to setup a class that would work, but with insane registry
        # indirection

        class MyClass:
            def __init__(self, a: int, b: Optional[str]):
                self.a = a
                self.b = b

            def forward(self, x: torch.Tensor)-> torch.Tensor:
                pass

        with self.assertRaises(TypeError):
            registry = TorchLayerRegistry[MyClass]("test_registry", MyClass, a="apple")

        # Try to setup a class that would work, but with a registry redirection that it cannot
        # consume

        class MyClass:
            def __init__(self, a: int, b: Optional[str]):
                self.a = a
                self.b = b

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                pass

        with self.assertRaises(TypeError):
            irrelevant_registry = TorchLayerRegistry[MyClass]("test_registry", MyClass)
            registry = TorchLayerRegistry[MyClass]("test_registry", MyClass, c=irrelevant_registry)

    def test_register_validation(self):
        """Test we catch when you are attempting to register something insane or nonmatching"""
        class MyClass:
            def __init__(self, a: int, b: Optional[str]):
                self.a = a
                self.b = b

        # Try to register something that is missing a required keyword
        with self.assertRaises(TypeError):
            registry = TorchLayerRegistry[MyClass](c=int)
            registry.register_class("my_class", MyClass)

        # Try to register to a registry with different required typing

        with self.assertRaises(TypeError):
            registry = TorchLayerRegistry[MyClass](a=float, b=str)
            registry.register_class("my_class", MyClass)

    def test_build_details(self):
        """Test the builder mechanism, in its various modes of operation"""
        class MyClass:
            def __init__(self, a: int, b: Optional[str]):
                self.a = a
                self.b = b

        # Test building with an optional, unprovided, parameter
        registry = TorchLayerRegistry[MyClass]()
        registry.register_class("my_class", MyClass)
        instance = registry.build("my_class", a=3)

        # Test building with other, unused, parameters
        instance = registry.build("my_class", a=3, c="potato")

        # Test when we attempt to build with something that is non optional, and
        # it is not provided, we throw
        with self.assertRaises(ValueError):
            registry.build("my_class")

        # Test when we attempt to build something, and it is the wrong
        # type, we throw

        with self.assertRaises(TypeError):
            registry.build("my_class", a = "potato")
    def test_origin(self):
        class MyClass:
            def __init__(self, a: int, b: Optional[str]):
                self.a = a
                self.b = b

        # Test building with an optional, unprovided, parameter
        registry = TorchLayerRegistry[MyClass]("test_registry")
        print(registry.layer_abstract_type)