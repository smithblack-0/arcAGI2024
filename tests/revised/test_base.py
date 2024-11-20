import copy
import unittest
import os
import tempfile
import torch
import torch.nn as nn
from typing import Any, Tuple
from src.main.arcAGI2024.base import (PytreeState,
                                      TensorTree,
                                      parallel_pytree_map,
                                      SavableConfig,
                                      GradientSubstitutionEndpoint)
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


class MyPytreeState(PytreeState):
    """
    A concrete implementation of SavableState for testing.
    This class holds a tensor and demonstrates how to save and load state.
    """
    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor

    def save_state(self) -> Tuple[TensorTree, None]:
        # Simply return the tensor itself as the state
        return self.tensor, None

    def load_state(self, pytree: TensorTree, bypass: None) -> 'MyPytreeState':
        # Restore from the given pytree, which should be a tensor
        self.tensor = pytree
        return self
    def __eq__(self, other: Any) -> bool:
        # Equality check to facilitate testing
        if not isinstance(other, MyPytreeState):
            return False
        return torch.equal(self.tensor, other.tensor)


class TestParallelPytreeMapWithSavableState(unittest.TestCase):
    """
    Unit tests for parallel_pytree_map function with support for SavableState.
    """

    def test_parallel_pytree_map_with_savable_state(self):
        """
        Test parallel_pytree_map with nested structures containing SavableState instances.
        """
        # Create test data: two SavableState objects
        tensor1 = torch.tensor([1.0, 2.0, 3.0])
        tensor2 = torch.tensor([4.0, 5.0, 6.0])
        state1 = MyPytreeState(tensor1)
        state2 = MyPytreeState(tensor2)

        # Function to add two tensors
        def add_tensors(x, y):
            if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
                return x + y
            return None

        # Apply parallel_pytree_map with SavableState instances
        result = parallel_pytree_map(add_tensors, state1, state2)

        # Verify that the result is a tuple of SavableState objects
        self.assertIsInstance(result, MyPytreeState)
        self.assertTrue(torch.equal(result.tensor, tensor1 + tensor2))

    def test_parallel_pytree_map_with_mixed_structures(self):
        """
        Test parallel_pytree_map with mixed nested structures containing lists, dicts, and SavableState.
        """
        # Create test data
        tensor1 = torch.tensor([1.0, 2.0, 3.0])
        tensor2 = torch.tensor([4.0, 5.0, 6.0])
        state1 = MyPytreeState(tensor1)
        state2 = MyPytreeState(tensor2)

        nested_structure1 = {'a': [copy.deepcopy(state1), torch.tensor([7.0, 8.0])], 'b': (torch.tensor([9.0]), state1)}
        nested_structure2 = {'a': [copy.deepcopy(state2), torch.tensor([1.0, 2.0])], 'b': (torch.tensor([3.0]), state2)}

        # Function to add two tensors
        def add_tensors(x, y):
            if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
                return x + y
            return None

        # Apply parallel_pytree_map with mixed structures
        result = parallel_pytree_map(add_tensors, nested_structure1, nested_structure2)

        # Verify the result structure matches the original nested structures
        self.assertIsInstance(result, dict)
        self.assertIn('a', result)
        self.assertIn('b', result)
        self.assertIsInstance(result['a'], list)
        self.assertIsInstance(result['b'], tuple)

        print(result["a"][0])

        # Verify the SavableState instances in the result
        expected_tensor = tensor1 + tensor2
        self.assertIsInstance(result['a'][0], MyPytreeState)
        self.assertTrue(torch.equal(result['a'][0].tensor, expected_tensor))

        # Verify the remaining tensor results
        self.assertTrue(torch.equal(result['a'][1], torch.tensor([8.0, 10.0])))
        self.assertTrue(torch.equal(result['b'][0], torch.tensor([12.0])))
        self.assertTrue(torch.equal(result['b'][1].tensor, expected_tensor))

class TestGradientSubstitutionEndpoint(unittest.TestCase):
    def test_gradient_substitution(self):
        # Create an input tensor with requires_grad=True
        input_tensor = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

        # Define desired gradients
        desired_grads = torch.tensor([0.1, 0.2, 0.3])

        # Use GradientSubstitutionEndpoint in the computation graph
        output = GradientSubstitutionEndpoint.apply(input_tensor, desired_grads)

        # Combine output with a dummy loss to trigger backpropagation
        loss = output
        loss.backward()

        # Check that the gradients of input_tensor match the desired gradients
        self.assertTrue(torch.allclose(input_tensor.grad, desired_grads),
                        f"Expected gradients: {desired_grads}, but got: {input_tensor.grad}")


# Assume the SavableConfig class is already imported as per your code.

# Helper classes and functions defined at the top level

# Basic types for testing functions
def custom_activation(x):
    return torch.relu(x) + 1

# For functions that cannot be pickled directly (e.g., lambdas), we can define them at the module level
def add_one(x):
    return x + 1

# Custom class for testing pickle serialization
class CustomFunction:
    def __call__(self, x):
        return x * 2

    def __eq__(self, other):
        return isinstance(other, CustomFunction)

# Dataclasses for testing

@dataclass
class BasicConfig(SavableConfig):
    int_value: int
    float_value: float
    str_value: str
    bool_value: bool
    nested_list: List[Any]
    nested_dict: Dict[str, Any]
    nested_tuple: Tuple[Any, ...]

@dataclass
class NestedConfig(SavableConfig):
    name: str
    configs: List[BasicConfig]
    sub_config: BasicConfig

@dataclass
class LayerConfig(SavableConfig):
    layer: nn.Module
    shared_layer: nn.Module
    layers_list: List[nn.Module]

@dataclass
class FunctionConfig(SavableConfig):
    activation_function: Any  # Functions are of type 'Any' since they are callable
    custom_function: Any


class CustomLayer(nn.Module):
    def __init__(self, activation):
        super().__init__()
        self.linear = nn.Linear(10, 10)
        self.activation = activation

    def forward(self, x):
        return self.activation(self.linear(x))

# Begin test suite
class TestSavableConfig(unittest.TestCase):

    def test_basic_operation(self):
        # Create a BasicConfig with nested pytree structures
        basic_config = BasicConfig(
            int_value=42,
            float_value=3.14,
            str_value="Test string",
            bool_value=True,
            nested_list=[1, [2, [3, [4, [5]]]]],
            nested_dict={'level1': {'level2': {'level3': 'deep_value'}}},
            nested_tuple=(1, (2, (3, (4, (5,)))))
        )

        # Serialize and deserialize
        serialized = basic_config.serialize()
        deserialized_config = BasicConfig.deserialize(serialized)

        # Assert that the deserialized values match the original
        self.assertEqual(basic_config.int_value, deserialized_config.int_value)
        self.assertEqual(basic_config.float_value, deserialized_config.float_value)
        self.assertEqual(basic_config.str_value, deserialized_config.str_value)
        self.assertEqual(basic_config.bool_value, deserialized_config.bool_value)
        self.assertEqual(basic_config.nested_list, deserialized_config.nested_list)
        self.assertEqual(basic_config.nested_dict, deserialized_config.nested_dict)
        self.assertEqual(basic_config.nested_tuple, deserialized_config.nested_tuple)

    def test_nested_config(self):
        # Create a BasicConfig instance
        basic_config_1 = BasicConfig(
            int_value=1,
            float_value=1.1,
            str_value="Config 1",
            bool_value=False,
            nested_list=[1, 2, 3],
            nested_dict={'a': 1},
            nested_tuple=(1, 2)
        )

        basic_config_2 = BasicConfig(
            int_value=2,
            float_value=2.2,
            str_value="Config 2",
            bool_value=True,
            nested_list=[4, 5, 6],
            nested_dict={'b': 2},
            nested_tuple=(3, 4)
        )

        # Create a NestedConfig with a list of BasicConfigs
        nested_config = NestedConfig(
            name="MasterConfig",
            configs=[basic_config_1, basic_config_2],
            sub_config=basic_config_1
        )

        # Serialize and deserialize
        serialized = nested_config.serialize()
        deserialized_config = NestedConfig.deserialize(serialized)

        # Assert that the deserialized values match the original
        self.assertEqual(nested_config.name, deserialized_config.name)
        self.assertEqual(len(nested_config.configs), len(deserialized_config.configs))

        for orig_config, deserialized_sub_config in zip(nested_config.configs, deserialized_config.configs):
            self.assertEqual(orig_config, deserialized_sub_config)

        # Check that the sub_config is correctly deserialized
        self.assertEqual(nested_config.sub_config, deserialized_config.sub_config)

    def test_layers(self):
        # Create layers
        layer_a = nn.Linear(10, 10)
        layer_b = nn.Linear(10, 10)
        shared_layer = nn.Linear(10, 10)

        # Create a LayerConfig
        layer_config = LayerConfig(
            layer=layer_a,
            shared_layer=shared_layer,
            layers_list=[layer_a, layer_b, shared_layer]
        )

        # Serialize and deserialize
        serialized = layer_config.serialize()
        deserialized_config = LayerConfig.deserialize(serialized)

        # Assert that the layers are correctly deserialized
        self.assertIsInstance(deserialized_config.layer, nn.Module)
        self.assertIsInstance(deserialized_config.shared_layer, nn.Module)

        # Check that parameters are the same
        for orig_param, deserialized_param in zip(layer_config.layer.parameters(), deserialized_config.layer.parameters()):
            self.assertTrue(torch.equal(orig_param, deserialized_param))

        # Check that shared layers remain shared
        self.assertIs(deserialized_config.shared_layer, deserialized_config.layers_list[2])

        # Check that layers in layers_list are correctly deserialized
        self.assertIsInstance(deserialized_config.layers_list[0], nn.Module)
        self.assertIsInstance(deserialized_config.layers_list[1], nn.Module)
        self.assertIsInstance(deserialized_config.layers_list[2], nn.Module)

        # Ensure that the shared layer in layers_list is the same as shared_layer
        self.assertIs(deserialized_config.shared_layer, deserialized_config.layers_list[2])

    def test_functions(self):
        # Create a FunctionConfig with picklable functions
        function_config = FunctionConfig(
            activation_function=custom_activation,
            custom_function=CustomFunction()
        )

        # Serialize and deserialize
        serialized = function_config.serialize()
        deserialized_config = FunctionConfig.deserialize(serialized)

        # Assert that the functions are correctly deserialized
        self.assertEqual(
            function_config.activation_function(torch.tensor(-1.0)),
            deserialized_config.activation_function(torch.tensor(-1.0))
        )

        self.assertEqual(
            function_config.custom_function(10),
            deserialized_config.custom_function(10)
        )

    def test_pickle_functions(self):
        # Test with a standard function
        function_config = FunctionConfig(
            activation_function=add_one,
            custom_function=None
        )

        # Serialize and deserialize
        serialized = function_config.serialize()
        deserialized_config = FunctionConfig.deserialize(serialized)

        # Assert that the function works as expected after deserialization
        self.assertEqual(
            function_config.activation_function(5),
            deserialized_config.activation_function(5)
        )

    def test_lambda_functions(self):
        # Note: Lambda functions cannot be pickled; this test should confirm that
        function_config = FunctionConfig(
            activation_function=lambda x: x + 2,
            custom_function=None
        )

        # Attempt to serialize and expect an exception
        with self.assertRaises(Exception):
            function_config.serialize()

    def test_function_in_nested_config(self):
        # Create a BasicConfig instance
        basic_config = BasicConfig(
            int_value=1,
            float_value=1.1,
            str_value="Config",
            bool_value=False,
            nested_list=[1, 2, 3],
            nested_dict={'a': 1},
            nested_tuple=(1, 2)
        )

        # Create a NestedConfig that includes a function
        nested_config = NestedConfig(
            name="ConfigWithFunction",
            configs=[basic_config],
            sub_config=basic_config
        )

        # Add a function to the nested_dict
        nested_config.sub_config.nested_dict['func'] = custom_activation

        # Serialize and deserialize
        serialized = nested_config.serialize()
        deserialized_config = NestedConfig.deserialize(serialized)

        # Assert that the function works as expected after deserialization
        self.assertEqual(
            deserialized_config.sub_config.nested_dict['func'](torch.tensor(-1.0)),
            custom_activation(torch.tensor(-1.0))
        )

    def test_shared_functions(self):
        # Create functions
        func_a = custom_activation
        func_b = custom_activation  # Reference to the same function

        # Create a FunctionConfig
        function_config = FunctionConfig(
            activation_function=func_a,
            custom_function=func_b
        )

        # Serialize and deserialize
        serialized = function_config.serialize()
        deserialized_config = FunctionConfig.deserialize(serialized)

        # Check that the functions are the same (shared)
        self.assertIs(deserialized_config.activation_function, deserialized_config.custom_function)

    def test_function_in_layers(self):
        # Create a layer with a custom activation function


        layer = CustomLayer(activation=custom_activation)

        # Create a LayerConfig
        layer_config = LayerConfig(
            layer=layer,
            shared_layer=None,
            layers_list=[]
        )

        # Serialize and deserialize
        serialized = layer_config.serialize()
        deserialized_config = LayerConfig.deserialize(serialized)

        # Assert that the layer works as expected after deserialization
        input_tensor = torch.randn(1, 10)
        orig_output = layer(input_tensor)
        deserialized_output = deserialized_config.layer(input_tensor)

        self.assertTrue(torch.equal(orig_output, deserialized_output))

    def test_nested_functions_in_pytree(self):
        # Create a complex pytree with functions
        pytree = {
            'level1': [
                custom_activation,
                {'level2': CustomFunction()}
            ]
        }

        # Create a BasicConfig
        basic_config = BasicConfig(
            int_value=0,
            float_value=0.0,
            str_value="",
            bool_value=False,
            nested_list=[],
            nested_dict=pytree,
            nested_tuple=()
        )

        # Serialize and deserialize
        serialized = basic_config.serialize()
        deserialized_config = BasicConfig.deserialize(serialized)

        # Assert that the functions work as expected
        self.assertEqual(
            deserialized_config.nested_dict['level1'][0](torch.tensor(-1.0)),
            custom_activation(torch.tensor(-1.0))
        )


    def test_function_with_closure(self):
        # Functions with closures cannot be pickled directly
        def make_multiplier(n):
            def multiplier(x):
                return x * n
            return multiplier

        multiplier_by_3 = make_multiplier(3)

        function_config = FunctionConfig(
            activation_function=multiplier_by_3,
            custom_function=None
        )

        # Attempt to serialize and expect an exception
        with self.assertRaises(Exception):
            function_config.serialize()
