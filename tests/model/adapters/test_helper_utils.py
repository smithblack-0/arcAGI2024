import unittest
from typing import Callable
from src.model.adapters.helper_utils import validate_config
# Global flag to show error messages
SHOW_MESSAGES = True

class TestValidateConfig(unittest.TestCase):

    def check_errors(self, func: Callable, *args, **kwargs):
        """
        Helper method to catch and print errors during test execution.
        Re-raises the error so that it can be caught by the unittest framework.
        """
        try:
            func(*args, **kwargs)
        except Exception as e:
            if SHOW_MESSAGES:
                # Get the entire error sequence
                error_stack = [e]
                error = e
                while error.__cause__ is not None:
                    error_stack.append(error.__cause__)
                    error = error.__cause__

                # Display the entire stack
                print("Beginning to display error messages for inspection")
                print(f"Originating error was: {error_stack.pop()}")
                while error_stack:
                    print(f"Induced error was: {error_stack.pop()}")
            raise e

    def test_primitive_type_match(self):
        validate_config(5, int)
        validate_config(3.14, float)
        validate_config("hello", str)
        validate_config(True, bool)

    def test_primitive_type_mismatch(self):
        with self.assertRaises(TypeError):
            self.check_errors(validate_config, 5, float)
        with self.assertRaises(TypeError):
            self.check_errors(validate_config, "hello", int)
        with self.assertRaises(TypeError):
            self.check_errors(validate_config, True, str)

    def test_literal_match(self):
        validate_config(5, 5)
        validate_config("foo", "foo")
        validate_config(True, True)

    def test_literal_mismatch(self):
        with self.assertRaises(TypeError):
            self.check_errors(validate_config, 5, 3)
        with self.assertRaises(TypeError):
            self.check_errors(validate_config, "bar", "foo")
        with self.assertRaises(TypeError):
            self.check_errors(validate_config, False, True)

    def test_dict_type_match(self):
        config = {"embedding_dim": 128, "batch_size": 32}
        type_spec = {"embedding_dim": int, "batch_size": int}
        validate_config(config, type_spec)

    def test_dict_type_mismatch(self):
        config = {"embedding_dim": 128, "batch_size": "32"}
        type_spec = {"embedding_dim": int, "batch_size": int}
        with self.assertRaises(TypeError):
            self.check_errors(validate_config, config, type_spec)

    def test_dict_key_mismatch(self):
        config = {"embedding_dim": 128, "batch_size": 32}
        type_spec = {"embedding_dim": int, "num_layers": int}
        with self.assertRaises(TypeError):
            self.check_errors(validate_config, config, type_spec)

    def test_list_type_match(self):
        config = [1, 2, 3]
        type_spec = [int, int, int]
        validate_config(config, type_spec)

    def test_list_type_mismatch(self):
        config = [1, "2", 3]
        type_spec = [int, int, int]
        with self.assertRaises(TypeError):
            self.check_errors(validate_config, config, type_spec)

    def test_tuple_type_match(self):
        config = (1, "foo", True)
        type_spec = (int, str, bool)
        validate_config(config, type_spec)

    def test_tuple_type_mismatch(self):
        config = (1, 2, 3)
        type_spec = (int, str, bool)
        with self.assertRaises(TypeError):
            self.check_errors(validate_config, config, type_spec)

    def test_generic_list_type_match(self):
        config = [1, 2, 3, 4, 5]
        type_spec = list[int]
        validate_config(config, type_spec)

    def test_generic_list_type_mismatch(self):
        config = [1, "2", 3, 4, 5]
        type_spec = list[int]
        with self.assertRaises(TypeError):
            self.check_errors(validate_config, config, type_spec)

    def test_generic_dict_type_match(self):
        config = {"key1": 1, "key2": 2}
        type_spec = dict[str, int]
        validate_config(config, type_spec)

    def test_generic_dict_type_mismatch(self):
        config = {"key1": 1, "key2": "2"}
        type_spec = dict[str, int]
        with self.assertRaises(TypeError):
            self.check_errors(validate_config, config, type_spec)

    def test_invalid_type_spec(self):
        config = {"key": "value"}
        type_spec = set[str]
        with self.assertRaises(RuntimeError):
            self.check_errors(validate_config, config, type_spec)