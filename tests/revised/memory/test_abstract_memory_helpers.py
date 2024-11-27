# test_core_helpers.py

import unittest
import torch
from src.main.arcAGI2024.memory.base import (
    _compute_erase_factor,
    _compute_write_factor,
    _step_state_forward,
    _step_state_reverse,
    _advance_memory,
    _retard_memory,
    _advance_metrics,
    _retard_metrics,
    MemoryState
)


class TestMemHelperFunctions(unittest.TestCase):
    """
    Test the memory helper functions for stability
    and usefulness
    """
    def setUp(self):
        # Common setup for all tests
        self.batch_size = 2
        self.num_elements = 4
        self.d_model = 3  # Example dimensionality

        # Initialize memory tensors
        self.memory_tensor = torch.randn(self.batch_size, self.num_elements, self.d_model)
        self.update_tensor = torch.randn(self.batch_size, self.num_elements, self.d_model)

        # Initialize probabilities
        self.write_gate = torch.rand(self.batch_size, self.num_elements)
        self.erase_gate = torch.rand(self.batch_size, self.num_elements)*0.99

    def test_forward_reverse_inverses(self):
        """
        Test that going forward, then reverse, actually results in inverses
        """
        batch_mask = torch.tensor([False, False])
        memory = _advance_memory(self.memory_tensor, self.update_tensor, self.write_gate, self.erase_gate, batch_mask)
        memory = _retard_memory(memory, self.update_tensor, self.write_gate, self.erase_gate, batch_mask)
        self.assertTrue(torch.allclose(memory, self.memory_tensor, atol=1e-5),
                        "Retarded memory does not match original memory tensor.")

    def test_forward_reverse_inverses_at_edge_cases(self):
        """
        Test that when the erase gate is nearly saturated, we still
        can make an inverse.
        """
        erase_gate = torch.ones([self.batch_size, self.num_elements])*0.99
        batch_mask = torch.tensor([False, False])
        memory = _advance_memory(self.memory_tensor, self.update_tensor, self.write_gate, erase_gate, batch_mask)
        memory = _retard_memory(memory, self.update_tensor, self.write_gate, erase_gate, batch_mask)
        self.assertTrue(torch.allclose(memory, self.memory_tensor, atol=1e-5),
                        "Retarded memory does not match original memory tensor.")

    def test_absolute_error(self):
        """
        Test that small pertubations at a nearly saturated erase gate only produce a large
        percent error, but not a large absolute error. So long as this is the case,
        the model is numerically very stable, as dense layers care about absolute not
        relative error.
        """
        erase_gate = torch.ones([self.batch_size, self.num_elements])*0.99
        batch_mask = torch.tensor([False, False])
        pertubation = 1e-5
        memory = _advance_memory(self.memory_tensor, self.update_tensor, self.write_gate, erase_gate, batch_mask)
        memory = _retard_memory(memory, self.update_tensor, self.write_gate, erase_gate-pertubation, batch_mask)
        self.assertTrue(torch.allclose(memory, self.memory_tensor, atol=1e-5),
                        "Retarded memory does not match original memory tensor.")

    def test_masking(self):
        """
        Test that the masking mechanism works.
        """
        batch_mask = torch.tensor([True, True])
        memory = _advance_memory(self.memory_tensor, self.update_tensor, self.write_gate, self.erase_gate, batch_mask)

        self.assertTrue(torch.allclose(memory, self.memory_tensor, atol=1e-5),
                        " memory does not match original memory tensor.")

        metrics = _retard_memory(self.memory_tensor, self.update_tensor, self.write_gate, self.erase_gate,
                                 batch_mask)
        self.assertTrue(torch.allclose(memory, self.memory_tensor, atol=1e-5),
                        " memory does not match original memory tensor.")


class TestMetricHelperFunctions(unittest.TestCase):
    """
    Test the metrics helper functions for functionality
    and stability. In particular, test they correctly invert
    the metrics.
    """
    def setUp(self):
        # Common setup for all tests
        self.batch_size = 2
        self.num_elements = 4
        self.d_model = 3  # Example dimensionality

        # Initialize gates
        self.write_gate = torch.rand(self.batch_size, self.num_elements)
        self.erase_gate = torch.rand(self.batch_size, self.num_elements)*0.99

        # Initialize batch_mask
        self.batch_mask = torch.zeros(self.batch_size, dtype=torch.bool)  # No masking

        # Initialize metrics
        self.metrics = {
            "cum_write_mass": torch.zeros(self.batch_size, self.num_elements),
            "cum_erase_mass": torch.zeros(self.batch_size, self.num_elements),
            "effective_write_mass": torch.zeros(self.batch_size, self.num_elements),
            "timestep": torch.zeros(self.batch_size),
            "average_timestep_distance": torch.zeros(self.batch_size, self.num_elements),
        }

    def test_advance_retard_metrics(self):
        """
        Test advancing then retarding the metrics, see if we get the same
        thing back.
        """

        metrics = _advance_metrics(self.metrics, self.write_gate, self.erase_gate, self.batch_mask)
        metrics = _retard_metrics(metrics, self.write_gate, self.erase_gate, self.batch_mask)
        for name in metrics.keys():
            expected = self.metrics[name]
            actual = metrics[name]
            self.assertTrue(torch.allclose(expected, actual, atol=1e-5), f"{name} were not the same")

    def test_advance_retard_extremes(self):
        """
        Test advancing then retarding the metrics with a pretty saturated
        erase gate.
        """
        erase_gate = torch.ones([self.batch_size, self.num_elements])*0.99
        batch_mask = torch.tensor([False, False])
        metrics = _advance_metrics(self.metrics, self.write_gate, erase_gate, batch_mask)
        metrics = _retard_metrics(metrics, self.write_gate, erase_gate, self.batch_mask)
        for name in metrics.keys():
            expected = self.metrics[name]
            actual = metrics[name]
            self.assertTrue(torch.allclose(expected, actual, atol=1e-5), f"{name} were not the same")

    def test_numeric_stability(self):
        """
        Test that under the duress of a very saturated erase gate, permutations
        due to errors have a minimal effect on the absolute error.
        """
        erase_gate = torch.ones([self.batch_size, self.num_elements])*0.99
        batch_mask = torch.tensor([False, False])
        pertubation = 1e-5
        metrics = _advance_metrics(self.metrics, self.write_gate, erase_gate, batch_mask)
        metrics = _retard_metrics(metrics, self.write_gate, erase_gate+pertubation, self.batch_mask)
        for name in metrics.keys():
            expected = self.metrics[name]
            actual = metrics[name]
            self.assertTrue(torch.allclose(expected, actual, atol=1e-3), f"{name} were not the same")

    def test_masking(self):
        """
        Test that we respond correctly to masking
        """
        batch_mask = torch.tensor([True, True])

        metrics = _advance_metrics(self.metrics, self.write_gate, self.erase_gate, batch_mask)
        for name in metrics.keys():
            expected = self.metrics[name]
            actual = metrics[name]
            self.assertTrue(torch.allclose(expected, actual, atol=1e-5), f"{name} were not the same")

        metrics = _retard_metrics(self.metrics, self.write_gate, self.erase_gate, batch_mask)
        for name in metrics.keys():
            expected = self.metrics[name]
            actual = metrics[name]
            self.assertTrue(torch.allclose(expected, actual, atol=1e-5), f"{name} were not the same")

    def test_cumulative_metrics_update(self):
        """
        Test that cum_write_mass and cum_erase_mass are correctly updated based on write_gate and erase_gate.
        """
        # Define specific write and erase gates
        write_gate = torch.tensor([[0.1, 0.2, 0.3, 0.4],
                                   [0.5, 0.6, 0.7, 0.8]])
        erase_gate = torch.tensor([[0.05, 0.05, 0.05, 0.05],
                                   [0.05, 0.05, 0.05, 0.05]])
        batch_mask = torch.tensor([False, False])

        # Initialize metrics
        initial_metrics = {
            "cum_write_mass": torch.zeros(self.batch_size, self.num_elements),
            "cum_erase_mass": torch.zeros(self.batch_size, self.num_elements),
            "effective_write_mass": torch.zeros(self.batch_size, self.num_elements),
            "timestep": torch.zeros(self.batch_size),
            "average_timestep_distance": torch.zeros(self.batch_size, self.num_elements),
        }

        # Advance metrics
        advanced_metrics = _advance_metrics(initial_metrics, write_gate, erase_gate, batch_mask)

        # Expected cum_write_mass and cum_erase_mass
        expected_cum_write_mass = write_gate
        expected_cum_erase_mass = erase_gate

        self.assertTrue(torch.allclose(advanced_metrics["cum_write_mass"], expected_cum_write_mass, atol=1e-5),
                        "cum_write_mass not updated correctly.")
        self.assertTrue(torch.allclose(advanced_metrics["cum_erase_mass"], expected_cum_erase_mass, atol=1e-5),
                        "cum_erase_mass not updated correctly.")


    def test_average_timestep_distance_update(self):
        """
        Test that average_timestep_distance is updated correctly based on erase_gate and timestep.
        """
        write_gate = torch.tensor([[0.5, 0.5, 0.5, 0.5],
                                   [0.5, 0.5, 0.5, 0.5]])
        erase_gate = torch.tensor([[0.5, 0.5, 0.5, 0.5],
                                   [0.5, 0.5, 0.5, 0.5]])
        batch_mask = torch.tensor([False, False])

        initial_metrics = {
            "cum_write_mass": torch.zeros(self.batch_size, self.num_elements),
            "cum_erase_mass": torch.zeros(self.batch_size, self.num_elements),
            "effective_write_mass": torch.zeros(self.batch_size, self.num_elements),
            "timestep": torch.tensor([1.0, 1.0]),
            "average_timestep_distance": torch.zeros(self.batch_size, self.num_elements),
        }

        # Advance metrics
        advanced_metrics = _advance_metrics(initial_metrics, write_gate, erase_gate, batch_mask)

        # Compute expected effective_write_mass
        # Assuming _step_state_forward computes: s_new = s_old * erase_gate + write_gate * 1
        expected_effective_write_mass = initial_metrics["effective_write_mass"] * erase_gate + write_gate * 1.0

        self.assertTrue(torch.allclose(advanced_metrics["effective_write_mass"], expected_effective_write_mass, atol=1e-5),
                        "effective_write_mass not updated correctly.")

        # Compute expected average_timestep_distance
        # Assuming it updates as effective_write_mass / (timestep + epsilon)
        expected_average_timestep_distance = expected_effective_write_mass / (
                    initial_metrics["timestep"].unsqueeze(-1) + 1e-9)

        self.assertTrue(
            torch.allclose(advanced_metrics["average_timestep_distance"], expected_average_timestep_distance, atol=1e-5),
            "average_timestep_distance not updated correctly.")
    def test_effective_write_mass_calculation(self):
        """
        Test that effective_write_mass is updated correctly.
        """
        write_gate = torch.tensor([[0.2, 0.3, 0.4, 0.1],
                                   [0.5, 0.6, 0.7, 0.8]])
        erase_gate = torch.tensor([[0.1, 0.2, 0.1, 0.3],
                                   [0.4, 0.1, 0.2, 0.1]])
        batch_mask = torch.tensor([False, False])

        initial_metrics = {
            "cum_write_mass": torch.zeros(self.batch_size, self.num_elements),
            "cum_erase_mass": torch.zeros(self.batch_size, self.num_elements),
            "effective_write_mass": torch.tensor([[0.5, 0.5, 0.5, 0.5],
                                                 [0.5, 0.5, 0.5, 0.5]]),
            "timestep": torch.tensor([1.0, 1.0]),
            "average_timestep_distance": torch.zeros(self.batch_size, self.num_elements),
        }

        metrics = _advance_metrics(initial_metrics, write_gate, erase_gate, batch_mask)

        expected_effective_write_mass = initial_metrics["effective_write_mass"] * erase_gate + write_gate
        self.assertTrue(torch.allclose(metrics["effective_write_mass"], expected_effective_write_mass, atol=1e-5),
                        "effective_write_mass not updated correctly based on write_gate and erase_gate.")

    def test_timestep_increment(self):
        """
        Test that timestep increments correctly when batch_mask is False and remains unchanged when True.
        """
        # Test Case 1: batch_mask is False
        initial_metrics = self.metrics.copy()
        write_gate = torch.rand(self.batch_size, self.num_elements)
        erase_gate = torch.rand(self.batch_size, self.num_elements)*0.99
        batch_mask = torch.tensor([False, False])

        metrics = _advance_metrics(initial_metrics, write_gate, erase_gate, batch_mask)

        expected_timestep = initial_metrics["timestep"] + 1
        self.assertTrue(torch.allclose(metrics["timestep"], expected_timestep, atol=1e-5),
                        "timestep did not increment correctly when batch_mask is False.")

        # Test Case 2: batch_mask is True
        initial_metrics = metrics.copy()
        batch_mask = torch.tensor([True, True])

        metrics = _advance_metrics(initial_metrics, write_gate, erase_gate, batch_mask)

        expected_timestep = initial_metrics["timestep"]  # Should remain unchanged
        self.assertTrue(torch.allclose(metrics["timestep"], expected_timestep, atol=1e-5),
                        "timestep incorrectly incremented when batch_mask is True.")

# test_memory_state.py

import unittest
import torch
from src.main.arcAGI2024.memory.base import MemoryState


class TestMemoryState(unittest.TestCase):
    """
    Test the MemoryState class for correct initialization, state transitions,
    serialization, gradient setup, and persistence integrity.
    """

    def setUp(self):
        # Common setup for all tests
        self.batch_size = 2
        self.num_elements = 4
        self.d_model = 3  # Example dimensionality

        # Initialize memory tensors
        self.memory_tensors = {
            "mem1": torch.randn(self.batch_size, self.num_elements, self.d_model),
            "mem2": torch.randn(self.batch_size, self.num_elements, self.d_model)
        }

        # Initialize metrics
        self.metric_tensors = {
            "cum_write_mass": torch.zeros(self.batch_size, self.num_elements),
            "cum_erase_mass": torch.zeros(self.batch_size, self.num_elements),
            "effective_write_mass": torch.zeros(self.batch_size, self.num_elements),
            "timestep": torch.zeros(self.batch_size),
            "average_timestep_distance": torch.zeros(self.batch_size, self.num_elements),
        }

        # Initialize persistent tensors
        self.persistent_tensors = {
            "persistent_metric": torch.tensor([1.0, 2.0, 3.0, 4.0])
        }

        # Create MemoryState instance
        self.memory_state = MemoryState(
            metric_tensors=self.metric_tensors.copy(),
            memory_tensors=self.memory_tensors.copy(),
            persistent=self.persistent_tensors.copy()
        )

    def test_initialization_correctness(self):
        """
        Test that MemoryState initializes correctly with provided metrics,
        memory tensors, and persistent tensors.
        """
        for name, tensor in self.metric_tensors.items():
            self.assertTrue(torch.allclose(getattr(self.memory_state, name), tensor, atol=1e-5),
                            f"Metric '{name}' not initialized correctly.")

        for name, tensor in self.memory_tensors.items():
            self.assertTrue(torch.allclose(self.memory_state.get_memories()[name], tensor, atol=1e-5),
                            f"Memory tensor '{name}' not initialized correctly.")

        for name, tensor in self.persistent_tensors.items():
            self.assertTrue(torch.allclose(self.memory_state.get_persistent()[name], tensor, atol=1e-5),
                            f"Persistent tensor '{name}' not initialized correctly.")

    def test_initialization_missing_metrics_raises_error(self):
        """
        Test that initializing MemoryState without required metrics raises KeyError.
        """
        incomplete_metrics = {
            "cum_write_mass": torch.zeros(self.batch_size, self.num_elements),
            # "cum_erase_mass" is missing
            "effective_write_mass": torch.zeros(self.batch_size, self.num_elements),
            "timestep": torch.zeros(self.batch_size),
            "average_timestep_distance": torch.zeros(self.batch_size, self.num_elements),
        }

        with self.assertRaises(KeyError):
            MemoryState(
                metric_tensors=incomplete_metrics,
                memory_tensors=self.memory_tensors.copy(),
                persistent=self.persistent_tensors.copy()
            )

    def test_get_memories_returns_correct_tensors(self):
        """
        Test that get_memories returns the correct memory tensors.
        """
        memories = self.memory_state.get_memories()
        for name, tensor in self.memory_tensors.items():
            self.assertTrue(torch.allclose(memories[name], tensor, atol=1e-5),
                            f"get_memories did not return correct tensor for '{name}'.")

    def test_get_persistent_returns_correct_tensors(self):
        """
        Test that get_persistent returns the correct persistent tensors.
        """
        persistent = self.memory_state.get_persistent()
        for name, tensor in self.persistent_tensors.items():
            self.assertTrue(torch.allclose(persistent[name], tensor, atol=1e-5),
                            f"get_persistent did not return correct tensor for '{name}'.")

    def test_step_memory_forward_updates_state_correctly(self):
        """
        Test that step_memory_forward correctly updates metrics and memory tensors.
        This test assumes that the helper functions have been thoroughly tested.
        """
        # Define dummy update data
        update_metrics = {
            "cum_write_mass": torch.tensor([[0.1, 0.2, 0.3, 0.4],
                                           [0.5, 0.6, 0.7, 0.8]]),
            "cum_erase_mass": torch.tensor([[0.05, 0.05, 0.05, 0.05],
                                           [0.05, 0.05, 0.05, 0.05]]),
            "effective_write_mass": torch.tensor([[0.5, 0.5, 0.5, 0.5],
                                                 [0.5, 0.5, 0.5, 0.5]]),
            "timestep": torch.tensor([1.0, 1.0]),
            "average_timestep_distance": torch.tensor([[0.25, 0.25, 0.25, 0.25],
                                                      [0.25, 0.25, 0.25, 0.25]]),
        }

        # Define control gates and batch_mask
        write_gate = torch.tensor([[0.1, 0.2, 0.3, 0.4],
                                   [0.5, 0.6, 0.7, 0.8]])
        erase_gate = torch.tensor([[0.05, 0.05, 0.05, 0.05],
                                   [0.05, 0.05, 0.05, 0.05]])
        batch_mask = torch.tensor([False, False])

        # Perform forward step
        new_memory_state = self.memory_state.step_memory_forward(
            update=update_metrics,
            write_gate=write_gate,
            erase_gate=erase_gate,
            batch_mask=batch_mask
        )

        # Expected metrics after forward step
        expected_metrics = {
            "cum_write_mass": self.metric_tensors["cum_write_mass"] + update_metrics["cum_write_mass"],
            "cum_erase_mass": self.metric_tensors["cum_erase_mass"] + update_metrics["cum_erase_mass"],
            "effective_write_mass": self.metric_tensors["effective_write_mass"] * erase_gate + update_metrics["cum_write_mass"],
            "timestep": self.metric_tensors["timestep"] + 1,
            "average_timestep_distance": (self.metric_tensors["effective_write_mass"] * erase_gate + update_metrics["cum_write_mass"]) / (self.metric_tensors["timestep"] + 1),
        }

        # Assert metrics
        for name in expected_metrics.keys():
            self.assertTrue(torch.allclose(new_memory_state.metric_tensors[name], expected_metrics[name], atol=1e-5),
                            f"Metric '{name}' not updated correctly in step_memory_forward.")

        # Assert memory tensors (assuming _advance_memory updates them based on update)
        # This depends on how _advance_memory is implemented. Assuming it's handled correctly.
        # Here, we'll just check that memory tensors have been updated (not detailed).
        for name, tensor in self.memory_tensors.items():
            updated_tensor = new_memory_state.memory_tensors[name]
            # Assuming _advance_memory adds some value; here we check that it's not equal to the original
            self.assertFalse(torch.allclose(updated_tensor, tensor),
                             f"Memory tensor '{name}' was not updated in step_memory_forward.")

    def test_step_memory_reverse_reverts_state_correctly(self):
        """
        Test that step_memory_reverse correctly reverts the state back to the original.
        This ensures that forward and reverse steps are inverses.
        """
        # Define dummy update data
        update_metrics = {
            "cum_write_mass": torch.tensor([[0.1, 0.2, 0.3, 0.4],
                                           [0.5, 0.6, 0.7, 0.8]]),
            "cum_erase_mass": torch.tensor([[0.05, 0.05, 0.05, 0.05],
                                           [0.05, 0.05, 0.05, 0.05]]),
            "effective_write_mass": torch.tensor([[0.5, 0.5, 0.5, 0.5],
                                                 [0.5, 0.5, 0.5, 0.5]]),
            "timestep": torch.tensor([1.0, 1.0]),
            "average_timestep_distance": torch.tensor([[0.25, 0.25, 0.25, 0.25],
                                                      [0.25, 0.25, 0.25, 0.25]]),
        }

        # Define control gates and batch_mask
        write_gate = torch.tensor([[0.1, 0.2, 0.3, 0.4],
                                   [0.5, 0.6, 0.7, 0.8]])
        erase_gate = torch.tensor([[0.05, 0.05, 0.05, 0.05],
                                   [0.05, 0.05, 0.05, 0.05]])
        batch_mask = torch.tensor([False, False])

        # Perform forward step
        new_memory_state = self.memory_state.step_memory_forward(
            update=update_metrics,
            write_gate=write_gate,
            erase_gate=erase_gate,
            batch_mask=batch_mask
        )

        # Perform reverse step
        reverted_memory_state = new_memory_state.step_memory_reverse(
            update=update_metrics,
            write_gate=write_gate,
            erase_gate=erase_gate,
            batch_mask=batch_mask
        )

        # Assert that reverted metrics match original metrics
        for name in self.metric_tensors.keys():
            self.assertTrue(torch.allclose(reverted_memory_state.metric_tensors[name], self.metric_tensors[name], atol=1e-5),
                            f"Metric '{name}' was not reverted correctly in step_memory_reverse.")

        # Assert that memory tensors have been reverted to original
        for name, tensor in self.memory_tensors.items():
            reverted_tensor = reverted_memory_state.memory_tensors[name]
            self.assertTrue(torch.allclose(reverted_tensor, tensor, atol=1e-5),
                            f"Memory tensor '{name}' was not reverted correctly in step_memory_reverse.")

    def test_save_and_load_state(self):
        """
        Test that save_state and load_state correctly serialize and deserialize the memory state.
        """
        # Save the current state
        saved_state, _ = self.memory_state.save_state()

        # Load the state into a new MemoryState instance
        loaded_memory_state = MemoryState.load_state(saved_state, None)

        # Assert that metrics match
        for name in self.metric_tensors.keys():
            original = self.memory_state.metric_tensors[name]
            loaded = loaded_memory_state.metric_tensors[name]
            self.assertTrue(torch.allclose(original, loaded, atol=1e-5),
                            f"Metric '{name}' does not match after load_state.")

        # Assert that memory tensors match
        for name in self.memory_tensors.keys():
            original = self.memory_state.memory_tensors[name]
            loaded = loaded_memory_state.memory_tensors[name]
            self.assertTrue(torch.allclose(original, loaded, atol=1e-5),
                            f"Memory tensor '{name}' does not match after load_state.")

        # Assert that persistent tensors match
        for name in self.persistent_tensors.keys():
            original = self.memory_state.persistent_tensors[name]
            loaded = loaded_memory_state.persistent_tensors[name]
            self.assertTrue(torch.allclose(original, loaded, atol=1e-5),
                            f"Persistent tensor '{name}' does not match after load_state.")

    def test_setup_for_gradients(self):
        """
        Test that setup_for_gradients_ correctly detaches tensors and sets requires_grad=True.
        """
        # Perform gradient setup
        self.memory_state.setup_for_gradients_()

        # Assert that memory tensors are detached and require gradients
        for tensor in self.memory_state.memory_tensors.values():
            self.assertFalse(tensor.is_leaf, "Memory tensor should be detached and not a leaf.")
            self.assertTrue(tensor.requires_grad, "Memory tensor should require gradients after setup.")

        # Assert that metric tensors are detached and require gradients
        for tensor in self.memory_state.metric_tensors.values():
            self.assertFalse(tensor.is_leaf, "Metric tensor should be detached and not a leaf.")
            self.assertTrue(tensor.requires_grad, "Metric tensor should require gradients after setup.")

    def test_persistent_tensors_unchanged_after_updates(self):
        """
        Test that persistent tensors remain unchanged after memory updates.
        """
        # Store initial persistent tensors
        initial_persistent = {
            name: tensor.clone()
            for name, tensor in self.memory_state.get_persistent().items()
        }

        # Define dummy update data
        update_metrics = {
            "cum_write_mass": torch.tensor([[0.1, 0.2, 0.3, 0.4],
                                           [0.5, 0.6, 0.7, 0.8]]),
            "cum_erase_mass": torch.tensor([[0.05, 0.05, 0.05, 0.05],
                                           [0.05, 0.05, 0.05, 0.05]]),
            "effective_write_mass": torch.tensor([[0.5, 0.5, 0.5, 0.5],
                                                 [0.5, 0.5, 0.5, 0.5]]),
            "timestep": torch.tensor([1.0, 1.0]),
            "average_timestep_distance": torch.tensor([[0.25, 0.25, 0.25, 0.25],
                                                      [0.25, 0.25, 0.25, 0.25]]),
        }

        # Define control gates and batch_mask
        write_gate = torch.tensor([[0.1, 0.2, 0.3, 0.4],
                                   [0.5, 0.6, 0.7, 0.8]])
        erase_gate = torch.tensor([[0.05, 0.05, 0.05, 0.05],
                                   [0.05, 0.05, 0.05, 0.05]])
        batch_mask = torch.tensor([False, False])

        # Perform forward step
        new_memory_state = self.memory_state.step_memory_forward(
            update=update_metrics,
            write_gate=write_gate,
            erase_gate=erase_gate,
            batch_mask=batch_mask
        )

        # Assert persistent tensors are unchanged
        for name, tensor in initial_persistent.items():
            self.assertTrue(torch.allclose(new_memory_state.persistent_tensors[name], tensor, atol=1e-5),
                            f"Persistent tensor '{name}' changed after memory update.")

    def test_step_memory_forward_with_masking(self):
        """
        Test that step_memory_forward does not update metrics or memory tensors when batch_mask is True.
        """
        # Define dummy update data
        update_metrics = {
            "cum_write_mass": torch.tensor([[0.1, 0.2, 0.3, 0.4],
                                           [0.5, 0.6, 0.7, 0.8]]),
            "cum_erase_mass": torch.tensor([[0.05, 0.05, 0.05, 0.05],
                                           [0.05, 0.05, 0.05, 0.05]]),
            "effective_write_mass": torch.tensor([[0.5, 0.5, 0.5, 0.5],
                                                 [0.5, 0.5, 0.5, 0.5]]),
            "timestep": torch.tensor([1.0, 1.0]),
            "average_timestep_distance": torch.tensor([[0.25, 0.25, 0.25, 0.25],
                                                      [0.25, 0.25, 0.25, 0.25]]),
        }

        # Define control gates and batch_mask with masking enabled
        write_gate = torch.tensor([[0.1, 0.2, 0.3, 0.4],
                                   [0.5, 0.6, 0.7, 0.8]])
        erase_gate = torch.tensor([[0.05, 0.05, 0.05, 0.05],
                                   [0.05, 0.05, 0.05, 0.05]])
        batch_mask = torch.tensor([True, True])

        # Store initial metrics and memory tensors
        initial_metrics = {
            "cum_write_mass": self.memory_state.metric_tensors["cum_write_mass"].clone(),
            "cum_erase_mass": self.memory_state.metric_tensors["cum_erase_mass"].clone(),
            "effective_write_mass": self.memory_state.metric_tensors["effective_write_mass"].clone(),
            "timestep": self.memory_state.metric_tensors["timestep"].clone(),
            "average_timestep_distance": self.memory_state.metric_tensors["average_timestep_distance"].clone(),
        }
        initial_memory = {
            name: tensor.clone()
            for name, tensor in self.memory_state.memory_tensors.items()
        }

        # Perform forward step with masking
        new_memory_state = self.memory_state.step_memory_forward(
            update=update_metrics,
            write_gate=write_gate,
            erase_gate=erase_gate,
            batch_mask=batch_mask
        )

        # Assert that metrics remain unchanged
        for name in initial_metrics.keys():
            expected = initial_metrics[name]
            actual = new_memory_state.metric_tensors[name]
            self.assertTrue(torch.allclose(expected, actual, atol=1e-5),
                            f"Metric '{name}' should not change when batch_mask is True.")

        # Assert that memory tensors remain unchanged
        for name, tensor in initial_memory.items():
            updated_tensor = new_memory_state.memory_tensors[name]
            self.assertTrue(torch.allclose(updated_tensor, tensor, atol=1e-5),
                            f"Memory tensor '{name}' should not change when batch_mask is True.")

    def test_step_memory_reverse_with_masking(self):
        """
        Test that step_memory_reverse does not update metrics or memory tensors when batch_mask is True.
        """
        # Define dummy update data
        update_metrics = {
            "cum_write_mass": torch.tensor([[0.1, 0.2, 0.3, 0.4],
                                           [0.5, 0.6, 0.7, 0.8]]),
            "cum_erase_mass": torch.tensor([[0.05, 0.05, 0.05, 0.05],
                                           [0.05, 0.05, 0.05, 0.05]]),
            "effective_write_mass": torch.tensor([[0.5, 0.5, 0.5, 0.5],
                                                 [0.5, 0.5, 0.5, 0.5]]),
            "timestep": torch.tensor([1.0, 1.0]),
            "average_timestep_distance": torch.tensor([[0.25, 0.25, 0.25, 0.25],
                                                      [0.25, 0.25, 0.25, 0.25]]),
        }

        # Define control gates and batch_mask with masking enabled
        write_gate = torch.tensor([[0.1, 0.2, 0.3, 0.4],
                                   [0.5, 0.6, 0.7, 0.8]])
        erase_gate = torch.tensor([[0.05, 0.05, 0.05, 0.05],
                                   [0.05, 0.05, 0.05, 0.05]])
        batch_mask = torch.tensor([True, True])

        # Perform forward step to create a new state
        new_memory_state = self.memory_state.step_memory_forward(
            update=update_metrics,
            write_gate=write_gate,
            erase_gate=erase_gate,
            batch_mask=torch.tensor([False, False])
        )

        # Store initial metrics and memory tensors before reverse
        initial_metrics = {
            "cum_write_mass": self.memory_state.metric_tensors["cum_write_mass"].clone(),
            "cum_erase_mass": self.memory_state.metric_tensors["cum_erase_mass"].clone(),
            "effective_write_mass": self.memory_state.metric_tensors["effective_write_mass"].clone(),
            "timestep": self.memory_state.metric_tensors["timestep"].clone(),
            "average_timestep_distance": self.memory_state.metric_tensors["average_timestep_distance"].clone(),
        }
        initial_memory = {
            name: tensor.clone()
            for name, tensor in self.memory_state.memory_tensors.items()
        }

        # Perform reverse step with masking
        reverted_memory_state = new_memory_state.step_memory_reverse(
            update=update_metrics,
            write_gate=write_gate,
            erase_gate=erase_gate,
            batch_mask=batch_mask
        )

        # Assert that metrics remain unchanged after reverse with masking
        for name in initial_metrics.keys():
            expected = initial_metrics[name]
            actual = reverted_memory_state.metric_tensors[name]
            self.assertTrue(torch.allclose(expected, actual, atol=1e-5),
                            f"Metric '{name}' should not change when batch_mask is True during reverse.")

        # Assert that memory tensors remain unchanged after reverse with masking
        for name, tensor in initial_memory.items():
            reverted_tensor = reverted_memory_state.memory_tensors[name]
            self.assertTrue(torch.allclose(reverted_tensor, tensor, atol=1e-5),
                            f"Memory tensor '{name}' should not change when batch_mask is True during reverse.")

    def test_save_and_load_state_consistency(self):
        """
        Test that saving and then loading the state preserves all metrics and tensors.
        """
        # Perform some updates
        update_metrics = {
            "cum_write_mass": torch.tensor([[0.1, 0.2, 0.3, 0.4],
                                           [0.5, 0.6, 0.7, 0.8]]),
            "cum_erase_mass": torch.tensor([[0.05, 0.05, 0.05, 0.05],
                                           [0.05, 0.05, 0.05, 0.05]]),
            "effective_write_mass": torch.tensor([[0.5, 0.5, 0.5, 0.5],
                                                 [0.5, 0.5, 0.5, 0.5]]),
            "timestep": torch.tensor([1.0, 1.0]),
            "average_timestep_distance": torch.tensor([[0.25, 0.25, 0.25, 0.25],
                                                      [0.25, 0.25, 0.25, 0.25]]),
        }
        write_gate = torch.tensor([[0.1, 0.2, 0.3, 0.4],
                                   [0.5, 0.6, 0.7, 0.8]])
        erase_gate = torch.tensor([[0.05, 0.05, 0.05, 0.05],
                                   [0.05, 0.05, 0.05, 0.05]])
        batch_mask = torch.tensor([False, False])

        # Advance memory state
        advanced_memory_state = self.memory_state.step_memory_forward(
            update=update_metrics,
            write_gate=write_gate,
            erase_gate=erase_gate,
            batch_mask=batch_mask
        )

        # Save state
        saved_state, _ = advanced_memory_state.save_state()

        # Load state into a new MemoryState instance
        loaded_memory_state = MemoryState.load_state(saved_state, None)

        # Assert that metrics match
        for name in self.metric_tensors.keys():
            original = advanced_memory_state.metric_tensors[name]
            loaded = loaded_memory_state.metric_tensors[name]
            self.assertTrue(torch.allclose(original, loaded, atol=1e-5),
                            f"Metric '{name}' does not match after load_state.")

        # Assert that memory tensors match
        for name in self.memory_tensors.keys():
            original = advanced_memory_state.memory_tensors[name]
            loaded = loaded_memory_state.memory_tensors[name]
            self.assertTrue(torch.allclose(original, loaded, atol=1e-5),
                            f"Memory tensor '{name}' does not match after load_state.")

        # Assert that persistent tensors match
        for name in self.persistent_tensors.keys():
            original = advanced_memory_state.persistent_tensors[name]
            loaded = loaded_memory_state.persistent_tensors[name]
            self.assertTrue(torch.allclose(original, loaded, atol=1e-5),
                            f"Persistent tensor '{name}' does not match after load_state.")

    def test_step_memory_forward_invalid_inputs(self):
        """
        Test that step_memory_forward raises appropriate errors when provided with invalid inputs.
        """
        # Define invalid update_metrics (missing a required metric)
        incomplete_update_metrics = {
            "cum_write_mass": torch.tensor([[0.1, 0.2, 0.3, 0.4],
                                           [0.5, 0.6, 0.7, 0.8]]),
            # "cum_erase_mass" is missing
            "effective_write_mass": torch.tensor([[0.5, 0.5, 0.5, 0.5],
                                                 [0.5, 0.5, 0.5, 0.5]]),
            "timestep": torch.tensor([1.0, 1.0]),
            "average_timestep_distance": torch.tensor([[0.25, 0.25, 0.25, 0.25],
                                                      [0.25, 0.25, 0.25, 0.25]]),
        }
        write_gate = torch.tensor([[0.1, 0.2, 0.3, 0.4],
                                   [0.5, 0.6, 0.7, 0.8]])
        erase_gate = torch.tensor([[0.05, 0.05, 0.05, 0.05],
                                   [0.05, 0.05, 0.05, 0.05]])
        batch_mask = torch.tensor([False, False])

        with self.assertRaises(KeyError):
            self.memory_state.step_memory_forward(
                update=incomplete_update_metrics,
                write_gate=write_gate,
                erase_gate=erase_gate,
                batch_mask=batch_mask
            )

        # Define mismatched tensor shapes
        mismatched_write_gate = torch.tensor([[0.1, 0.2, 0.3],  # Missing one element
                                             [0.5, 0.6, 0.7, 0.8]])
        with self.assertRaises(ValueError):
            self.memory_state.step_memory_forward(
                update=self.metric_tensors,
                write_gate=mismatched_write_gate,
                erase_gate=erase_gate,
                batch_mask=batch_mask
            )

    def test_step_memory_reverse_invalid_inputs(self):
        """
        Test that step_memory_reverse raises appropriate errors when provided with invalid inputs.
        """
        # Define invalid update_metrics (missing a required metric)
        incomplete_update_metrics = {
            "cum_write_mass": torch.tensor([[0.1, 0.2, 0.3, 0.4],
                                           [0.5, 0.6, 0.7, 0.8]]),
            # "cum_erase_mass" is missing
            "effective_write_mass": torch.tensor([[0.5, 0.5, 0.5, 0.5],
                                                 [0.5, 0.5, 0.5, 0.5]]),
            "timestep": torch.tensor([1.0, 1.0]),
            "average_timestep_distance": torch.tensor([[0.25, 0.25, 0.25, 0.25],
                                                      [0.25, 0.25, 0.25, 0.25]]),
        }
        write_gate = torch.tensor([[0.1, 0.2, 0.3, 0.4],
                                   [0.5, 0.6, 0.7, 0.8]])
        erase_gate = torch.tensor([[0.05, 0.05, 0.05, 0.05],
                                   [0.05, 0.05, 0.05, 0.05]])
        batch_mask = torch.tensor([False, False])

        with self.assertRaises(KeyError):
            self.memory_state.step_memory_reverse(
                update=incomplete_update_metrics,
                write_gate=write_gate,
                erase_gate=erase_gate,
                batch_mask=batch_mask
            )

        # Define mismatched tensor shapes
        mismatched_erase_gate = torch.tensor([[0.05, 0.05, 0.05],  # Missing one element
                                             [0.05, 0.05, 0.05, 0.05]])
        with self.assertRaises(ValueError):
            self.memory_state.step_memory_reverse(
                update=self.metric_tensors,
                write_gate=write_gate,
                erase_gate=mismatched_erase_gate,
                batch_mask=batch_mask
            )

    def test_step_memory_forward_and_reverse_multiple_steps(self):
        """
        Test stepping forward and then reversing multiple steps to ensure state consistency.
        """
        # Define multiple steps of updates
        updates = [
            {
                "cum_write_mass": torch.tensor([[0.1, 0.2, 0.3, 0.4],
                                               [0.5, 0.6, 0.7, 0.8]]),
                "cum_erase_mass": torch.tensor([[0.05, 0.05, 0.05, 0.05],
                                               [0.05, 0.05, 0.05, 0.05]]),
                "effective_write_mass": torch.tensor([[0.5, 0.5, 0.5, 0.5],
                                                     [0.5, 0.5, 0.5, 0.5]]),
                "timestep": torch.tensor([1.0, 1.0]),
                "average_timestep_distance": torch.tensor([[0.25, 0.25, 0.25, 0.25],
                                                          [0.25, 0.25, 0.25, 0.25]]),
            },
            {
                "cum_write_mass": torch.tensor([[0.2, 0.3, 0.4, 0.5],
                                               [0.6, 0.7, 0.8, 0.9]]),
                "cum_erase_mass": torch.tensor([[0.1, 0.1, 0.1, 0.1],
                                               [0.1, 0.1, 0.1, 0.1]]),
                "effective_write_mass": torch.tensor([[0.6, 0.6, 0.6, 0.6],
                                                     [0.6, 0.6, 0.6, 0.6]]),
                "timestep": torch.tensor([2.0, 2.0]),
                "average_timestep_distance": torch.tensor([[0.3, 0.3, 0.3, 0.3],
                                                          [0.3, 0.3, 0.3, 0.3]]),
            }
        ]

        write_gates = [
            torch.tensor([[0.1, 0.2, 0.3, 0.4],
                          [0.5, 0.6, 0.7, 0.8]]),
            torch.tensor([[0.1, 0.1, 0.1, 0.1],
                          [0.1, 0.1, 0.1, 0.1]])
        ]

        erase_gates = [
            torch.tensor([[0.05, 0.05, 0.05, 0.05],
                          [0.05, 0.05, 0.05, 0.05]]),
            torch.tensor([[0.05, 0.05, 0.05, 0.05],
                          [0.05, 0.05, 0.05, 0.05]])
        ]

        batch_mask = torch.tensor([False, False])

        # Perform multiple forward steps
        current_memory_state = self.memory_state
        for update, write_gate, erase_gate in zip(updates, write_gates, erase_gates):
            current_memory_state = current_memory_state.step_memory_forward(
                update=update,
                write_gate=write_gate,
                erase_gate=erase_gate,
                batch_mask=batch_mask
            )

        # Perform multiple reverse steps
        for update, write_gate, erase_gate in zip(reversed(updates), reversed(write_gates), reversed(erase_gates)):
            current_memory_state = current_memory_state.step_memory_reverse(
                update=update,
                write_gate=write_gate,
                erase_gate=erase_gate,
                batch_mask=batch_mask
            )

        # Assert that reverted state matches initial state
        for name in self.metric_tensors.keys():
            expected = self.metric_tensors[name]
            actual = current_memory_state.metric_tensors[name]
            self.assertTrue(torch.allclose(expected, actual, atol=1e-5),
                            f"Metric '{name}' did not revert correctly after multiple forward and reverse steps.")

        for name, tensor in self.memory_tensors.items():
            reverted_tensor = current_memory_state.memory_tensors[name]
            self.assertTrue(torch.allclose(reverted_tensor, tensor, atol=1e-5),
                            f"Memory tensor '{name}' did not revert correctly after multiple forward and reverse steps.")

    def test_memory_state_requires_grad_after_setup(self):
        """
        Test that after calling setup_for_gradients_, all tensors have requires_grad=True.
        """
        # Call setup_for_gradients_
        self.memory_state.setup_for_gradients_()

        # Check memory tensors
        for tensor in self.memory_state.memory_tensors.values():
            self.assertTrue(tensor.requires_grad, "Memory tensor does not require grad after setup_for_gradients_.")
            self.assertFalse(tensor.is_leaf, "Memory tensor should be detached and not a leaf after setup_for_gradients_.")

        # Check metric tensors
        for tensor in self.memory_state.metric_tensors.values():
            self.assertTrue(tensor.requires_grad, "Metric tensor does not require grad after setup_for_gradients_.")
            self.assertFalse(tensor.is_leaf, "Metric tensor should be detached and not a leaf after setup_for_gradients_.")

    def test_memory_state_persistence_integrity(self):
        """
        Test that persistent tensors remain unchanged after multiple memory steps.
        """
        # Store initial persistent tensors
        initial_persistent = {
            name: tensor.clone()
            for name, tensor in self.memory_state.get_persistent().items()
        }

        # Define dummy update data
        updates = [
            {
                "cum_write_mass": torch.tensor([[0.1, 0.2, 0.3, 0.4],
                                               [0.5, 0.6, 0.7, 0.8]]),
                "cum_erase_mass": torch.tensor([[0.05, 0.05, 0.05, 0.05],
                                               [0.05, 0.05, 0.05, 0.05]]),
                "effective_write_mass": torch.tensor([[0.5, 0.5, 0.5, 0.5],
                                                     [0.5, 0.5, 0.5, 0.5]]),
                "timestep": torch.tensor([1.0, 1.0]),
                "average_timestep_distance": torch.tensor([[0.25, 0.25, 0.25, 0.25],
                                                          [0.25, 0.25, 0.25, 0.25]]),
            }
        ]

        write_gate = torch.tensor([[0.1, 0.2, 0.3, 0.4],
                                   [0.5, 0.6, 0.7, 0.8]])
        erase_gate = torch.tensor([[0.05, 0.05, 0.05, 0.05],
                                   [0.05, 0.05, 0.05, 0.05]])
        batch_mask = torch.tensor([False, False])

        # Perform multiple forward steps
        for update in updates:
            self.memory_state = self.memory_state.step_memory_forward(
                update=update,
                write_gate=write_gate,
                erase_gate=erase_gate,
                batch_mask=batch_mask
            )

        # Assert that persistent tensors remain unchanged
        for name, tensor in initial_persistent.items():
            self.assertTrue(torch.allclose(self.memory_state.get_persistent()[name], tensor, atol=1e-5),
                            f"Persistent tensor '{name}' changed after multiple memory updates.")

    def test_step_memory_forward_with_partial_masking(self):
        """
        Test that step_memory_forward updates metrics and memory tensors correctly when only some batches are masked.
        """
        # Define dummy update data
        update_metrics = {
            "cum_write_mass": torch.tensor([[0.1, 0.2, 0.3, 0.4],
                                           [0.5, 0.6, 0.7, 0.8]]),
            "cum_erase_mass": torch.tensor([[0.05, 0.05, 0.05, 0.05],
                                           [0.05, 0.05, 0.05, 0.05]]),
            "effective_write_mass": torch.tensor([[0.5, 0.5, 0.5, 0.5],
                                                 [0.5, 0.5, 0.5, 0.5]]),
            "timestep": torch.tensor([1.0, 1.0]),
            "average_timestep_distance": torch.tensor([[0.25, 0.25, 0.25, 0.25],
                                                      [0.25, 0.25, 0.25, 0.25]]),
        }

        write_gate = torch.tensor([[0.1, 0.2, 0.3, 0.4],
                                   [0.5, 0.6, 0.7, 0.8]])
        erase_gate = torch.tensor([[0.05, 0.05, 0.05, 0.05],
                                   [0.05, 0.05, 0.05, 0.05]])
        batch_mask = torch.tensor([False, True])  # Only first batch is unmasked

        # Perform forward step
        new_memory_state = self.memory_state.step_memory_forward(
            update=update_metrics,
            write_gate=write_gate,
            erase_gate=erase_gate,
            batch_mask=batch_mask
        )

        # Expected metrics:
        # For first batch: updated
        # For second batch: unchanged

        expected_metrics = {
            "cum_write_mass": torch.tensor([[0.1, 0.2, 0.3, 0.4],
                                           [0.0, 0.0, 0.0, 0.0]]),
            "cum_erase_mass": torch.tensor([[0.05, 0.05, 0.05, 0.05],
                                           [0.0, 0.0, 0.0, 0.0]]),
            "effective_write_mass": torch.tensor([[0.0 * 0.05 + 0.1, 0.0 * 0.05 + 0.2,
                                                 0.0 * 0.05 + 0.3, 0.0 * 0.05 + 0.4],
                                                [0.0, 0.0, 0.0, 0.0]]),
            "timestep": torch.tensor([1.0 + 1, 1.0]),  # Only first batch increments
            "average_timestep_distance": torch.tensor([[0.1 / 2.0, 0.2 / 2.0, 0.3 / 2.0, 0.4 / 2.0],
                                                      [0.0, 0.0, 0.0, 0.0]]),
        }

        # Assert metrics for first batch
        for name, expected in expected_metrics.items():
            actual = new_memory_state.metric_tensors[name]
            self.assertTrue(torch.allclose(actual, expected, atol=1e-5),
                            f"Metric '{name}' not updated correctly with partial masking.")

        # Assert that memory tensors for first batch have been updated
        # This depends on how _advance_memory updates them; assuming it's handled correctly
        # Here, we just check that unmasked batches have been updated
        for name, tensor in self.memory_tensors.items():
            updated_tensor = new_memory_state.memory_tensors[name]
            # For first batch, tensors should have changed
            self.assertFalse(torch.allclose(updated_tensor[0], tensor[0], atol=1e-5),
                             f"Memory tensor '{name}' for first batch was not updated correctly.")
            # For second batch, tensors should remain unchanged
            self.assertTrue(torch.allclose(updated_tensor[1], tensor[1], atol=1e-5),
                            f"Memory tensor '{name}' for second batch should not have been updated.")
