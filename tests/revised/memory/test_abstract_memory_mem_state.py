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

    def test_save_and_load_state_consistency(self):
        """
        Test that save_state and load_state correctly serialize and deserialize the memory state.
        """
        # Perform some updates by defining new memory tensors
        new_mem1 = self.memory_tensors["mem1"] + torch.tensor([[[0.1, 0.1, 0.1],
                                                               [0.1, 0.1, 0.1],
                                                               [0.1, 0.1, 0.1],
                                                               [0.1, 0.1, 0.1]],
                                                              [[0.2, 0.2, 0.2],
                                                               [0.2, 0.2, 0.2],
                                                               [0.2, 0.2, 0.2],
                                                               [0.2, 0.2, 0.2]]])

        new_mem2 = self.memory_tensors["mem2"] + torch.tensor([[[0.05, 0.05, 0.05],
                                                               [0.05, 0.05, 0.05],
                                                               [0.05, 0.05, 0.05],
                                                               [0.05, 0.05, 0.05]],
                                                              [[0.1, 0.1, 0.1],
                                                               [0.1, 0.1, 0.1],
                                                               [0.1, 0.1, 0.1],
                                                               [0.1, 0.1, 0.1]]])

        new_memory_tensors = {
            "mem1": new_mem1,
            "mem2": new_mem2
        }

        # Define control gates and batch_mask
        write_gate = torch.tensor([[0.1, 0.2, 0.3, 0.4],
                                   [0.5, 0.6, 0.7, 0.8]])
        erase_gate = torch.tensor([[0.05, 0.05, 0.05, 0.05],
                                   [0.05, 0.05, 0.05, 0.05]])
        batch_mask = torch.tensor([False, False])

        # Perform forward step
        advanced_memory_state = self.memory_state.step_memory_forward(
            update=new_memory_tensors,
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

    def test_setup_for_gradients(self):
        """
        Test that setup_for_gradients_ correctly detaches tensors and sets requires_grad=True.
        """
        # Perform gradient setup
        self.memory_state.setup_for_gradients_()

        # Check memory tensors
        for name, tensor in self.memory_state.memory_tensors.items():
            self.assertTrue(tensor.requires_grad,
                            f"Memory tensor '{name}' does not require grad after setup_for_gradients_.")
            self.assertTrue(tensor.is_leaf,
                            f"Memory tensor '{name}' should be a leaf tensor after setup_for_gradients_.")

        # Check metric tensors
        for name, tensor in self.memory_state.metric_tensors.items():
            self.assertTrue(tensor.requires_grad,
                            f"Metric tensor '{name}' does not require grad after setup_for_gradients_.")
            self.assertTrue(tensor.is_leaf,
                            f"Metric tensor '{name}' should be a leaf tensor after setup_for_gradients_.")

    def test_persistent_tensors_unchanged_after_updates(self):
        """
        Test that persistent tensors remain unchanged after memory updates.
        """
        # Store initial persistent tensors
        initial_persistent = {
            name: tensor.clone()
            for name, tensor in self.memory_state.get_persistent().items()
        }

        # Define new memory tensors
        new_mem1 = self.memory_tensors["mem1"] + torch.tensor([[[0.1, 0.1, 0.1],
                                                               [0.1, 0.1, 0.1],
                                                               [0.1, 0.1, 0.1],
                                                               [0.1, 0.1, 0.1]],
                                                              [[0.2, 0.2, 0.2],
                                                               [0.2, 0.2, 0.2],
                                                               [0.2, 0.2, 0.2],
                                                               [0.2, 0.2, 0.2]]])

        new_mem2 = self.memory_tensors["mem2"] + torch.tensor([[[0.05, 0.05, 0.05],
                                                               [0.05, 0.05, 0.05],
                                                               [0.05, 0.05, 0.05],
                                                               [0.05, 0.05, 0.05]],
                                                              [[0.1, 0.1, 0.1],
                                                               [0.1, 0.1, 0.1],
                                                               [0.1, 0.1, 0.1],
                                                               [0.1, 0.1, 0.1]]])

        new_memory_tensors = {
            "mem1": new_mem1,
            "mem2": new_mem2
        }

        # Define control gates and batch_mask
        write_gate = torch.tensor([[0.1, 0.2, 0.3, 0.4],
                                   [0.5, 0.6, 0.7, 0.8]])
        erase_gate = torch.tensor([[0.05, 0.05, 0.05, 0.05],
                                   [0.05, 0.05, 0.05, 0.05]])
        batch_mask = torch.tensor([False, False])

        # Perform forward step
        updated_memory_state = self.memory_state.step_memory_forward(
            update=new_memory_tensors,
            write_gate=write_gate,
            erase_gate=erase_gate,
            batch_mask=batch_mask
        )

        # Assert persistent tensors are unchanged
        for name, tensor in initial_persistent.items():
            self.assertTrue(torch.allclose(updated_memory_state.persistent_tensors[name], tensor, atol=1e-5),
                            f"Persistent tensor '{name}' changed after memory update.")

    def test_step_forward_then_reverse_restores_state(self):
        """
        Test that stepping the memory forward and then backward restores the original state.
        Assumes that step_memory_forward and step_memory_reverse are correctly implemented.
        """
        # Step 1: Save the original state
        original_metrics = {k: v.clone() for k, v in self.memory_state.metric_tensors.items()}
        original_memories = {k: v.clone() for k, v in self.memory_state.memory_tensors.items()}
        original_persistent = {k: v.clone() for k, v in self.memory_state.persistent_tensors.items()}

        # Step 2: Define new memory tensors by applying known increments
        # For mem1, add 0.2 to each element
        new_mem1 = self.memory_tensors["mem1"] + 0.2

        # For mem2, add 0.1 to each element
        new_mem2 = self.memory_tensors["mem2"] + 0.1

        new_memory_tensors = {
            "mem1": new_mem1,
            "mem2": new_mem2
        }

        # Step 3: Define write_gate and erase_gate
        write_gate = torch.tensor([[0.2, 0.2, 0.2, 0.2],
                                   [0.4, 0.4, 0.4, 0.4]])
        erase_gate = torch.tensor([[0.1, 0.1, 0.1, 0.1],
                                   [0.1, 0.1, 0.1, 0.1]])
        batch_mask = torch.tensor([False, False])

        # Step 4: Perform forward step
        forward_memory_state = self.memory_state.step_memory_forward(
            update=new_memory_tensors,
            write_gate=write_gate,
            erase_gate=erase_gate,
            batch_mask=batch_mask
        )

        # Step 5: Verify that tensors have changed
        # Check that at least one metric or memory tensor has changed
        tensors_changed = False

        # Check metrics
        for name in self.metric_tensors.keys():
            if not torch.allclose(original_metrics[name], forward_memory_state.metric_tensors[name], atol=1e-5):
                tensors_changed = True
                break

        # Check memories if metrics haven't changed yet
        if not tensors_changed:
            for name in self.memory_tensors.keys():
                if not torch.allclose(original_memories[name], forward_memory_state.memory_tensors[name], atol=1e-5):
                    tensors_changed = True
                    break

        self.assertTrue(tensors_changed, "MemoryState did not update after step_memory_forward.")

        # Step 6: Perform reverse step
        reverse_memory_state = forward_memory_state.step_memory_reverse(
            update=new_memory_tensors,  # original memories
            write_gate=write_gate,
            erase_gate=erase_gate,
            batch_mask=batch_mask
        )

        # Step 7: Verify that the reverted state matches the original state

        # Verify metrics
        for name in self.metric_tensors.keys():
            expected = original_metrics[name]
            actual = reverse_memory_state.metric_tensors[name]
            self.assertTrue(torch.allclose(expected, actual, atol=1e-5),
                            f"Metric '{name}' was not reverted correctly in step_memory_reverse.")

        # Verify memory tensors
        for name in self.memory_tensors.keys():
            expected = original_memories[name]
            actual = reverse_memory_state.memory_tensors[name]
            self.assertTrue(torch.allclose(expected, actual, atol=1e-5),
                            f"Memory tensor '{name}' was not reverted correctly in step_memory_reverse.")

        # Verify persistent tensors remain unchanged
        for name in self.persistent_tensors.keys():
            expected = original_persistent[name]
            actual = reverse_memory_state.persistent_tensors[name]
            self.assertTrue(torch.allclose(expected, actual, atol=1e-5),
                            f"Persistent tensor '{name}' was altered after reversing memory step.")


    def test_normalized_timestep_distance_bounds(self):
        """
        Test that the normalized_timestep_distance is always between 0 and 1, inclusive.
        """
        # Step 1: Save the original state
        original_metrics = {k: v.clone() for k, v in self.memory_state.metric_tensors.items()}
        original_memories = {k: v.clone() for k, v in self.memory_state.memory_tensors.items()}
        original_persistent = {k: v.clone() for k, v in self.memory_state.persistent_tensors.items()}

        # Step 2: Define new memory tensors by applying known increments
        # For mem1, add 0.3 to each element
        new_mem1 = self.memory_tensors["mem1"] + 0.3

        # For mem2, add 0.2 to each element
        new_mem2 = self.memory_tensors["mem2"] + 0.2

        new_memory_tensors = {
            "mem1": new_mem1,
            "mem2": new_mem2
        }

        # Step 3: Define write_gate and erase_gate
        write_gate = torch.tensor([[0.3, 0.3, 0.3, 0.3],
                                   [0.6, 0.6, 0.6, 0.6]])
        erase_gate = torch.tensor([[0.15, 0.15, 0.15, 0.15],
                                   [0.15, 0.15, 0.15, 0.15]])
        batch_mask = torch.tensor([False, False])

        # Step 4: Perform forward step
        forward_memory_state = self.memory_state.step_memory_forward(
            update=new_memory_tensors,
            write_gate=write_gate,
            erase_gate=erase_gate,
            batch_mask=batch_mask
        )

        # Step 5: Check normalized_timestep_distance bounds after forward step
        normalized_timestep_distance_forward = forward_memory_state.normalized_timestep_distance
        self.assertTrue(
            torch.all(normalized_timestep_distance_forward >= 0) and torch.all(normalized_timestep_distance_forward <= 1),
            "normalized_timestep_distance is not within [0, 1] after forward step."
        )

        # Step 6: Perform reverse step
        reverse_memory_state = forward_memory_state.step_memory_reverse(
            update=new_memory_tensors,  # original memories
            write_gate=write_gate,
            erase_gate=erase_gate,
            batch_mask=batch_mask
        )

        # Step 7: Check normalized_timestep_distance bounds after reverse step
        normalized_timestep_distance_reverse = reverse_memory_state.normalized_timestep_distance
        self.assertTrue(
            torch.all(normalized_timestep_distance_reverse >= 0) and torch.all(normalized_timestep_distance_reverse <= 1),
            "normalized_timestep_distance is not within [0, 1] after reverse step."
        )

        # Optional: Check bounds on the original state
        original_normalized_timestep_distance = self.memory_state.normalized_timestep_distance
        self.assertTrue(
            torch.all(original_normalized_timestep_distance >= 0) and torch.all(original_normalized_timestep_distance <= 1),
            "normalized_timestep_distance is not within [0, 1] in the original state."
        )