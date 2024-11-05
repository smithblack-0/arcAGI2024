import unittest
import torch
from torch import nn
from src.old.arcAGI2024 import act_factory_registry, act_controller_registry


class TestAdaptiveComputationTime(unittest.TestCase):
    """
    Unit tests for the AdaptiveComputationTime and AdaptiveComputationTimeFactory classes.
    """

    def setUp(self):
        """
        Sets up the basic parameters used for creating ACT instances.
        """
        self.batch_shape = torch.Size([4, 3])
        self.d_model = 8
        self.dtype = torch.float32
        self.device = torch.device('cpu')

        # Threshold for halting
        self.threshold = 0.99

        # Initialize the factory
        self.factory = act_factory_registry.build(threshold=self.threshold,
                                                  dtype=self.dtype,
                                                  device=self.device)

        # Create example accumulator templates
        self.accumulator_templates = {
            'layer1': torch.zeros(*self.batch_shape, self.d_model, dtype=self.dtype, device=self.device),
            'layer2': torch.ones(*self.batch_shape, self.d_model, dtype=self.dtype, device=self.device) * 0.5
        }

    def test_initialization(self):
        """
        Tests the initialization of the ACT mechanism using the factory.
        """
        act_instance = self.factory.forward(batch_shape=self.batch_shape, **self.accumulator_templates)

        # Check that the ACT instance was initialized correctly
        self.assertEqual(act_instance.halting_probabilities.shape, self.batch_shape)
        self.assertEqual(act_instance.residual_probabilities.shape, self.batch_shape)
        self.assertEqual(act_instance.has_halted.shape, self.batch_shape)
        self.assertEqual(act_instance.steps_taken.shape, self.batch_shape)
        self.assertEqual(act_instance.probabilistic_steps_taken.shape, self.batch_shape)

        # Verify the accumulator shapes
        for key, accumulator in act_instance.accumulated_outputs.items():
            self.assertEqual(accumulator.shape, self.accumulator_templates[key].shape)

    def test_step_functionality(self):
        """
        Tests the ACT step mechanism for halting and accumulating outputs.
        """
        act_instance = self.factory.forward(batch_shape=self.batch_shape, **self.accumulator_templates)

        # Create example halting probabilities and outputs
        halting_prob = torch.full(self.batch_shape, 0.1, dtype=self.dtype, device=self.device)
        outputs = {
            'layer1': torch.randn(*self.batch_shape, self.d_model, dtype=self.dtype, device=self.device),
            'layer2': torch.randn(*self.batch_shape, self.d_model, dtype=self.dtype, device=self.device)
        }

        # Perform a step
        act_instance.step(halting_prob, **outputs)

        # Ensure that halting probabilities are updated
        self.assertTrue((act_instance.halting_probabilities > 0).all())

        # Verify that outputs have been accumulated
        for key, output in outputs.items():
            self.assertTrue(torch.allclose(act_instance.accumulated_outputs[key], output * halting_prob.unsqueeze(-1)))

    def test_should_continue(self):
        """
        Tests the `should_continue` method to ensure that it correctly identifies when the process should halt.
        """
        act_instance = self.factory.forward(batch_shape=self.batch_shape, **self.accumulator_templates)

        # Initially, should_continue should return True since no samples have halted
        self.assertTrue(act_instance.should_continue())

        # Simulate all samples halting
        act_instance.has_halted[:] = True

        # Now, should_continue should return False
        self.assertFalse(act_instance.should_continue())

    def test_finalize(self):
        """
        Tests the finalize method to ensure it completes the accumulation and provides the final results.
        """
        act_instance = self.factory.forward(batch_shape=self.batch_shape, **self.accumulator_templates)

        # Perform several steps to simulate computation
        halting_prob = torch.full(self.batch_shape, 0.1, dtype=self.dtype, device=self.device)
        outputs = {
            'layer1': torch.randn(*self.batch_shape, self.d_model, dtype=self.dtype, device=self.device),
            'layer2': torch.randn(*self.batch_shape, self.d_model, dtype=self.dtype, device=self.device)
        }

        for _ in range(10):
            act_instance.step(halting_prob, **outputs)

        # Simulate halting for all samples
        act_instance.has_halted[:] = True

        # Finalize and get accumulated outputs
        final_outputs = act_instance.finalize()

        # Ensure the final outputs have the correct shapes
        for key, accumulated in final_outputs.items():
            self.assertEqual(accumulated.shape, self.accumulator_templates[key].shape)

    def test_finalize_without_halt_raises_error(self):
        """
        Tests that finalizing without all samples having halted raises a RuntimeError.
        """
        act_instance = self.factory.forward(batch_shape=self.batch_shape, **self.accumulator_templates)

        # Perform some steps but do not halt all samples
        halting_prob = torch.full(self.batch_shape, 0.1, dtype=self.dtype, device=self.device)
        outputs = {
            'layer1': torch.randn(*self.batch_shape, self.d_model, dtype=self.dtype, device=self.device),
            'layer2': torch.randn(*self.batch_shape, self.d_model, dtype=self.dtype, device=self.device)
        }

        for _ in range(5):
            act_instance.step(halting_prob, **outputs)

        # Attempt to finalize should raise an error since not all samples have halted
        with self.assertRaises(RuntimeError):
            act_instance.finalize()

class TestACTController(unittest.TestCase):
    """
    Unit tests for the ACTController class.
    """

    def setUp(self):
        """
        Sets up the basic parameters used for creating ACTController instances.
        """
        self.batch_shape = torch.Size([4, 3])
        self.d_model = 8
        self.dtype = torch.float32
        self.device = torch.device('cpu')

        # Threshold for halting
        self.threshold = 0.99

        # Initialize the controller using the registry
        self.controller = act_controller_registry.build(
            d_model=self.d_model,
            threshold = self.threshold,
            device=self.device,
            dtype=self.dtype
        )

        # Create example accumulator templates
        self.accumulator_templates = {
            'layer1': torch.zeros(*self.batch_shape, self.d_model, dtype=self.dtype, device=self.device),
            'layer2': torch.ones(*self.batch_shape, self.d_model, dtype=self.dtype, device=self.device) * 0.5
        }

    def test_initialization(self):
        """
        Tests the initialization and first step of the ACTController.
        """

        # Initialize ACT state using the controller
        act_state = self.controller.create_state(self.batch_shape, **self.accumulator_templates)

        # Check that the ACT state was initialized correctly
        self.assertEqual(act_state.halting_probabilities.shape, self.batch_shape)
        self.assertEqual(act_state.residual_probabilities.shape, self.batch_shape)
        self.assertEqual(act_state.has_halted.shape, self.batch_shape)
        self.assertEqual(act_state.steps_taken.shape, self.batch_shape)
        self.assertEqual(act_state.probabilistic_steps_taken.shape, self.batch_shape)

    def test_step_functionality(self):
        """
        Tests the ACTController step mechanism for updating the halting state and accumulating outputs.
        """
        # Create an embedding to initialize the ACT process
        embedding = torch.randn(*self.batch_shape, self.d_model, dtype=self.dtype, device=self.device)

        # Initialize ACT state
        state = self.controller.create_state(self.batch_shape, **self.accumulator_templates)

        # Perform a few steps to simulate computation
        for _ in range(3):
            outputs = {
                'layer1': torch.randn(*self.batch_shape, self.d_model, dtype=self.dtype, device=self.device),
                'layer2': torch.randn(*self.batch_shape, self.d_model, dtype=self.dtype, device=self.device)
            }
            act_state = self.controller(embedding, state, **outputs)

        # Ensure that halting probabilities have been updated
        self.assertTrue((act_state.halting_probabilities > 0).all())

        # Verify that outputs have been accumulated
        for key in self.accumulator_templates:
            self.assertIn(key, act_state.accumulated_outputs)

    def test_should_continue(self):
        """
        Tests the `should_continue` method to ensure that it correctly identifies when the process should halt.
        """
        # Create an embedding to initialize the ACT process
        embedding = torch.randn(*self.batch_shape, self.d_model, dtype=self.dtype, device=self.device)

        # Initialize ACT state
        state = self.controller.create_state(self.batch_shape, **self.accumulator_templates)

        # Initially, should_continue should return True since no samples have halted
        self.assertTrue(state.should_continue())

        # Simulate all samples halting
        state.has_halted[:] = True

        # Now, should_continue should return False
        self.assertFalse(state.should_continue())

    def test_finalize(self):
        """
        Tests the finalize method to ensure it completes the accumulation and provides the final results.
        """
        # Create an embedding to initialize the ACT process
        embedding = torch.randn(*self.batch_shape, self.d_model, dtype=self.dtype, device=self.device)

        # Initialize ACT state
        state = self.controller.create_state(self.batch_shape, **self.accumulator_templates)

        # Perform several steps to simulate computation
        for _ in range(3):
            outputs = {
                'layer1': torch.randn(*self.batch_shape, self.d_model, dtype=self.dtype, device=self.device),
                'layer2': torch.randn(*self.batch_shape, self.d_model, dtype=self.dtype, device=self.device)
            }
            state = self.controller(embedding, state, **outputs)

        # Simulate halting for all samples
        state.has_halted[:] = True

        # Finalize and get accumulated outputs
        final_outputs = state.finalize()

        # Ensure the final outputs have the correct shapes
        for key, accumulated in final_outputs.items():
            self.assertEqual(accumulated.shape, self.accumulator_templates[key].shape)

    def test_finalize_without_halt_raises_error(self):
        """
        Tests that finalizing without all samples having halted raises a RuntimeError.
        """
        # Create an embedding to initialize the ACT process
        embedding = torch.randn(*self.batch_shape, self.d_model, dtype=self.dtype, device=self.device)

        # Initialize ACT state
        act_state = self.controller.create_state(self.batch_shape, **self.accumulator_templates)

        # Perform some steps but do not halt all samples
        for _ in range(2):
            outputs = {
                'layer1': torch.randn(*self.batch_shape, self.d_model, dtype=self.dtype, device=self.device),
                'layer2': torch.randn(*self.batch_shape, self.d_model, dtype=self.dtype, device=self.device)
            }
            act_state = self.controller(embedding, act_state=act_state, **outputs)

        # Attempt to finalize should raise an error since not all samples have halted
        with self.assertRaises(RuntimeError):
            act_state.finalize()