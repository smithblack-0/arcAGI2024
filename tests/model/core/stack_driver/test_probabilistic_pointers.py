import torch
import unittest
from src.main.model.core.subroutine_driver import ProbabilisticPointers
class TestProbabilisticPointers(unittest.TestCase):

    def create_pointer_manager(self, initial_probabilities):
        """
        Helper function to create a fresh ProbabilisticPointers instance for each test.
        """
        return ProbabilisticPointers(initial_probabilities)


    def test_pointer_movement(self):
        """
        Tests that we can move a pointer around, superimpose it, and break it apart
        as desired.
        """
        batch_size = 2
        stack_depth = 4

        # Start condition.
        root_probability = torch.zeros([stack_depth, batch_size])
        root_actions = torch.zeros([stack_depth, batch_size, 3])

        start_probability = root_probability.clone()
        start_probability[0, ...] = 1.0
        pointer_manager = self.create_pointer_manager(start_probability)

        # Define the expectations and actions during each portion of the test
        expectations = []
        actions = []

        # Define the enstack, no op, and destack actions

        enstack_action = root_actions.clone()
        enstack_action[..., 2] = 1.0

        no_op_action = root_actions.clone()
        no_op_action[..., 1] = 1.0

        destack_action = root_actions.clone()
        destack_action[..., 0] = 1.0

        # Define the enstack situation 1 and 2

        probabilities = root_probability.clone()
        probabilities[1, ...] = 1.0

        actions.append(enstack_action)
        expectations.append(probabilities)

        probabilities = root_probability.clone()
        probabilities[2, ...] = 1.0

        actions.append(enstack_action)
        expectations.append(probabilities)

        # Define a no op. We expect things to stay as is

        action = root_actions.clone()
        action[..., 1] = 1.0

        actions.append(no_op_action)
        expectations.append(probabilities)

        # Define a destack. We expect to go down by one

        probabilities = root_probability.clone()
        probabilities[1, ...] = 1.0

        actions.append(destack_action)
        expectations.append(probabilities)

        # Finally, try smearing it out into a superposition

        action = 0.5*destack_action + 0.5*enstack_action

        probability = root_probability.clone()
        probability[0, ...] = 0.5
        probability[2, ...] = 0.5

        expectations.append(probability)
        actions.append(action)

        # Lets go check them

        for action, expectation in zip(actions, expectations):
            pointer_manager.change_superposition(action)
            pointers = pointer_manager.get()
            self.assertTrue(torch.allclose(pointers, expectation))


    def test_lost_probability(self):
        """
        Test that lost probability is correctly returned by immediately
        destacking the input
        :return:
        """
        stack_depth = 5


        start_probability =  torch.zeros([stack_depth])
        start_probability[0] = 0.6
        start_probability[1] = 0.4
        pointer_manager = self.create_pointer_manager(start_probability)

        # action

        action = torch.zeros([stack_depth, 3])
        action[..., 0] = 1.0

        # emit
        probs = pointer_manager.change_superposition(action)
        self.assertTrue(probs == 0.6)
        self.assertTrue(pointer_manager.get()[0] == 0.4)

    def test_mask_behavior(self):
        """
        Test the mask prevents pointers from wrapping around the edge of the stack
        """

        stack_depth = 5
        probability_mocks = torch.ones([stack_depth])
        pointers = self.create_pointer_manager(probability_mocks)

        # After enstacking, then destacking, edges should be zero

        enstack = torch.zeros([stack_depth, 3])
        enstack[..., 2] = 1.0

        destack = torch.zeros([stack_depth, 3])
        destack[..., 0] = 1.0

        # Do it

        pointers.change_superposition(enstack)
        probability_mocks = pointers.get()
        self.assertEqual(probability_mocks[0], 0)


        pointers.change_superposition(destack)
        probability_mocks = pointers.get()
        self.assertEqual(probability_mocks[-1], 0)



