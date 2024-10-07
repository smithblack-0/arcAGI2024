import torch
import unittest
from src.main.model.core.subroutine_driver import ActionsManagement
class TestActionsManagement(unittest.TestCase):

    def create_action_manager(self, num_times_before_pop, num_times_before_flush, stack_size, statistics):
        """
        Helper function to create a fresh ActionsManagement instance for each test.
        """
        return ActionsManagement(num_times_before_pop, num_times_before_flush, stack_size, statistics)

    def test_runs_at_all(self):
        """
        Verifies that subroutine initiation logits result in appropriate action probabilities.
        Basically just checks if we run at all.
        """
        stack_size = 5
        batch_size = 7
        statistics = torch.zeros([stack_size, batch_size, 3])

        action_manager = self.create_action_manager(0, 10, stack_size, statistics)
        action_logits = torch.randn([batch_size, 3])  # Simulating random logits
        probabilities = action_manager(action_logits)


        self.assertTrue(torch.all(probabilities >= 0), "Probabilities should be non-negative")
        self.assertTrue(torch.allclose(torch.tensor(1.0), probabilities.sum(dim=-1)), "Probabilities should sum to 1")
        self.assertTrue(probabilities.shape == (stack_size, batch_size, 3))

    def test_destack_logic(self):
        """
        Test the destack action is triggered when apppropriate.
        Also, test if we block destacking when not ready to release.
        """

        stack_size = 5
        batch_size = 7
        statistics = torch.zeros([stack_size, batch_size, 3])
        action_manager = self.create_action_manager(1, 10, stack_size, statistics)

        # Set to exist only in the destack state.
        action_logits = torch.zeros([batch_size, 3])
        action_logits[..., 0] = 1e10
        action_logits[..., 1] = -1e10
        action_logits[..., 2] = -1e10

        probabilities = action_manager(action_logits)

        # Test if we are in a destack config, and if the destack config is not
        # letting us pop
        self.assertTrue(torch.all((probabilities[1:, ..., 0] - 1.0).abs() < 1e-4))
        self.assertTrue(torch.all((probabilities[0, ..., 0]).abs() < 1e-4))
        self.assertTrue(torch.all(action_manager.action_statistics == probabilities))

        # Test if we now will allow that pop
        probabilities = action_manager(action_logits)
        self.assertTrue(torch.all((probabilities[0, ..., 0]-1.0).abs() < 1e-4))

    def test_enstack_logic(self):
        """
        Test the enstack logic. This includes verifying
        we cannot go off the end of the stack, and that after
        flush is reached enstack probability is zero.
        """

        stack_size = 5
        batch_size = 7
        statistics = torch.zeros([stack_size, batch_size, 3])
        action_manager = self.create_action_manager(0, 1, stack_size, statistics)

        # Set to exist only in the enstack state.
        action_logits = torch.zeros([batch_size, 3])
        action_logits[..., 0] = -1e10
        action_logits[..., 1] = -1e10
        action_logits[..., 2] = 1e10

        # Get probabilities. Verify we are mostly in the enstack state, except
        # for the last stack level, which can not be in that state

        probabilities = action_manager(action_logits)

        self.assertTrue(torch.all((probabilities[:-1, ..., 2] - 1.0).abs() < 1e-4))
        self.assertTrue(torch.all((probabilities[-1, ..., 2]).abs() < 1e-4))
        self.assertTrue(torch.all(action_manager.action_statistics == probabilities))

        # We are now in forced flush mode. We should no longer be in enstack state.
        # but in destack

        probabilities = action_manager(action_logits)

        self.assertTrue(torch.all((probabilities[:-1, ..., 2]).abs() < 1e-4))
        self.assertTrue(torch.all((probabilities[-1, ..., 2]).abs() < 1e-4))

    def test_no_op_logic(self):
        """
        Test the no op logic.
        """
        stack_size = 5
        batch_size = 7
        statistics = torch.zeros([stack_size, batch_size, 3])
        action_manager = self.create_action_manager(0, 1, stack_size, statistics)

        # Set to exist only in the no op state.
        action_logits = torch.zeros([batch_size, 3])
        action_logits[..., 0] = -1e10
        action_logits[..., 1] = 1e10
        action_logits[..., 2] = -1e10

        # Get probabilities. Verify we are mostly in the no op state, except
        # for the last stack level, which can not be in that state

        probabilities = action_manager(action_logits)

        self.assertTrue(torch.all((probabilities[ ..., 1] - 1.0).abs() < 1e-4))
        self.assertTrue(torch.all(action_manager.action_statistics == probabilities))

        # We are now in forced flush mode. We should no longer be in no op state.
        # but in destack

        probabilities = action_manager(action_logits)

        self.assertTrue(torch.all((probabilities[..., 1]).abs() < 1e-4))