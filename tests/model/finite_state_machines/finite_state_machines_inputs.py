import unittest
import torch
from src.main.model.finite_state_machines.finite_state_operators import (IntOperand, NotMatchingOperand,
                                                                         MatchingOperand, FSMOperator)
from src.main.CBTensors.channel_bound_tensors import CBTensor, CBTensorSpec
from src.main.model.finite_state_machines.finite_state_machine import (IntStateTrigger, MatchingCasesTrigger,
                                                                       IntakeMachine)


class TestIntStateTrigger(unittest.TestCase):

    def setUp(self):
        """
        Setup common objects used for the tests.
        """
        # Define the CBTensorSpec with some mock channels
        self.spec = CBTensorSpec({
            'state': 1,
            'mode': 1,
            'substate': 2  # Example: 'substate' has width 2
        })

        # Device to run on
        self.device = torch.device('cpu')

    def test_register_simple_operands(self):
        """
        Test match works when working in the case with one, simple, operand
        """

        # Create trigger
        trigger = IntStateTrigger(self.spec, self.device)

        # Create operators
        operators = []

        # One int state
        operand = IntOperand("state", 0, 0)
        operator = FSMOperator()
        operator.register_trigger(operand)
        trigger.register_operator(operator)

        # Another int state
        operand = IntOperand("state", 1, 0)
        operator = FSMOperator()
        operator.register_trigger(operand)
        trigger.register_operator(operator)

        # This will act as a wildcard, match everything, state
        operator = FSMOperator()
        trigger.register_operator(operator)

        # Create state

        tensor = CBTensor.create_from_channels({"state" : torch.tensor([[0], [1], [2]]),
                                                "mode" : torch.tensor([[0],[0],[0]]),
                                                "substate" : torch.randn([3, 2])
                                                })
        tensor.tensor = tensor.tensor.to(torch.long)

        result = torch.tensor([[True, False, True], [False, True, True], [False, False, True]])

        # Run test
        test = trigger(tensor)
        self.assertTrue(torch.equal(test, result))

    def test_register_multiple_int_operand_triggers(self):
        """
        Test if we still act sanely when dealing with multiple integer triggers per operator.
        """

        # Create trigger
        trigger = IntStateTrigger(self.spec, self.device)

        # Create operator with multiple triggers
        operand1 = IntOperand("state", 0, 0)
        operand2 = IntOperand("mode", 0, 0)
        operator = FSMOperator()
        operator.register_trigger(operand1)
        operator.register_trigger(operand2)
        trigger.register_operator(operator)

        # Another operator
        operand1 = IntOperand("state", 1, 0)
        operand2 = IntOperand("mode", 0, 0)
        operator = FSMOperator()
        operator.register_trigger(operand1)
        operator.register_trigger(operand2)
        trigger.register_operator(operator)

        # Create state tensor
        tensor = CBTensor.create_from_channels({"state": torch.tensor([[0], [1], [2]]),
                                                "mode": torch.tensor([[0], [0], [1]]),
                                                "substate": torch.randn([3, 2])
                                                })
        tensor.tensor = tensor.tensor.to(torch.long)

        # Expected result - only the first two entries match the corresponding states
        result = torch.tensor([[True, False], [False, True], [False, False]])

        # Run the test
        test = trigger(tensor)
        self.assertTrue(torch.equal(test, result))

class TestMatchingCasesTrigger(unittest.TestCase):

    def setUp(self):
        """
        Setup common objects used for the tests.
        """
        # Define the CBTensorSpec with mock channels and widths. Ensure equal widths for matching.
        self.spec = CBTensorSpec({
            'state': 1,
            'mode': 1,
            'substate': 1  # Adjust 'substate' to match the width of 'state' and 'mode'
        })

        # Device to run on
        self.device = torch.device('cpu')

    def test_register_simple_matching_operands(self):
        """
        Test simple matching of operands for matching cases.
        """

        # Create the MatchingCasesTrigger
        trigger = MatchingCasesTrigger(self.spec, self.device)

        # Define matching operands for state and mode
        operand1 = MatchingOperand("state", "mode")
        operator = FSMOperator()
        operator.register_trigger(operand1)
        trigger.register_operator(operator)

        # Create a state tensor that should match this condition
        tensor = CBTensor.create_from_channels({
            "state": torch.tensor([[0], [1], [2]]),
            "mode": torch.tensor([[0], [1], [2]]),
            "substate": torch.randn([3, 1])  # Substate is now of width 1
        })

        # Expected result: State and mode match, so we should get True for all rows
        expected_result = torch.tensor([[True], [True], [True]])

        # Run test
        result = trigger(tensor)
        self.assertTrue(torch.equal(result, expected_result))

    def test_register_non_matching_operands(self):
        """
        Test simple non-matching of operands using NotMatchingOperands.
        """
        # Create the MatchingCasesTrigger
        trigger = MatchingCasesTrigger(self.spec, self.device)

        # Define a non-matching condition: state != mode
        operand1 = NotMatchingOperand("state", "mode")
        operator = FSMOperator()
        operator.register_trigger(operand1)
        trigger.register_operator(operator)

        # Create a state tensor where state and mode are equal
        tensor = CBTensor.create_from_channels({
            "state": torch.tensor([[0], [1], [2]]),
            "mode": torch.tensor([[0], [1], [2]]),
            "substate": torch.randn([3, 1])
        })

        # Expected result: No match because state == mode, so all rows should be False
        expected_result = torch.tensor([[False], [False], [False]])

        # Run test
        result = trigger(tensor)
        self.assertTrue(torch.equal(result, expected_result))

    def test_register_mixed_matching_and_non_matching(self):
        """
        Test matching and non-matching conditions combined.
        """
        # Create the MatchingCasesTrigger
        trigger = MatchingCasesTrigger(self.spec, self.device)

        # Define a matching operand for the first operator
        operand1 = MatchingOperand("state", "mode")
        operator = FSMOperator()
        operator.register_trigger(operand1)
        trigger.register_operator(operator)

        # Define a non-matching operand for the second operator
        operand2 = NotMatchingOperand("state", "mode")
        operator2 = FSMOperator()
        operator2.register_trigger(operand2)
        trigger.register_operator(operator2)

        # Create a state tensor
        tensor = CBTensor.create_from_channels({
            "state": torch.tensor([[0], [1], [2]]),
            "mode": torch.tensor([[0], [0], [1]]),
            "substate": torch.randn([3, 1])
        })

        # Expected result:
        # - First row should match the first operator (state == mode)
        # - Second row should match neither (no trigger)
        # - Third row should match the second operator (state != mode)
        expected_result = torch.tensor([[True, False], [False, True], [False, True]])

        # Run test
        result = trigger(tensor)
        self.assertTrue(torch.equal(result, expected_result))

    def test_multiple_matching_conditions(self):
        """
        Test multiple channels being compared and triggering conditions.
        """

        # Create the MatchingCasesTrigger
        trigger = MatchingCasesTrigger(self.spec, self.device)

        # Define matching for state == mode and substate != mode
        operand1 = MatchingOperand("state", "mode")
        operand2 = NotMatchingOperand("substate", "mode")
        operator = FSMOperator()
        operator.register_trigger(operand1)
        operator.register_trigger(operand2)
        trigger.register_operator(operator)

        # Create a state tensor
        tensor = CBTensor.create_from_channels({
            "state": torch.tensor([[0], [1], [2]]),
            "mode": torch.tensor([[0], [0], [1]]),
            "substate": torch.tensor([[1], [0], [2]])  # Substate now matches width of state and mode
        })

        # Expected result:
        # - First row matches (state == mode, substate != mode)
        # - Second and third rows don't match due to substate == mode
        expected_result = torch.tensor([[True], [False], [False]])

        # Run test
        result = trigger(tensor)
        self.assertTrue(torch.equal(result, expected_result))

class TestIntakeMachine(unittest.TestCase):
    def setUp(self):
        """
        Setup common objects used for the tests.
        """
        # Define the CBTensorSpec with some mock channels
        self.spec = CBTensorSpec({
            'state': 1,
            'mode': 1,
            'substate': 1  # Example: 'substate' has width 2
        })
        self.device = torch.device('cpu')

    def test_intake_machine_with_int_patterns(self):
        """
        Test if the IntakeMachine correctly evaluates FSM operators with integer operands.
        """
        # Define FSM operators
        operators = []

        # Operator 1: Match on state=0
        operator_1 = FSMOperator()
        operator_1.register_trigger(IntOperand(channel="state", position=0, value=0))
        operators.append(operator_1)

        # Operator 2: Match on state=1
        operator_2 = FSMOperator()
        operator_2.register_trigger(IntOperand(channel="state", position=0, value=1))
        operators.append(operator_2)

        # Operator 3: Match on state=2, mode=1
        operator_3 = FSMOperator()
        operator_3.register_trigger(IntOperand(channel="state", position=0, value=2))
        operator_3.register_trigger(IntOperand(channel="mode", position=0, value=1))
        operators.append(operator_3)

        # Create an intake machine with these operators
        intake_machine = IntakeMachine(operators, self.spec, self.device)

        # Create state tensor
        state_tensor = CBTensor.create_from_channels({
            "state": torch.tensor([[0], [1], [2]]),
            "mode": torch.tensor([[0], [0], [1]]),
            "substate": torch.randn([3, 1])  # Random values for 'substate'
        })
        state_tensor.tensor = state_tensor.tensor.to(torch.long)

        # Expected result:
        # - First row should trigger operator 1 (state=0)
        # - Second row should trigger operator 2 (state=1)
        # - Third row should trigger operator 3 (state=2, mode=1)
        expected_result = torch.tensor([0, 1, 2])

        # Run the intake machine
        result = intake_machine(state_tensor)

        # Assert that the result matches the expected output
        self.assertTrue(torch.equal(result, expected_result))

    def test_intake_machine_with_both_patterns(self):
        """
        Test an intake machine that may have trigger dependencies on int state,
        or on matching or lack of matching of mode
        """

        # Define FSM operator
        operators = []

        # State is 0, trigger
        operator = FSMOperator()
        operator.register_trigger(IntOperand(channel="state", position=0, value=0))
        operators.append(operator)

        # State is 1, mode matches state
        operator = FSMOperator()
        operator.register_trigger(IntOperand(channel="state", position=0, value=1))
        operator.register_trigger(MatchingOperand(channel="mode", target_channel="state"))
        operators.append(operator)

        # State is 1, mode does NOT match state
        operator = FSMOperator()
        operator.register_trigger(IntOperand(channel="state", position=0, value=1))
        operator.register_trigger(NotMatchingOperand(channel="mode", target_channel="state"))
        operators.append(operator)

        # Define machine
        machine = IntakeMachine(operators, self.spec, self.device)

        # Create state tensor
        state_tensor = CBTensor.create_from_channels({
            "state": torch.tensor([[0], [1], [1]], dtype=torch.int64),
            "mode": torch.tensor([[0], [1], [2]], dtype=torch.int64),
            "substate": torch.randn([3, 1])  # Random values for 'substate'
        })

        # Create expected output
        expected_output = torch.tensor([0, 1, 2])

        # Test
        output = machine(state_tensor)
        self.assertTrue(torch.equal(output, expected_output))

    def test_intake_machine_error_on_multiple_triggers(self):
        """
        Test if the IntakeMachine raises an error when more than one operator is triggered at the same time.
        This will happen, for example, with two wildcard operators.
        """
        # Define FSM operators
        operators = []

        # Operator 1: A wildcard operator (matches any state/mode/substate)
        operator_1 = FSMOperator()
        operators.append(operator_1)

        # Operator 2: Another wildcard operator (also matches any state/mode/substate)
        operator_2 = FSMOperator()
        operators.append(operator_2)

        # Create an intake machine with these operators
        intake_machine = IntakeMachine(operators, self.spec, self.device)

        # Create state tensor (arbitrary values, since both operators match anything)
        state_tensor = CBTensor.create_from_channels({
            "state": torch.tensor([[0], [1], [2]]),
            "mode": torch.tensor([[0], [0], [1]]),
            "substate": torch.randn([3, 2])  # Random values for 'substate'
        })
        state_tensor.tensor = state_tensor.tensor.to(torch.long)

        # Run the intake machine and expect a ValueError due to multiple triggered states
        with self.assertRaises(ValueError):
            intake_machine(state_tensor)

    def test_intake_machine_multidimensional_batches(self):
        """
        Test if the IntakeMachine handles multidimensional inputs correctly.
        """
        # Define FSM operators
        operators = []

        # Operator 1: Match on state=0
        operator_1 = FSMOperator()
        operator_1.register_trigger(IntOperand(channel="state", position=0, value=0))
        operators.append(operator_1)

        # Operator 2: Match on state=1
        operator_2 = FSMOperator()
        operator_2.register_trigger(IntOperand(channel="state", position=0, value=1))
        operators.append(operator_2)

        # Create an intake machine with these operators
        intake_machine = IntakeMachine(operators, self.spec, self.device)

        # Create a multidimensional state tensor
        state_tensor = CBTensor.create_from_channels({
            "state": torch.randint(0, 1, [20, 10, 3, 4, 1]),  # 2x2 tensor
            "mode": torch.randint(0, 1, [20, 10, 3, 4, 1]),   # 2x2 tensor
            "substate": torch.randn([20, 10, 3, 4, 1])  # Random values for 'substate'
        })

        # Run the intake machine
        result = intake_machine(state_tensor)

        self.assertTrue(result.shape== (20, 10, 3, 4))