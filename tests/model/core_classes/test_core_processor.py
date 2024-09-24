import unittest
from unittest.mock import MagicMock
from typing import List, Tuple
import torch
from torch.nn import functional as F
from src.old.core_classes_old.core_processer import (BatchAssembly, BatchDisassembly,
                                                     CoreSyncProcessor, ContractedModule)

### Batch Assembly testing ###

class MockBatchAssembly(BatchAssembly):
    """
    A mocked-up version of a batch assembly
    device. This one will expect to see 1d
    tensors in channels, and will pad those to a common
    length then return them, plus the unpadded length
    """
    def make_batch(self,
                   name: str,
                   cases: List[torch.Tensor]
                   ) ->Tuple[torch.Tensor, torch.Tensor]:
        """
        Creates the batch. Expects a list of 1d tensors.
        Returns batch, and case.

        :param name: The name of the channel
        :param cases: The cases across the differing batches.
        :return: The batch, and the shapes
        """
        shape_length = [case.shape[0] for case in cases]
        max_length = max(shape_length)
        cases = [F.pad(case, (0, max_length - case.shape[0])) for case in cases]
        batch = torch.stack(cases, dim=0)
        shapes = torch.tensor([[shape] for shape in shape_length])
        return batch, shapes

class TestBatchAssembly(unittest.TestCase):
    """
    Test for batch assembly, unit tests
    """
    def test_success(self):
        # Input cases.
        channels = ["channel1", "channel2"]
        cases_backend = {"datacase1" : {"channel1" : torch.tensor([2.0, 3.0]), "channel2" : torch.tensor([1.0])},
                         "datacase2" : {"channel1" : torch.tensor([1.0]), "channel2" : torch.tensor([2.0, 2.0, 2.0])},
                         "datacase3" : {"channel1" : torch.tensor([0.0]), "channel2" : torch.tensor([3.0])}
                         }
        selection = ["datacase1", "datacase2"]

        logging_callback = MagicMock()
        termination_callback = MagicMock()

        # Expectations
        selected_output_uuids = ["datacase1", "datacase2"]
        expected_exception_info = {}
        expected_batch = {"channel1" : torch.tensor([[2.0, 3.0],[1.0, 0.0]]),
                 "channel2" : torch.tensor([[1.0, 0.0, 0.0],[2.0, 2.0, 2.0]])
                 }
        expected_shapes = {"channel1" : torch.tensor([[2], [1]]), "channel2" : torch.tensor([[1], [3]])}

        # Run test
        assembler = MockBatchAssembly(channels)
        actual_output_uuids, actual_exception_info, actual_batch, actual_shapes = assembler(selection,
                                                                                            cases_backend,
                                                                                            logging_callback,
                                                                                            termination_callback)

        # Check stateful manipulation is sane
        self.assertNotIn("datacase1", cases_backend)
        self.assertNotIn("datacase2", cases_backend)
        self.assertIn("datacase3", cases_backend)

        # Check values
        self.assertEqual(selected_output_uuids, actual_output_uuids)
        self.assertEqual(actual_exception_info, expected_exception_info)
        self.assertEqual(set(actual_batch.keys()), set(expected_batch.keys()))
        self.assertEqual(set(actual_shapes.keys()), set(expected_batch.keys()))
        for key in expected_batch.keys():
            self.assertTrue(torch.all(actual_batch[key] == expected_batch[key]))
            self.assertTrue(torch.all(actual_shapes[key] == expected_shapes[key]))

        print(logging_callback.mock_calls)

class TestBatchAssemblyErrorConditions(unittest.TestCase):
    """
    Additional unit tests for various error conditions in batch assembly.
    """
    def check_for_logged_exception(self,
                                   mock: MagicMock,
                                   exception: Exception,
                                   severity: int
                                   )->bool:
        """
        Checks if a magic mock logging callback was invoked
        with the specified exception, severity pair
        :param mock: The magic mock
        :param exception: The exception we expect
        :param severity: The severity we expect
        :return: If it was seen.
        """
        for call in mock.mock_calls:
            arg1, arg2 = call.args
            if isinstance(arg1, exception) and arg2 == severity:
                return True
        return False
    def test_missing_uuid_in_case_buffer(self):
        channels = ["channel1", "channel2"]
        cases_backend = {"datacase1": {"channel1": torch.tensor([2.0, 3.0]), "channel2": torch.tensor([1.0])}}
        selection = ["datacase1", "datacase2"]  # "datacase2" does not exist

        logging_callback = MagicMock()
        termination_callback = MagicMock()

        assembler = MockBatchAssembly(channels)
        actual_output_uuids, actual_exception_info, actual_batch, actual_shapes = assembler(
            selection, cases_backend, logging_callback, termination_callback
        )

        self.assertEqual(actual_output_uuids, ["datacase1"])
        self.assertIn("datacase2", actual_exception_info)
        self.assertTrue(self.check_for_logged_exception(logging_callback, KeyError, 1))

        # Ensure dictionary side effects
        self.assertEqual(cases_backend, {})

    def test_missing_channel_in_case(self):
        channels = ["channel1", "channel2"]
        cases_backend = {"datacase1": {"channel1": torch.tensor([2.0, 3.0])},  # "channel2" is missing
                         "datacase2": {"channel1": torch.tensor([1.0]), "channel2": torch.tensor([2.0, 2.0, 2.0])}}

        selection = ["datacase1", "datacase2"]

        logging_callback = MagicMock()
        termination_callback = MagicMock()

        assembler = MockBatchAssembly(channels)
        actual_output_uuids, actual_exception_info, actual_batch, actual_shapes = assembler(
            selection, cases_backend, logging_callback, termination_callback
        )

        self.assertEqual(actual_output_uuids, ["datacase2"])
        self.assertIn("datacase1", actual_exception_info)
        self.assertTrue(self.check_for_logged_exception(logging_callback, KeyError, 1))

        # Ensure dictionary side effects
        self.assertEqual(cases_backend, {})

    def test_extra_channel_in_case(self):
        # Inputs
        channels = ["channel1", "channel2"]
        cases_backend = {"datacase1": {"channel1": torch.tensor([2.0, 3.0]), "channel2": torch.tensor([1.0]), "channel3": torch.tensor([4.0])},
                         "datacase2": {"channel1": torch.tensor([1.0]), "channel2": torch.tensor([2.0, 2.0, 2.0])}}

        selection = ["datacase1", "datacase2"]

        logging_callback = MagicMock()
        termination_callback = MagicMock()

        assembler = MockBatchAssembly(channels)
        actual_output_uuids, actual_exception_info, actual_batch, actual_shapes = assembler(
            selection, cases_backend, logging_callback, termination_callback
        )

        self.assertEqual(actual_output_uuids, ["datacase2"])
        self.assertIn("datacase1", actual_exception_info)
        self.assertTrue(self.check_for_logged_exception(logging_callback, KeyError, 1))

        # Ensure dictionary side effects. Bad data should have been rmoved
        self.assertEqual(cases_backend, {})

    def test_mismatched_batch_size(self):
        channels = ["channel1", "channel2"]
        cases_backend = {"datacase1": {"channel1": torch.tensor([2.0, 3.0]), "channel2": torch.tensor([1.0])},
                         "datacase2": {"channel1": torch.tensor([1.0]), "channel2": torch.tensor([2.0, 2.0, 2.0])}}

        selection = ["datacase1", "datacase2"]

        original_cases_backend = cases_backend.copy()

        # Mock the batch assembly to return a mismatched batch size
        class MockBatchAssemblyWithError(MockBatchAssembly):
            def make_batch(self, name: str, cases: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
                batch, shape = super().make_batch(name, cases)
                return batch[:-1], shape  # Returning mismatched batch

        logging_callback = MagicMock()
        termination_callback = MagicMock()

        assembler = MockBatchAssemblyWithError(channels)
        with self.assertRaises(RuntimeError):
            assembler(selection, cases_backend, logging_callback, termination_callback)

        self.assertTrue(self.check_for_logged_exception(logging_callback, RuntimeError, 0))
        termination_callback.assert_called_once_with(True)

        # Ensure dictionary side effects
        self.assertEqual(cases_backend, {})

## Batch Disassembly
#
# Try batch dissassembly mechanisms

class TestBatchDisassembly(unittest.TestCase):

    def test_success_disassembly(self):
        # Everything has been assumed to have completed successfully by this point

        # Inputs
        run_uuids = ["dataset1", "dataset2"]
        exception_data = {}
        batch = {"channel1": torch.tensor([[2.0, 3.0], [1.0, 0.0]]),
                 "channel2": torch.tensor([[1.0, 0.0, 0.0], [2.0, 2.0, 2.0]])}
        shapes = {"channel1": torch.tensor([[2], [1]]), "channel2": torch.tensor([[1], [3]])}  # 2D tensors

        logging_callback = MagicMock()
        termination_callback = MagicMock()

        # Expected outputs
        expected_output = {}
        expected_output["dataset1"] = (
            {"channel1": torch.tensor([2]), "channel2": torch.tensor([1])},
            {"channel1": torch.tensor([2.0, 3.0]), "channel2": torch.tensor([1.0, 0.0, 0.0])},
        )
        expected_output["dataset2"] = (
            {"channel1": torch.tensor([1]), "channel2": torch.tensor([3])},
            {"channel1": torch.tensor([1.0, 0.0]), "channel2": torch.tensor([2.0, 2.0, 2.0])}
        )

        # Run
        disassembler = BatchDisassembly()
        outcome = disassembler(run_uuids, exception_data, shapes, batch, logging_callback, termination_callback)

        # Validation
        self.assertEqual(set(expected_output.keys()), set(outcome.keys()))

        for uuid in run_uuids:
            self.assertTrue(torch.equal(expected_output[uuid][0]["channel1"], outcome[uuid][0]["channel1"]))
            self.assertTrue(torch.equal(expected_output[uuid][0]["channel2"], outcome[uuid][0]["channel2"]))
            self.assertTrue(torch.equal(expected_output[uuid][1]["channel1"], outcome[uuid][1]["channel1"]))
            self.assertTrue(torch.equal(expected_output[uuid][1]["channel2"], outcome[uuid][1]["channel2"]))

        print(logging_callback.mock_calls)

    def test_failing_reassembly(self):
        # Simulate a case where an error occurred for one of the datasets

        # Inputs
        run_uuids = ["dataset1"]  # Only dataset1 was successfully processed
        exception_data = {"dataset2": RuntimeError("Processing error")}
        batch = {"channel1": torch.tensor([[2.0, 3.0]]),  # Only dataset1 data present
                 "channel2": torch.tensor([[1.0, 0.0, 0.0]])}
        shapes = {"channel1": torch.tensor([[2]]), "channel2": torch.tensor([[1]])}  # 2D tensors

        logging_callback = MagicMock()
        termination_callback = MagicMock()

        # Expected outputs
        expected_output = {}
        expected_output["dataset1"] = (
            {"channel1": torch.tensor([2]), "channel2": torch.tensor([1])},
            {"channel1": torch.tensor([2.0, 3.0]), "channel2": torch.tensor([1.0, 0.0, 0.0])},
        )
        expected_output["dataset2"] = exception_data["dataset2"]

        # Run
        disassembler = BatchDisassembly()
        outcome = disassembler(run_uuids, exception_data, shapes, batch, logging_callback, termination_callback)

        # Validation
        self.assertEqual(set(expected_output.keys()), set(outcome.keys()))

        for uuid in run_uuids:
            self.assertTrue(torch.equal(expected_output[uuid][0]["channel1"], outcome[uuid][0]["channel1"]))
            self.assertTrue(torch.equal(expected_output[uuid][0]["channel2"], outcome[uuid][0]["channel2"]))
            self.assertTrue(torch.equal(expected_output[uuid][1]["channel1"], outcome[uuid][1]["channel1"]))
            self.assertTrue(torch.equal(expected_output[uuid][1]["channel2"], outcome[uuid][1]["channel2"]))

        # Ensure that the exception was handled correctly
        self.assertIsInstance(outcome["dataset2"], RuntimeError)
        self.assertEqual(str(outcome["dataset2"]), "Processing error")

        print(logging_callback.mock_calls)

    def test_assembler_integration(self):
        # Use the assembler's output as input to the disassembler

        # Setup the batch assembler
        channels = ["channel1", "channel2"]
        cases_backend = {"dataset1": {"channel1": torch.tensor([2.0, 3.0]), "channel2": torch.tensor([1.0])},
                         "dataset2": {"channel1": torch.tensor([1.0]), "channel2": torch.tensor([2.0, 2.0, 2.0])}}
        selection = ["dataset1", "dataset2"]

        logging_callback = MagicMock()
        termination_callback = MagicMock()

        assembler = MockBatchAssembly(channels)
        used_ids, exception_info, batch, shapes = assembler(selection, cases_backend, logging_callback,
                                                            termination_callback)

        # Expected outputs
        expected_output = {}
        expected_output["dataset1"] = (
            {"channel1": torch.tensor([2]), "channel2": torch.tensor([1])},
            {"channel1": torch.tensor([2.0, 3.0]), "channel2": torch.tensor([1.0, 0.0, 0.0])},
        )
        expected_output["dataset2"] = (
            {"channel1": torch.tensor([1]), "channel2": torch.tensor([3])},
            {"channel1": torch.tensor([1.0, 0.0]), "channel2": torch.tensor([2.0, 2.0, 2.0])}
        )

        # Run disassembly
        disassembler = BatchDisassembly()
        outcome = disassembler(used_ids, exception_info, shapes, batch, logging_callback, termination_callback)

        # Validation
        self.assertEqual(set(expected_output.keys()), set(outcome.keys()))

        for uuid in used_ids:
            self.assertTrue(torch.equal(expected_output[uuid][0]["channel1"], outcome[uuid][0]["channel1"]))
            self.assertTrue(torch.equal(expected_output[uuid][0]["channel2"], outcome[uuid][0]["channel2"]))
            self.assertTrue(torch.equal(expected_output[uuid][1]["channel1"], outcome[uuid][1]["channel1"]))
            self.assertTrue(torch.equal(expected_output[uuid][1]["channel2"], outcome[uuid][1]["channel2"]))

        print(logging_callback.mock_calls)


class TestBatchDisassemblyErrorConditions(unittest.TestCase):
    """
    Unit tests for various error conditions in batch disassembly.
    """

    def check_for_logged_exception(self,
                                   mock: MagicMock,
                                   exception: Exception,
                                   severity: int
                                   ) -> bool:
        """
        Checks if a magic mock logging callback was invoked
        with the specified exception, severity pair
        :param mock: The magic mock
        :param exception: The exception we expect
        :param severity: The severity we expect
        :return: If it was seen.
        """
        for call in mock.mock_calls:
            arg1, arg2 = call.args
            if isinstance(arg1, exception) and arg2 == severity:
                return True
        return False

    def test_shape_batch_size_mismatch(self):
        # Test case where the shape tensor batch size does not match the run_uuids
        run_uuids = ["dataset1", "dataset2"]
        exception_data = {}
        batch = {"channel1": torch.tensor([[2.0, 3.0], [1.0, 0.0]]),
                 "channel2": torch.tensor([[1.0, 0.0, 0.0], [2.0, 2.0, 2.0]])}
        shapes = {"channel1": torch.tensor([[2]]), "channel2": torch.tensor([[1], [3]])}  # Mismatched shape size

        logging_callback = MagicMock()
        termination_callback = MagicMock()

        disassembler = BatchDisassembly()

        with self.assertRaises(RuntimeError):
            disassembler(run_uuids, exception_data, shapes, batch, logging_callback, termination_callback)

        self.assertTrue(self.check_for_logged_exception(logging_callback, RuntimeError, 0))
        termination_callback.assert_called_once_with(True)

    def test_shape_tensor_dimensionality_error(self):
        # Test case where the shape tensor does not have 2 dimensions
        run_uuids = ["dataset1", "dataset2"]
        exception_data = {}
        batch = {"channel1": torch.tensor([[2.0, 3.0], [1.0, 0.0]]),
                 "channel2": torch.tensor([[1.0, 0.0, 0.0], [2.0, 2.0, 2.0]])}
        shapes = {"channel1": torch.tensor([2, 1]), "channel2": torch.tensor([[1], [3]])}  # 1D tensor for channel1

        logging_callback = MagicMock()
        termination_callback = MagicMock()

        disassembler = BatchDisassembly()

        with self.assertRaises(RuntimeError):
            disassembler(run_uuids, exception_data, shapes, batch, logging_callback, termination_callback)

        self.assertTrue(self.check_for_logged_exception(logging_callback, RuntimeError, 0))
        termination_callback.assert_called_once_with(True)

    def test_model_output_batch_size_mismatch(self):
        # Test case where the main output tensor batch size does not match the run_uuids
        run_uuids = ["dataset1", "dataset2"]
        exception_data = {}
        batch = {"channel1": torch.tensor([[2.0, 3.0], [1.0, 0.0]]),
                 "channel2": torch.tensor([[1.0, 0.0, 0.0], [2.0, 2.0, 2.0]])}
        shapes = {"channel1": torch.tensor([[2], [1]]), "channel2": torch.tensor([[1], [3]])}
        model_output = {"channel1": torch.tensor([[2.0, 3.0]]),  # Mismatched size (only one item)
                        "channel2": torch.tensor([[1.0, 0.0, 0.0], [2.0, 2.0, 2.0]])}

        logging_callback = MagicMock()
        termination_callback = MagicMock()

        disassembler = BatchDisassembly()

        with self.assertRaises(RuntimeError):
            disassembler(run_uuids, exception_data, shapes, model_output, logging_callback, termination_callback)

        self.assertTrue(self.check_for_logged_exception(logging_callback, RuntimeError, 0))
        termination_callback.assert_called_once_with(True)


class TestCoreSyncProcessor(unittest.TestCase):

    def test_successful_processing(self):
        # Test for successful end-to-end processing through the CoreSyncProcessor

        channels = ["channel1", "channel2"]
        cases_backend = {"dataset1": {"channel1": torch.tensor([2.0, 3.0]), "channel2": torch.tensor([1.0])},
                         "dataset2": {"channel1": torch.tensor([1.0]), "channel2": torch.tensor([2.0, 2.0, 2.0])}}
        selection = ["dataset1", "dataset2"]

        logging_callback = MagicMock()
        termination_callback = MagicMock()

        # Mock implementations
        assembler = MockBatchAssembly(channels)
        model = MagicMock(spec=ContractedModule)
        model.return_value = {"channel1": torch.tensor([[2.0, 3.0], [1.0, 0.0]]),
                              "channel2": torch.tensor([[1.0, 0.0, 0.0], [2.0, 2.0, 2.0]])}
        disassembler = BatchDisassembly()

        processor = CoreSyncProcessor(assembler, model, disassembler)
        case_buffer = cases_backend.copy()  # Use the same cases backend as the buffer

        outcome = processor(selection, case_buffer, logging_callback, termination_callback)

        # Expected outputs
        expected_output = {
            "dataset1": (
                {"channel1": torch.tensor([2]), "channel2": torch.tensor([1])},
                {"channel1": torch.tensor([2.0, 3.0]), "channel2": torch.tensor([1.0, 0.0, 0.0])}
            ),
            "dataset2": (
                {"channel1": torch.tensor([1]), "channel2": torch.tensor([3])},
                {"channel1": torch.tensor([1.0, 0.0]), "channel2": torch.tensor([2.0, 2.0, 2.0])}
            )
        }

        # Validation
        self.assertEqual(set(expected_output.keys()), set(outcome.keys()))

        for uuid in selection:
            self.assertTrue(torch.equal(expected_output[uuid][0]["channel1"], outcome[uuid][0]["channel1"]))
            self.assertTrue(torch.equal(expected_output[uuid][0]["channel2"], outcome[uuid][0]["channel2"]))
            self.assertTrue(torch.equal(expected_output[uuid][1]["channel1"], outcome[uuid][1]["channel1"]))
            self.assertTrue(torch.equal(expected_output[uuid][1]["channel2"], outcome[uuid][1]["channel2"]))

        print(logging_callback.mock_calls)

    def test_partial_processing(self):
        # Test where one of the datasets encounters an exception during disassembly

        channels = ["channel1", "channel2"]
        cases_backend = {"dataset1": {"channel1": torch.tensor([2.0, 3.0]), "channel2": torch.tensor([1.0])},
                         "dataset2": {"channel1": torch.tensor([1.0])}}
        selection = ["dataset1", "dataset2"]

        logging_callback = MagicMock()
        termination_callback = MagicMock()

        # Mock implementations
        assembler = MockBatchAssembly(channels)
        model = MagicMock(spec=ContractedModule)
        model.return_value = {"channel1": torch.tensor([[2.0, 3.0]]),
                              "channel2": torch.tensor([[1.0, 0.0, 0.0]])}
        disassembler = BatchDisassembly()

        processor = CoreSyncProcessor(assembler, model, disassembler)
        case_buffer = cases_backend.copy()  # Use the same cases backend as the buffer

        # Intentionally insert an exception for dataset2
        exception_data = {"dataset2": RuntimeError("Processing error")}

        outcome = processor(selection, case_buffer, logging_callback, termination_callback)

        # Expected outputs
        expected_output = {
            "dataset1": (
                {"channel1": torch.tensor([2]), "channel2": torch.tensor([1])},
                {"channel1": torch.tensor([2.0, 3.0]), "channel2": torch.tensor([1.0, 0.0, 0.0])}
            ),
            "dataset2": exception_data["dataset2"]
        }

        # Validation
        self.assertEqual(set(expected_output.keys()), set(outcome.keys()))

        for uuid in selection:
            if uuid in exception_data:
                self.assertIsInstance(outcome[uuid], KeyError)
            else:
                self.assertTrue(torch.equal(expected_output[uuid][0]["channel1"], outcome[uuid][0]["channel1"]))
                self.assertTrue(torch.equal(expected_output[uuid][0]["channel2"], outcome[uuid][0]["channel2"]))
                self.assertTrue(torch.equal(expected_output[uuid][1]["channel1"], outcome[uuid][1]["channel1"]))
                self.assertTrue(torch.equal(expected_output[uuid][1]["channel2"], outcome[uuid][1]["channel2"]))

        print(logging_callback.mock_calls)


class TestCoreSyncProcessorErrorConditions(unittest.TestCase):

    def check_for_logged_exception(self,
                                   mock: MagicMock,
                                   exception: Exception,
                                   severity: int
                                   ) -> bool:
        """
        Checks if a magic mock logging callback was invoked
        with the specified exception, severity pair
        :param mock: The magic mock
        :param exception: The exception we expect
        :param severity: The severity we expect
        :return: If it was seen.
        """
        for call in mock.mock_calls:
            arg1, arg2 = call.args
            if isinstance(arg1, exception) and arg2 == severity:
                return True
        return False
    def test_model_processing_exception(self):
        # Test where the main processing raises an exception

        channels = ["channel1", "channel2"]
        cases_backend = {"dataset1": {"channel1": torch.tensor([2.0, 3.0]), "channel2": torch.tensor([1.0])},
                         "dataset2": {"channel1": torch.tensor([1.0]), "channel2": torch.tensor([2.0, 2.0, 2.0])}}
        selection = ["dataset1", "dataset2"]

        logging_callback = MagicMock()
        termination_callback = MagicMock()

        # Mock implementations
        assembler = MockBatchAssembly(channels)
        model = MagicMock(spec=ContractedModule)
        model.side_effect = RuntimeError("Model processing error")
        disassembler = BatchDisassembly()

        processor = CoreSyncProcessor(assembler, model, disassembler)
        case_buffer = cases_backend.copy()

        with self.assertRaises(RuntimeError):
            processor(selection, case_buffer, logging_callback, termination_callback)

        self.assertTrue(self.check_for_logged_exception(logging_callback, RuntimeError, 1))
