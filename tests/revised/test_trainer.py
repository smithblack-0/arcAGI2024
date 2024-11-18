import unittest
import pandas as pd
import os
import shutil
from src.main.arcAGI2024.training import (LogMetrics, TerminalDisplay,
                                          TrainingConfigOld, CheckpointProcess)
import os
from unittest.mock import MagicMock, patch
import torch.multiprocessing as mp
from unittest import TestCase, mock
from tempfile import TemporaryDirectory

class TestLogMetrics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.file_path = "test_metrics.csv"

    def setUp(self):
        # Ensure a clean slate for each test
        if os.path.exists(self.file_path):
            os.remove(self.file_path)

    def tearDown(self):
        # Cleanup after tests
        if os.path.exists(self.file_path):
            os.remove(self.file_path)

    def test_initialization(self):
        # Test normal initialization
        logger = LogMetrics(self.file_path)
        self.assertEqual(logger.current_epoch, 0)
        self.assertFalse(logger._initialized)

        # Test resuming from an existing file
        with open(self.file_path, 'w') as f:
            f.write("epoch,batch,worker,accuracy\n")
        logger_resume = LogMetrics(self.file_path, start_epoch=1)
        self.assertTrue(logger_resume._initialized)
        self.assertEqual(logger_resume.current_epoch, 1)

    def test_logging_metrics(self):
        logger = LogMetrics(self.file_path)
        logger.log(epoch=0, batch=1, worker=1, metrics={"accuracy": 0.85, "loss": 0.15})

        # Check the internal DataFrame structure and contents
        self.assertIsNotNone(logger.data)
        self.assertEqual(len(logger.data), 1)
        self.assertIn("accuracy", logger.data.columns)
        self.assertEqual(logger.data.iloc[0]["accuracy"], 0.85)

    def test_write_epoch(self):
        logger = LogMetrics(self.file_path)
        # Log some metrics for epoch 0
        logger.log(epoch=0, batch=1, worker=1, metrics={"accuracy": 0.85, "loss": 0.15})
        logger.log(epoch=0, batch=2, worker=1, metrics={"accuracy": 0.88, "loss": 0.12})

        # Write the epoch and verify the output
        summary = logger.write_epoch(["accuracy", "loss"])
        self.assertEqual(summary, "Epoch 0 averages: average accuracy: 0.8650, average loss: 0.1350")

        # Check that the data was written to the file
        df_written = pd.read_csv(self.file_path)
        self.assertEqual(len(df_written), 2)
        self.assertIn("accuracy", df_written.columns)
        self.assertEqual(df_written["accuracy"].iloc[0], 0.85)

        # Ensure data is cleared after writing
        self.assertTrue(logger.data.empty)

    def test_append_on_resume(self):
        # Create an initial log file with some data
        logger = LogMetrics(self.file_path)
        logger.log(epoch=0, batch=1, worker=1, metrics={"accuracy": 0.85, "loss": 0.15})
        logger.write_epoch(["accuracy", "loss"])

        # Initialize a new logger with start_epoch to resume logging
        logger_resume = LogMetrics(self.file_path, start_epoch=1)
        logger_resume.log(epoch=1, batch=1, worker=2, metrics={"accuracy": 0.90, "loss": 0.10})
        logger_resume.write_epoch(["accuracy", "loss"])

        # Check that both epochs are present in the CSV file
        df_written = pd.read_csv(self.file_path)
        self.assertEqual(len(df_written), 2)  # Two rows written (one from each epoch)
        self.assertEqual(df_written["epoch"].iloc[0], 0)
        self.assertEqual(df_written["epoch"].iloc[1], 1)
        self.assertEqual(df_written["accuracy"].iloc[1], 0.90)


class TestCheckpointProcess(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for checkpoints
        self.checkpoint_dir = "test_checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Configure TrainingConfig
        self.config = TrainingConfigOld(
            pretokenized_datasets=None,  # Not used in this test
            training_run_prefix="test_run",
            metrics_logging_directory="",
            checkpoint_save_directory=self.checkpoint_dir,
            checkpoint_batch_frequency=2,
            batch_size=8,
            num_workers=1,
            truncate_length=128,
            num_epochs=10,
            epoch_position=0
        )

        # Mock model with a save_to_folder method that writes a test file
        self.mock_model = MagicMock()

        def mock_save_to_folder(path):
            # Ensure the directory exists before saving the checkpoint file
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, "checkpoint.txt"), "w") as f:
                f.write("Checkpoint saved.")

        self.mock_model.save_to_folder.side_effect = mock_save_to_folder

        # Initialize CheckpointProcess with the mock model and config
        self.checkpoint_process = CheckpointProcess(self.mock_model, self.config)

    def tearDown(self):
        # Clean up by removing the temporary checkpoint directory after each test
        shutil.rmtree(self.checkpoint_dir)

    def test_checkpoint_saving_frequency(self):
        # Simulate stepping through batches and check file creation
        for i in range(5):
            self.checkpoint_process.step_batch()

        # Check that the expected checkpoint files are saved
        expected_files = [
            os.path.join(self.checkpoint_dir, "test_run_epoch_0_batch_0", "checkpoint.txt"),
            os.path.join(self.checkpoint_dir, "test_run_epoch_0_batch_2", "checkpoint.txt"),
            os.path.join(self.checkpoint_dir, "test_run_epoch_0_batch_4", "checkpoint.txt"),
        ]
        for file_path in expected_files:
            self.assertTrue(os.path.exists(file_path))
            with open(file_path, "r") as f:
                content = f.read()
                self.assertEqual(content, "Checkpoint saved.")

    def test_checkpoint_on_epoch_step(self):
        # Ensure a checkpoint is saved at the end of each epoch
        self.checkpoint_process.step_epoch()
        first_checkpoint = os.path.join(self.checkpoint_dir, "test_run_epoch_0_batch_0", "checkpoint.txt")
        self.assertTrue(os.path.exists(first_checkpoint))

        # Simulate moving to the next epoch and check the new checkpoint location
        self.checkpoint_process.step_epoch()
        second_checkpoint = os.path.join(self.checkpoint_dir, "test_run_epoch_1_batch_0", "checkpoint.txt")
        self.assertTrue(os.path.exists(second_checkpoint))

        # Verify file contents
        for checkpoint in [first_checkpoint, second_checkpoint]:
            with open(checkpoint, "r") as f:
                content = f.read()
                self.assertEqual(content, "Checkpoint saved.")

    def test_increment_batch_and_epoch(self):
        # Check if batch and epoch counters increment correctly
        self.assertEqual(self.checkpoint_process.batch, 0)
        self.assertEqual(self.checkpoint_process.epoch, 0)

        # Step through a few batches
        self.checkpoint_process.step_batch()
        self.assertEqual(self.checkpoint_process.batch, 1)

        # Step through an epoch
        self.checkpoint_process.step_epoch()
        self.assertEqual(self.checkpoint_process.epoch, 1)
        self.assertEqual(self.checkpoint_process.batch, 0)  # batch should reset after epoch
