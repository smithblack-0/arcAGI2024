import unittest
import pandas as pd
import os
from src.main.arcAGI2024.training import LogMetrics  # assuming the class is saved in log_metrics.py


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


if __name__ == '__main__':
    unittest.main()
