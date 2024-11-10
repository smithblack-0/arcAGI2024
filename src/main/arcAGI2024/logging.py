from typing import Dict, Any

import csv
from typing import List, Dict, Any


class LogMetrics:
    def __init__(self, file: str, metric_slots: List[str]):
        """
        A logging interface for saving metrics to a CSV file.

        :param file: The file path to save to.
        :param metric_slots: List of metric names to be tracked.
        """
        self.file = file
        self.metric_slots = ["epoch", "batch", "worker"] + metric_slots
        self._initialized = False  # To check if headers are written

    def __call__(self, epoch: int, batch: int, worker: int, metrics: Dict[str, Any]):
        """
        Logs metrics for a given epoch, batch, and worker.

        :param epoch: Current epoch number.
        :param batch: Current batch number.
        :param worker: Worker ID or number.
        :param metrics: Dictionary of metric names and their values.
        """
        # Ensure all metric slots are present in the data
        row = {
            "epoch": epoch,
            "batch": batch,
            "worker": worker,
            **metrics  # Merges the metrics dictionary into the row
        }

        # Write to file
        with open(self.file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.metric_slots)

            # Write headers if this is the first write
            if not self._initialized:
                writer.writeheader()
                self._initialized = True

            # Write the row data
            writer.writerow(row)
