"""
Training interfaces are stored here
"""
from typing import Optional, Callable, Tuple, List, Dict, Any, Union
from abc import ABC, abstractmethod

import torch
import csv
import os
import pandas as pd

from torch import nn
from torch.utils.data import DataLoader
from concurrent.futures import ThreadPoolExecutor

from .base import TensorTree
from .grad_utils import AbstractGradientControl, AutorescaleGradientControl
from .losses import MainLossInterface, MemAccessLossInterface
from .model import CausalLMTrainer

try:
    from IPython.display import clear_output

    CLEAR_OUTPUT_AVAILABLE = True
except ImportError:
    CLEAR_OUTPUT_AVAILABLE = False


# Define important interfaces





# Define more concrete implementatons.
class LoggingContext:
    """
    Context manager to provide a thread pool executor for logging or other
    asynchronous tasks.
    """
    def __init__(self, max_workers: int = 2):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def __enter__(self):
        return self.executor

    def __exit__(self, exc_type, exc_value, traceback):
        self.executor.shutdown(wait=True)

class LogMetrics:
    def __init__(self, file: str, start_epoch: Optional[int] = None):
        """
        A logging interface for saving metrics to an in-memory DataFrame and
        periodically writing epochs to a CSV file.

        :param file: The file path to save the metrics.
        :param start_epoch: The epoch to start from. If not None, assumes
                            resuming.
        """
        self.file = file
        self.current_epoch = start_epoch if start_epoch is not None else 0
        self.data = None  # DataFrame initialized on first log call if not resuming
        self._initialized = False

        # If resuming, load headers from the existing file
        if start_epoch is not None and os.path.exists(file):
            self._initialize_from_existing_file()

    def _initialize_from_existing_file(self):
        """
        Initializes the DataFrame with headers from an existing CSV file to
        allow for resuming a session.
        """
        with open(self.file, 'r') as f:
            headers = f.readline().strip().split(",")  # Read headers from the file
            self.data = pd.DataFrame(columns=headers)
            self._initialized = True

    def make_logging_factories(
            self, worker: int
    ) -> Callable[[int, int], Callable[[Dict[str, Any]], None]]:
        """
        Creates a factory bound to a specific worker that, when invoked with
        an epoch and batch, creates a logging callback for logging metrics.

        :param worker: Worker ID to bind to the factory.
        :return: A factory function to create a logging callback.
        """

        def logging_factory(epoch: int, batch: int) -> Callable[[Dict[str, Any]], None]:
            def logging_callback(metrics: Dict[str, Any]) -> None:
                self.log(epoch, batch, worker, metrics)

            return logging_callback

        return logging_factory

    def log(self, epoch: int, batch: int, worker: int, metrics: Dict[str, Any]):
        """
        Logs metrics for a given epoch, batch, and worker to an in-memory
        DataFrame.

        :param epoch: Current epoch number.
        :param batch: Current batch number.
        :param worker: Worker ID or number.
        :param metrics: Dictionary of metric names and their values.
        """
        # Initialize the DataFrame if it hasn't been set up yet
        if self.data is None:
            columns = ["epoch", "batch", "worker"] + list(metrics.keys())
            self.data = pd.DataFrame(columns=columns)

        # Append the new row of metrics
        row = {"epoch": epoch, "batch": batch, "worker": worker, **metrics}
        self.data = pd.concat([self.data, pd.DataFrame([row])], ignore_index=True)

    def write_epoch(self, summary_columns: List[str]) -> str:
        """
        Writes the accumulated data for the current epoch to the CSV file and
        returns a formatted string with average metrics over the specified
        columns.

        :param summary_columns: List of column names to calculate averages
                                over for the epoch summary.
        :return: A formatted string with epoch number and averages.
        """
        if self.data is None or self.data.empty:
            return f"No data available for epoch {self.current_epoch}."

        # Select only the specified summary columns for averaging
        summary_data = self.data[summary_columns].select_dtypes(include=['number'])
        averages = summary_data.mean().to_dict()

        # Write the epoch data to the CSV file using pandas
        self.data.to_csv(self.file, mode='a' if self._initialized else 'w',
                         index=False, header=not self._initialized)
        self._initialized = True  # File is now initialized

        # Format the averages into a readable string
        summary_str = f"Epoch {self.current_epoch} averages: " + ", ".join(
            f"average {metric}: {value:.4f}" for metric, value in averages.items()
        )

        # Clear data for the current epoch and reinitialize the DataFrame
        self.data = pd.DataFrame(columns=self.data.columns)
        self.current_epoch += 1  # Increment to the next epoch

        return summary_str


class TerminalDisplay:
    """
    A helper class that initializes and keeps
    track of various worker instances and whatever
    feedback they are providing.

    """

    def __init__(self,
                 num_workers: int
                 ):
        """
        Initializes a terminal display for a certain number
        of workers. Each worker gets a line, and we use
        carriage returns to overwrite when seeing updates
        :param num_workers: The number of workers to reserver
        space for.
        """
        self.num_workers = num_workers
        self.metrics_status = []
        self.worker_display_status = ["No status yet"] * num_workers

    def make_terminal_callbacks(self) -> List[Callable[[str], None]]:
        """
        Creates a list of terminal callbacks, one slot per worker,
        which information can be inserted into. Whenever called,
        it replaces what is currently displayed.

        Note that each callback is assigned one terminal line.
        :return: The list of terminal callbacks.
        """
        outputs = []
        for i in range(self.num_workers):
            def terminal_callback(message: str, capture=i) -> None:
                self.worker_display_status[capture] = message

            outputs.append(terminal_callback)
        return outputs

    def store_epoch_message(self, message: str):
        """
        Stores metrics related to an entire epoch
        """
        self.metrics_status.append(message)

    def render(self):
        """
        Renders the worker status to the terminal
        """
        if CLEAR_OUTPUT_AVAILABLE:
            clear_output()

        output = "----Epoch metrics -----\n"
        output += "\n".join(self.metrics_status)
        output += "----worker feedback----\n"
        output += "\n".join(self.worker_display_status)
        print(output)

class BoundTrainingEndpoint:
    """
    Represents a model bound to a device, or
    series of devices. Can be invoked with a batch
    in order to run it.
    """
    def __init__(self,
                 device: torch.device,
                 model: CausalLMTrainer,
                 optim: torch.optim.Optimizer,
                 save_cached_to_cpu: bool,
                 numeric_cache_rate: int,
                 schedule: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
                 ):
        super().__init__()
        self.device = device
        self.model = model
        self.optim = optim
        self.schedule = schedule
        self.save_cached_to_cpu = save_cached_to_cpu
        self.numeric_cache_rate = numeric_cache_rate


    def __call__(self,
                 loader: DataLoader,
                 logger:
                ):


class Trainer:
    """
    A class designed to train a machine learning model.

    It includes
    """
    def __init__(self,
                 logging_file: str,
                 train_loader: DataLoader,
                 validation_loader: DataLoader,
                 devices: List[torch.device],
                 model_factory: Callable[[torch.device], CausalLMTrainer],
                 ):
