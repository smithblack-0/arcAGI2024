"""
Training interfaces are stored here
"""
import functools
from dataclasses import dataclass
from typing import Optional, Callable, Tuple, List, Dict, Any, Union
from abc import ABC, abstractmethod

import torch
import csv
import os
import pandas as pd
import threading
import concurrent.futures as futures

from torch import nn
from torch.utils.data import DataLoader
from concurrent.futures import ThreadPoolExecutor, Future

from transformers import PreTrainedTokenizer

from .base import TensorTree, parallel_pytree_map
from .grad_utils import AbstractGradientControl, AutorescaleGradientControl
from .losses import MainLossInterface, MemAccessLossInterface
from .model import CausalLMTrainer, Logger, CausalLMCore

try:
    from IPython.display import clear_output

    CLEAR_OUTPUT_AVAILABLE = True
except ImportError:
    CLEAR_OUTPUT_AVAILABLE = False


# Define more concrete implementatons.


class LoggingContext:
    """
    Context manager to provide a thread pool executor for logging or other
    asynchronous tasks.
    """

    def __init__(self, max_workers: int = 2):
        self.max_workers = max_workers
        self.executor = None

    def __enter__(self):
        self.executer = ThreadPoolExecutor(max_workers=self.max_workers)
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

    def make_logging_callback(self,
                              worker: int,
                              epoch: int,
                              batch: int,
                              pass_type: str) -> Callable[[Dict[str, Any]], None]:
        """
        Creates a metric callback that is bound to log information under the
        given worker, epoch, and batch.

        :param worker: The worker id.
        :param epoch: The epoch.
        :param batch: The batch in the epoch.
        :param pass_type: Training, validation, or test.
        :return: The functional callback.
        """

        def logging_callback(metrics: Dict[str, Any]) -> None:
            self.log(epoch, batch, worker, metrics, pass_type)

        return logging_callback

    def log(self,
            epoch: int,
            batch: int,
            worker: int,
            pass_type: str,
            metrics: Dict[str, Any]):
        """
        Logs metrics for a given epoch, batch, and worker to an in-memory
        DataFrame.

        :param epoch: Current epoch number.
        :param batch: Current batch number.
        :param worker: Worker ID or number.
        :param pass_type: Training, validation, or test.
        :param metrics: Dictionary of metric names and their values.
        """
        # Initialize the DataFrame if it hasn't been set up yet
        if self.data is None:
            columns = ["epoch", "batch", "worker", "pass_type"] + list(metrics.keys())
            self.data = pd.DataFrame(columns=columns)
        metrics = {name: float(metric) if isinstance(metric, torch.Tensor) else metric
                   for name, metric in metrics.items()}
        # Append the new row of metrics
        row = {"epoch": epoch, "batch": batch, "worker": worker, "pass_type": pass_type, **metrics}
        self.data = pd.concat([self.data, pd.DataFrame([row])], ignore_index=True)

    def write_epoch(self,) -> Dict[str, str]:
        """
        Writes the accumulated data for the current epoch to the CSV file and
        returns a dictionary with formatted strings of average metrics, organized
        by pass type (e.g., training, validation).

        :return: A dictionary with pass type keys and formatted average summaries.
        """
        if self.data is None or self.data.empty:
            return {f"No data available for epoch {self.current_epoch}."}

        # Group data by pass_type and calculate averages for each group
        summaries = {}
        for pass_type, group in self.data.groupby("pass_type"):
            # Select and average only the specified summary columns
            summary_data = group.select_dtypes(include=['number'])
            averages = summary_data.mean().to_dict()

            # Format the averages into a readable string for this pass_type
            summary_str = f"Epoch {self.current_epoch}, {pass_type} averages: " + ", ".join(
                f"average {metric}: {value:.4f}" for metric, value in averages.items()
            )
            summaries[pass_type] = summary_str

        # Write the epoch data to the CSV file using pandas
        self.data.to_csv(self.file, mode='a' if self._initialized else 'w',
                         index=False, header=not self._initialized)
        self._initialized = True  # File is now initialized

        # Clear data for the current epoch and reinitialize the DataFrame
        self.data = pd.DataFrame(columns=self.data.columns)
        self.current_epoch += 1  # Increment to the next epoch

        return summaries


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

    def make_terminal_callback(self,
                               worker: int,
                               epoch: int,
                               batch: int,
                               ) -> Callable[[str], Any]:
        """
        Creates a terminal callback bound to a particular worker, epoch, and batch
        :param worker: The worker to bind to
        :param epoch: The epoch to display as
        :param batch: The batch to display as
        :return: A callback that will invoke a terminal display
        """

        def terminal_callback(message: str):
            msg = f"W: {worker} | E: {epoch} | B: {batch} | Feedback" + message
            self.worker_display_status[worker] = msg
            self.render()

        return terminal_callback

    def store_epoch_message(self, message: str):
        """
        Stores metrics related to an entire epoch
        """
        self.metrics_status.append(message)
        self.render()

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


class CheckpointProcess:
    """
    A model checkpoint process, that
    can be utilized for training.

    It is capable of saving the model
    to a given location every so often.
    """

    def __init__(self,
                 checkpoint_directory: str,
                 prefix: str,
                 model: CausalLMCore,
                 starting_epoch: int = 0,
                 checkpoint_every_n_batches: int = 100,
                 checkpoint_at_epoch_end=True,
                 ):
        self.epoch = starting_epoch
        self.folder_path = checkpoint_directory
        self.checkpoint_every_n_batches = checkpoint_every_n_batches
        self.checkpoint_at_epoch_end = checkpoint_at_epoch_end
        self.batch = 0
        self.model = model
        self.prefix = prefix

        os.makedirs(checkpoint_directory, exist_ok=True)

    def save_checkpoint(self):
        name = self.prefix + "_" + f"epoch_{self.epoch}_batch_{self.batch}"
        path = os.path.join(self.folder_path, name)
        self.model.save_to_folder(path)

    def step_batch(self):

        if self.batch % self.checkpoint_every_n_batches == 0:
            self.save_checkpoint()
        self.batch += 1

    def step_epoch(self):
        if self.checkpoint_at_epoch_end:
            self.save_checkpoint()
        self.batch = 0
        self.epoch += 1


@dataclass
class LoggingResources:
    """
    Contains the resources used
    to log and display feedback
    while training the model
    """
    metrics_logger: LogMetrics
    terminal_display: TerminalDisplay
    logging_thread: ThreadPoolExecutor


def run_training_epoch(
        numbers: Dict[str, int],
        loaders: Dict[str, DataLoader],
        model: CausalLMCore,
        model_options: Dict[str, Any],
        optim: torch.optim.Optimizer,
        logging_utils: LoggingResources,
        model_checkpointing: CheckpointProcess,
        device: torch.device
    ):
    """
    Runs a singlular training epoch. This includes taking the
    loaders, transferring the relevant bits, advancing optim,
    etc.

    :param numbers: The worker and epoch number and the number of workers
    :param loaders: The training and validation loaders
    :param model: The model to train
    :param model_options: Some auxiliary data
    :param optim: The optimizer
    :param logging_utils: The logging utils
    :param model_checkpointing: The checkpointing util
    """

    # Run training pass
    for i, (tokens, targets, nonpadding_mask) in loaders["training_loader"]:
        # If main process, advance checkpointing.
        if numbers["worker_number"] == 0:
            model_checkpointing.step_batch()

        # Move all to the right device
        tokens = tokens.to(device=device)
        targets = targets.to(device=device)
        nonpadding_mask = nonpadding_mask.to(device=device)

        # Compute the scaling factor. This is used to perform an average over all the active tokens
        scaling_factor = (nonpadding_mask).sum().to(model.dtype)
        scaling_factor = 1 / scaling_factor
        scaling_factor = float(scaling_factor)

        # Setup the logging and feedback
        terminal_callback = logging_utils.terminal_display.make_terminal_callback(numbers["worker_number"],
                                                                                  numbers["epoch_number"],
                                                                                  i)
        metrics_callback = logging_utils.metrics_logger.make_logging_callback(numbers["worker_number"],
                                                                              numbers["epoch_number"],
                                                                              i,
                                                                              "training_pass")
        logging_case = Logger(logging_utils.logging_thread, terminal_callback, metrics_callback)

        # Perform the training step. Most of your time is spent here.
        model.step(tokens, targets, ~nonpadding_mask, logging_case,
                   scheduling_rates=(scaling_factor, scaling_factor),
                   **model_options)

        # Fetch the gradients off the model. Perform our all reduce. Then integrate
        # the results back into the model.

        def reduce_gradients(parameter: torch.Tensor):
            if parameter.grad is not None:
                grad = torch.distributed.all_reduce(parameter.grad, op=torch.distributed.ReduceOp.SUM)
                grad = grad / numbers["num_workers"]
                parameter.grad = grad

        parallel_pytree_map(reduce_gradients, model.parameters())

        # Perform the optim step
        optim.step()
        optim.zero_grad()

def run_validation_epoch(
        numbers: Dict[str, int],
        loaders: Dict[str, DataLoader],
        model: CausalLMCore,
        model_options: Dict[str, Any],
        optim: torch.optim.Optimizer,
        logging_utils: LoggingResources,
        device: torch.device,

    ):
    """
    Runs a sigular validation epoch process
    :param numbers: The worker and epoch number, along with the number of workers
    :param loaders: The training and validation loaders
    :param model: The model to train
    :param model_options: Some auxiliary data
    :param optim: The optimizer
    :param logging_utils: The logging utils
    """
    # Run training pass
    for i, (tokens, targets, nonpadding_mask) in loaders["validation_loader"]:
        # Move all to the right device
        tokens = tokens.to(device=device)
        targets = targets.to(device=device)
        nonpadding_mask = nonpadding_mask.to(device=device)

        # Compute the scaling factor. This is used to perform an average over all the active tokens
        scaling_factor = (nonpadding_mask).sum().to(model.dtype)
        scaling_factor = 1 / scaling_factor
        scaling_factor = float(scaling_factor)

        # Setup the logging and feedback
        terminal_callback = logging_utils.terminal_display.make_terminal_callback(numbers["worker_number"],
                                                                                  numbers["epoch_number"],
                                                                                  i)
        metrics_callback = logging_utils.metrics_logger.make_logging_callback(numbers["worker_number"],
                                                                              numbers["epoch_number"],
                                                                              i,
                                                                              "validation_pass")
        logging_case = Logger(logging_utils.logging_thread, terminal_callback, metrics_callback)

        # Perform the training step. Most of your time is spent here.
        model.step(tokens, targets, ~nonpadding_mask, logging_case,
                   scheduling_rates=(scaling_factor, scaling_factor),
                   **model_options)


        # Zero the grads. We do not need them
        optim.zero_grad()


def run_distributed_epoch(numbers: Dict[str, int],
                          loaders: Dict[str, DataLoader],
                          model: CausalLMCore,
                          model_options: Dict[str, Any],
                          optim: torch.optim.Optimizer,
                          logging_utils: LoggingResources,
                          model_checkpointing: CheckpointProcess,
                          device: torch.Device,
                          ):
    """
    Runs a training epoch in a distributed manner.

    :param numbers: The worker and epoch number, along with the number of workers
    :param loaders: The training and validation loaders
    :param model: The model to train
    :param model_options: Some auxiliary data
    :param optim: The optimizer
    :param logging_utils: The logging utils
    :param model_checkpointing: The checkpointing util
    """
    # Run the training, validation epochs
    run_training_epoch(numbers, loaders, model, model_options, optim, logging_utils, model_checkpointing, device)
    run_validation_epoch(numbers, loaders, model, model_options, optim, logging_utils, device)

    # Perform end of epoch processes, such as checkpointing and advancement
    model_checkpointing.step_epoch()
    summaries = logging_utils.metrics_logger.write_epoch()
    logging_utils.terminal_display.store_epoch_message("during training: " + summaries["training_pass"])
    logging_utils.terminal_display.store_epoch_message("during validation: " + summaries["validation_pass"])


def data_collator(batch: List[Dict[str, List[int]]],
                  tokenizer: PreTrainedTokenizer,
                  truncate_length: int,
                  )->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    A function designed to be bound to using functools.

    It will finish padding and tokenization.

    :param batch: The batch under consideration
    :param tokenizer: The tokenizer to use
    :param truncate_length: What to truncate if we go beyond
    :return: The tokens, targets, and attn mask.
    """
    # Process the batch, truncating and appending ids as needed
    batch = [item["input_ids"] for item in batch]
    batch = [[tokenizer.bos_token_id] + item + [tokenizer.eos_token_id] for item in batch] # ids
    batch = [item[:truncate_length] for item in batch]
    batch = [{"input_ids": item} for item in batch]

    # Produce the encodings
    encodings = tokenizer.pad(batch,
                              padding=True,
                              return_tensors='pt',
                              return_attention_mask=True)

    input_ids= encodings['input_ids']
    batch_mask = encodings['attention_mask']

    # Produce the targets and inputs
    inputs = input_ids[..., :-1]
    targets = input_ids[..., 1:]
    batch_mask = batch_mask[..., :-1].to(dtype=torch.bool)

    # Return
    return inputs, targets, batch_mask

def run_training_in_process(worker_number: int,
                            num_workers: int,
                            batch_size: int,
                            epoch_length_schedule: List[int],
                            logging_utils: LoggingResources,
                            model_factory: Callable[[torch.device], Tuple[CausalLMCore, CausalLMTrainer]],
                            optim_factory: Callable[[nn.Module], torch.optim.Optimizer],
                            checkpoint_factory: Callable[[CausalLMCore], CheckpointProcess],
                            pretokenized_datasets: Dict[str, torch.utils.data.Dataset],
                            ):
    """
    Runs a training process, presumably in parallel.

    :param worker_number: The worker number assigned
    :param batch_size: How wide the batches should be. This should generally be considerably wider than
                       normal.
    :param epoch_length_schedule: For each epoch, how long to make the batch
    :param logging_utils: The logging utilities
    :param model_factory: A factory method capable of making a model given a device
    :param optim_factory: A factory method capable of providing an optim when given a model
    :param pretokenized_datasets: training, test, and validation datasets.
        - These have been tokenized, but not padded, truncated, or placed into a batch
    """
    # Setup the models and training mechanisms
    device = torch.device(f"cuda:{worker_number}" if torch.cuda.is_available() else "cpu")
    core, trainer = model_factory(device)
    optim = optim_factory(trainer)
    checkpointing_process = checkpoint_factory(core)

    # Run the various epochs.
    #
    # Early ones will tend to have much shorter batch lengths than
    # the later ones.
    for sequence_length in epoch_length_schedule:
        loaders = {}

        for name, dataset in pretokenized_datasets.items():
            distributed_sample = torch.utils.data.distributed.DistributedSampler(dataset,
                                                                                 num_workers,
                                                                                 worker_number,
                                                                                 shuffle=True,
                                                                                 )

            collate_fn = functools.partial(data_collator,
                                           tokenizer = core.vocabulary.tokenizer,
                                           truncate_length = sequence_length)

            loader = torch.utils.data.DataLoader(dataset,
                                                 batch_size,
                                                 shuffle=True,
                                                 num_workers=1,
                                                 sampler=distributed_sample,
                                                 pin_memory=True,
                                                 prefetch_factor=2,
                                                 collate_fn=collate_fn
                                                 )
            loaders[name] = loader
            epoch_loaders.append(loaders)

    for epoch, epoch_loaders in enumerate(epoch_loaders):
        numbers = {
            "epoch_number" : epoch,
            "worker_number" : worker_number,
            "num_workers" : num_workers,
        }









