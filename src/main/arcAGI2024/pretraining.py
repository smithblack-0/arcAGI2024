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

from tinycss2 import tokenizer
from torch import nn
from torch.utils import data
from concurrent.futures import ThreadPoolExecutor, Future

from datasets import DatasetDict, Dataset
from transformers import PreTrainedTokenizer

from .base import TensorTree, parallel_pytree_map
from .grad_utils import AbstractGradientControl, AutorescaleGradientControl
from .losses import MainLossInterface, MemAccessLossInterface
from .model import CausalLMTrainer, Logger, CausalLMCore
from .data.pretraining import PretrainingLoaderConfig, create_dataloader_factory

try:
    from IPython.display import clear_output

    CLEAR_OUTPUT_AVAILABLE = True
except ImportError:
    CLEAR_OUTPUT_AVAILABLE = False


# Define more concrete implementatons.

@dataclass
class TrainingConfig:
    """
    A central dataclass that holds all the dynamic features needed to
    properly set up and run a training instance.

    Fields:
    --------
    loader_factories: Callable[[int], Dict[str, data.DataLoader]]
        A function that produces data loaders for training and validation,
        keyed by dataset names. Takes the worker num as input
    training_run_prefix: str
        Prefix for naming or identifying this training run.
    metrics_logging_directory: str
        Directory where metrics logs will be saved.
    checkpoint_save_directory: str
        Directory where model checkpoints will be saved.
    checkpoint_batch_frequency: int
        Frequency (in batches) at which checkpoints are saved.
    num_workers: int
        Number of workers used for data loading.
    num_epochs: int
        Total number of epochs for the training run.
    epoch_position: Optional[int]
        Current epoch position in the training process. Useful for resuming
        from checkpoints. Defaults to None if starting from the beginning.
    """

    # devices
    devices: List[torch.device]

    # Loader config
    loader_config: PretrainingLoaderConfig

    # Training and logging niceties
    training_run_prefix: str
    metrics_logging_directory: str
    checkpoint_save_directory: str
    checkpoint_batch_frequency: int

    # Important features
    num_epochs: int
    num_workers: int
    epoch_position: Optional[int] = None

    @property
    def num_workers(self)->int:
        return len(self.devices)


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
    def __init__(self, training_config: TrainingConfig):
        """
        A logging interface for saving metrics to an in-memory DataFrame and
        periodically writing epochs to a CSV file.
        """
        self.file = training_config.metrics_logging_directory
        self.current_epoch = training_config.epoch_position
        self.data = None  # DataFrame initialized on first log call if not resuming
        self._initialized = training_config.epoch_position > 0

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

    def write_epoch(self, ) -> Dict[str, str]:
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

    def __init__(self, training_config: TrainingConfig):
        """
        Initializes a terminal display for a certain number
        of workers. Each worker gets a line, and we use
        carriage returns to overwrite when seeing updates
        :param num_workers: The number of workers to reserver
        space for.
        """
        self.num_workers = training_config.num_workers
        self.metrics_status = []
        self.worker_display_status = ["No status yet"] * training_config.num_workers
        self.render()

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

    def __init__(self, model: CausalLMCore, training_config: TrainingConfig):
        self.model = model

        self.epoch = training_config.epoch_position
        self.folder_path = training_config.checkpoint_save_directory
        self.checkpoint_every_n_batches = training_config.checkpoint_batch_frequency
        self.batch = 0
        self.prefix = training_config.training_run_prefix

        os.makedirs(self.folder_path, exist_ok=True)

    def save_checkpoint(self):
        name = self.prefix + "_" + f"epoch_{self.epoch}_batch_{self.batch}"
        path = os.path.join(self.folder_path, name)
        self.model.save_to_folder(path)

    def step_batch(self):
        if self.batch % self.checkpoint_every_n_batches == 0:
            self.save_checkpoint()
        self.batch += 1

    def step_epoch(self):
        self.save_checkpoint()
        self.batch = 0
        self.epoch += 1


@dataclass
class TrainingResources:
    """
    Contains the resources used
    to log and display feedback
    while training the model
    """
    # Device
    device: torch.device
    num_workers: int

    # Logging pieces
    metrics_logger: LogMetrics
    terminal_display: TerminalDisplay
    logging_thread: ThreadPoolExecutor

    # Checkpointing and Optim
    core: CausalLMCore
    trainer: CausalLMTrainer
    checkpointing: CheckpointProcess
    optim: torch.optim.Optimizer


def run_training_epoch(
        worker_num: int,
        epoch_num: int,
        train_loader: data.DataLoader,
        training_resources: TrainingResources,
):
    """
    Runs a singlular training epoch. This includes taking the
    loaders, transferring the relevant bits, advancing optim,
    etc.
    :param worker_num: The worker num associated with this. Used for logging
    :param epoch_num: The epoch num associated with this. Used for logging
    :param train_loader: The training dataloader
    :param training_resources: Various training resources. See the class
    """

    # Run training pass
    for i, (tokens, targets, nonpadding_mask) in train_loader:

        # If main process, advance checkpointing.
        if worker_num == 0:
            # The first worker is in charge of checkpointing
            training_resources.checkpointing.step_batch()

        # Move all to the right device
        tokens = tokens.to(device=training_resources.device)
        targets = targets.to(device=training_resources.device)
        nonpadding_mask = nonpadding_mask.to(device=training_resources.device)

        # Compute the scaling factor. This is used to perform an average over all the active tokens
        scaling_factor = (nonpadding_mask).sum().to(training_resources.core.dtype)
        scaling_factor = 1 / scaling_factor
        scaling_factor = float(scaling_factor)

        # Setup the logging and feedback
        terminal_callback = training_resources.terminal_display.make_terminal_callback(worker_num,
                                                                                       epoch_num,
                                                                                       i)
        metrics_callback = training_resources.metrics_logger.make_logging_callback(worker_num,
                                                                                   epoch_num,
                                                                                   i,
                                                                                   "training_pass")
        logging_case = Logger(training_resources.logging_thread, terminal_callback, metrics_callback)

        # Perform the training step. Most of your time is spent here.
        training_resources.trainer.step(tokens, targets, ~nonpadding_mask, logging_case,
                                        scheduling_rates=(scaling_factor, scaling_factor))

        # Fetch the gradients off the model. Perform our all reduce. Then integrate
        # the results back into the model.

        def reduce_gradients(parameter: torch.Tensor):
            if parameter.grad is not None:
                grad = torch.distributed.all_reduce(parameter.grad, op=torch.distributed.ReduceOp.SUM)
                grad = grad / training_resources.num_workers
                parameter.grad = grad

        parallel_pytree_map(reduce_gradients, training_resources.core.parameters())

        # Perform the optim step
        training_resources.optim.step()
        training_resources.optim.zero_grad()


def run_validation_epoch(
        worker_num: int,
        epoch_num: int,
        validation_loader: data.DataLoader,
        training_resources: TrainingResources,
):
    """
    Runs a validation epoch process
    :param worker_num: The worker num associated with this. Used for logging
    :param epoch_num: The epoch num associated with this. Used for logging
    :param validation_loader: The validation dataloader
    :param training_resources: Various training resources. See the class
    """
    # Run training pass
    for i, (tokens, targets, nonpadding_mask) in validation_loader:
        # Move all to the right device
        tokens = tokens.to(device=training_resources.device)
        targets = targets.to(device=training_resources.device)
        nonpadding_mask = nonpadding_mask.to(device=training_resources.device)

        # Compute the scaling factor. This is used to perform an average over all the active tokens
        scaling_factor = (nonpadding_mask).sum().to(training_resources.core.dtype)
        scaling_factor = 1 / scaling_factor
        scaling_factor = float(scaling_factor)

        # Setup the logging and feedback
        terminal_callback = training_resources.terminal_display.make_terminal_callback(worker_num,
                                                                                       epoch_num,
                                                                                       i)
        metrics_callback = training_resources.metrics_logger.make_logging_callback(worker_num,
                                                                                   epoch_num,
                                                                                   i,
                                                                                   "validation_pass")
        logging_case = Logger(training_resources.logging_thread, terminal_callback, metrics_callback)

        # Perform the training step. Most of your time is spent here.
        training_resources.trainer.step(tokens, targets, ~nonpadding_mask, logging_case,
                                        scheduling_rates=(scaling_factor, scaling_factor))

        # Zero the grads. We do not need them. But we cannot let them get full and give us NAN's
        training_resources.optim.zero_grad()


def run_test_epoch(
        worker_num: int,
        epoch_num: int,
        test_loader: data.DataLoader,
        training_resources: TrainingResources,
):
    """
    Runs a validation epoch process
    :param worker_num: The worker num associated with this. Used for logging
    :param epoch_num: The epoch num associated with this. Used for logging
    :param test_loader: The validation dataloader
    :param training_resources: Various training resources. See the class
    """
    # Run training pass
    for i, (tokens, targets, nonpadding_mask) in test_loader:
        # Move all to the right device
        tokens = tokens.to(device=training_resources.device)
        targets = targets.to(device=training_resources.device)
        nonpadding_mask = nonpadding_mask.to(device=training_resources.device)

        # Compute the scaling factor. This is used to perform an average over all the active tokens
        scaling_factor = (nonpadding_mask).sum().to(training_resources.core.dtype)
        scaling_factor = 1 / scaling_factor
        scaling_factor = float(scaling_factor)

        # Setup the logging and feedback
        terminal_callback = training_resources.terminal_display.make_terminal_callback(worker_num,
                                                                                       epoch_num,
                                                                                       i)
        metrics_callback = training_resources.metrics_logger.make_logging_callback(worker_num,
                                                                                   epoch_num,
                                                                                   i,
                                                                                   "test_pass")
        logging_case = Logger(training_resources.logging_thread, terminal_callback, metrics_callback)

        # Perform the training step. Most of your time is spent here.
        training_resources.trainer.step(tokens, targets, ~nonpadding_mask, logging_case,
                                        scheduling_rates=(scaling_factor, scaling_factor))

        # Zero the grads. We do not need them. But we cannot let them get full and give us NAN's
        training_resources.optim.zero_grad()


def run_training_process(worker_num: int,
                         training_configs: List[TrainingConfig],
                         loader_factory: Callable[[int], data.DataLoader],
                         model_factory: Callable[[torch.device], Tuple[CausalLMCore, CausalLMTrainer]],
                         optim_factory: Callable[[nn.Module], torch.optim.Optimizer],
                         schedule_factory: Callable[
                             [torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LRScheduler]],
                         ):
    """
    Runs a training process, in parallel, and presumably
    on the GPU.

    :param worker_num: The worker number assigned. Given out by spawn
    :param training_configs: The training configs to use. Each will be used in sequence, advancing from the first
                             batch length.
    :param loader_factory: A factory that when given a rank returns a working loader set
    :param model_factory: A factory method capable of making a model given a device
    :param optim_factory: A factory method capable of providing an optim when given a model.
    :param schedule_factory: A factory method capable of making a schedule given an optimizer.
    """
    # Setup the models and training mechanisms
    device = training_configs[0].devices[worker_num]
    core, trainer = model_factory(device)
    optim = optim_factory(trainer)
    checkpointing = CheckpointProcess(core, training_configs)
    schedule = schedule_factory(optim)
    epoch_num = training_configs[0].epoch_position
    num_workers = training_configs[0].num_workers

    with ThreadPoolExecutor(max_workers=2) as logging_threads:
        # Run training under configurations
        for training_config in training_configs:
            training_config.epoch_position = epoch_num

            terminal_display = TerminalDisplay(training_config)
            metrics_logger = LogMetrics(training_config)

            # Create the training resources
            resources = TrainingResources(
                device=device,
                num_workers=num_workers,
                logging_thread=logging_threads,
                terminal_display=terminal_display,
                metrics_logger=metrics_logger,
                core=core,
                trainer=trainer,
                checkpointing=checkpointing,
                optim=optim,
            )

            # Create the dataloaders
            loaders = loader_factory(worker_num)

            for _ in range(training_config.num_epochs):
                # Run primary epochs
                run_training_epoch(worker_num, epoch_num, loaders["train_loader"], resources)
                run_validation_epoch(worker_num, epoch_num, loaders["validation_loader"], resources)

                # Perform end of epoch processes, such as checkpointing and advancement
                resources.checkpointing.step_epoch()
                summaries = resources.metrics_logger.write_epoch()
                resources.terminal_display.store_epoch_message("during training: " + summaries["training_pass"])
                resources.terminal_display.store_epoch_message("during validation: " + summaries["validation_pass"])
                if schedule is not None:
                    schedule.step()

                # advance epoch
                epoch_num += 1

            # Done with this configuration, so we run the test set
            run_test_epoch(worker_num, epoch_num, loaders["test_loader"], resources)
            epoch_num += 1


def spawn_process_factory(training_configs: List[TrainingConfig],
                          core: CausalLMCore,
                          trainer: CausalLMTrainer,
                          optim_factory: Callable[[nn.Module], torch.optim.Optimizer],
                          schedule_factory: Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler],
                          ) -> Callable[[int], None]:

    """
    Provided with everything except the process number, and this
    will give you back something that can accept that process number
    in order to spawn a config bound to the device.

    :param training_configs:
    :param core:
    :param trainer:
    :param optim_factory:
    :param schedule_factory:
    :return:
    """
    template_config = training_configs[0]
    num_workers = template_config.num_workers
    def spawn_model(worker_num: int, core=core, trainer=trainer)->Tuple[CausalLMCore, CausalLMTrainer]:
        device = template_config.devices[worker_num]
        core = core.to(device)
        trainer = trainer.to(device)
        return core, trainer

    loader_factory = create_dataloader_factory(num_workers,
                                               core.vocabulary.tokenizer,
                                               template_config.loader_config)

    run_spawned_process = functools.partial(run_training_process,
                                            training_configs=training_configs,
                                            loader_factory=loader_factory,
                                            model_factory=spawn_model,
                                            optim_factory=optim_factory,
                                            schedule_factory=schedule_factory
                                            )
    return run_spawned_process



