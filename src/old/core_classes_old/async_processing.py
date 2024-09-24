"""
Define the core async batch processor classes we can utilize
"""
import asyncio
import textwrap
from dataclasses import dataclass

import torch
from torch import nn
from torch.futures import Future
from abc import ABC, abstractmethod
from typing import Tuple, List

from .clustering import ClusteringStrategy
from src.old.model.data import ActionRequest
from .types import (
                    # Input handling, mostly
                    DataCase,
                    DataCaseBuffer,
                    FutureProcessingBuffer,

                    # Callbacks
                    LoggingCallback,
                    FutureProcessingCallback,
                    TerminationCallback,

                    # Outputs and stream
                    CaseResponse,
                    ExceptionAugmentedResponse


)
from core_processer import CoreSyncProcessor

## Basic batch strategy classes
#
# These primarily have the responsibility of getting the statistics
# that the  clustering strategies need. At the moment, we base
# everything on position.
class BatchStrategy(ABC):
    """
    The abstract batch strategy class.

    The batch strategy class has the primary responsibility of
    extracting important statistics from items in the request
    buffer to allow the various clustering strategies to work.

    It is dependency-injected with a clustering strategy
    """
    @abstractmethod
    def get_vital_statistics(self, case_buffer: DataCaseBuffer)->torch.Tensor:
        """
        An extremely important method, this gives us vital information on
        the various items in the request buffer. It needs to be compatible
        with the clustering mechanism.

        Exact details will vary depending on how vital statistics are extracted
        :return:
        """
        pass

    def __init__(self,
                 clustering_strategy: ClusteringStrategy
                 ):
        self.clustering_strategy = clustering_strategy

    def __call__(self,
                 batch_case_buffer: DataCaseBuffer,
                 batch_size: int,
                 logging_callback: LoggingCallback,
                 force_selection: str,
                 )->List[str]:
        vitals = self.get_vital_statistics(batch_case_buffer)
        output = self.clustering_strategy(vitals, batch_size, logging_callback, force_selection)
        return output

###
# Results processor
#
# The result processor system is designed to take the
# info produced by the main and turn it into another
# action request the system can use.
###
class ResultsProcessor(ABC):
    """
    The results processor is intended to
    take within it the results of executing
    the core neural network main, and
    several important factors, then return
    an ActionRequest letting us know what
    to do next.

    This is an abstract method. The method
    "process_model_results" must be implemented before
    the class can work
    """
    def __init__(self,
                 available_destinations: List[str]
                 ):
        self.destinations = available_destinations

    @abstractmethod
    def process_model_results(self,
                              available_destinations: List[str],
                              request: ActionRequest,
                              results: CaseResponse,
                              ) -> ActionRequest:
        """
        The required abstract method. It should accept
        an action request, the results of processing the request,
        and integrate them into a novel action request. Notably,
        the available destinations are also available for use

        :param available_destinations: The valid next finite states
        :param request: The original action request
        :param results: The results of running the request
        :return: The revised action request that needs to be returned
        """
        pass

    def __call__(self,
                 request: ActionRequest,
                 results: CaseResponse,
                 ) -> ActionRequest:
        """
        When invoked, this will use the provided information
        to compute the next action request.

            :param request: The original action request
        :param results: The results of running the request
        :return: A new action request
        """

        action_request = self.process_model_results(self.destinations, request, results)
        assert action_request.state_tracker.destination in self.destinations
        return action_request

###
#
# Futures factory.
#
# The futures factory is designed to implement two parallel responsibilities
# It develops futures, and it develops callbacks for processing the results
# of the associated computations
##

class FuturesFactory:
    """
    Futures factory.

    This small class has two primary responsibilities.
    It creates futures, and it creates callbacks to ultimately
    process the computations associated with those futures.
    """
    def __init__(self,
                 results_processor: ResultsProcessor,
                 ):
        self.results_processor = results_processor

    def create_future_callback(self,
                               future: Future,
                               id: str,
                               logging_callback: LoggingCallback
                               )->FutureProcessingCallback:
        """
        Creates a callback which will fufill or error out the
        future, one way or another

        :param future: The future to build the callback around
        :return: A callback that will either populate the future with
                 the next action request, or an exception, depending on the data
        """
        def future_callback(result: ExceptionAugmentedResponse):
            # Handle the case in which an exception was encountered.
            #
            # We populate the future with the exception
            if isinstance(result, Exception):
                future.set_exception(result)

            # We did NOT encounter an exception. Attempt
            # to process and populate the future
            try:
                action_result = self.results_processor(result)
                future.set_result(action_result)
            except Exception as err:
                msg = f"""
                Issue was encountered while attempting to run the 
                results processor. This prevented the return of a 
                valid future, despite the main running correctly.
                This occurred while running callback for {id}
                """
                msg = textwrap.dedent(msg)
                exception = RuntimeError(msg)
                exception.__cause__ = err
                logging_callback(exception, 0)
                future.set_exception(exception)
        return future_callback

    def __call__(self,
                 id: str,
                 logging_callback: LoggingCallback,
                 )->Tuple[Future, FutureProcessingCallback]:
        """
        The factory method.

        We will create a future, and an associated processing callback

        :param id: The id of the case we are building this for. Used in messages
        :param logging_callback: The logging callback
        :return:
            - Future: A future which we promise to fill with a ActionRequest... or raise an Error.
            - Callback: Should be called with the result of running the batch.
        """
        future = Future()
        callback = self.create_future_callback(future, id, logging_callback)
        return future, callback

###
# Request Receiver
#
# The request processor is responsible for moving requests into
# the batch case buffer. It is a mix of an interface and, if it ever
# matters, a cache in and of itself.
###

class RequestDataExtractor(ABC):
    """
    The request data extractor is one of the interfaces
    that must be implemented for the main to function.
    Simply put, it should extract from a request the
    key tensors needed for batching and core main processing.
    """

    @abstractmethod
    def process_request(self, request: ActionRequest) -> DataCase:
        """
        The method the implementer must implement for the class to function

        It must take an action request and turn it into a dictionary of tensors.
        These tensors will then be the only ones available downstream in the main.

        :param request: An action request
        :return: A dictionary of tensors, like {"shape" : tensor, "targets" : data}
        """
        pass
    def __call__(self, request: ActionRequest) -> DataCase:
        return self.process_request(request)


##
# End of user servicable logic. Beginning of wrapper classes and dataclasses
##

@dataclass
class AsyncBuffers:
    """
    Contains the primary processing buffers, including
    the async works.
    """
    cases_buffer: DataCaseBuffer
    callbacks_buffer: FutureProcessingBuffer
    selection_buffer: asyncio.Queue


class RequestInserter(ABC):
    """
    Request Inserter

    The request inserter is responsible for moving requests into
    the batch case buffer and the callback buffer. These moves
    are hopefully executed in a thread-safe manner.

    It is also responsible for triggering the batch creation
    logic once the amount of cases stored exceeds the buffer
    threshold.

    This class is not intended to be implemented by the user.
    """
    def __init__(self,
                 # parameters
                 buffer_threshold: int,
                 batch_size: int,

                 # Support classes
                 future_factory: FuturesFactory,
                 batch_strategy: BatchStrategy,
                 data_extractor: RequestDataExtractor,
                 ):
        assert buffer_threshold >= batch_size

        self.batch_size = batch_size
        self.buffer_threshold = buffer_threshold
        self.batch_strategy = batch_strategy
        self.future_factory = future_factory
        self.data_extractor = data_extractor

    def set_timeout_trigger(self,
                            id: str,
                            timeout: float,
                            buffers: AsyncBuffers,
                            logging_callback: LoggingCallback
                            ):
        """
        Sets a timeout trigger on the async task collection. In the event that
        a case is not handled by the time it times out, we force its run.

        :param id: The id to force completion on timeout, if it still has not been handled
        :param timeout: The timeout to wait for, in milliseconds.
        :parma buffers: The buffers to inspect for run completion
        """
        time_seconds = timeout/1000

        # Define the timeout task. It will simply await
        # until the time is passed, then see if the id is still
        # waiting for processing.

        async def timer_task():
            try:
                # Wait before timeout
                await asyncio.sleep(time_seconds)

                # Check if task has already been processed
                if id not in buffers.cases_buffer:
                    return
                # Task was NOT already processed. Log first
                msg = f"""
                Timeout was reached for feature with uuid: '{id}'
                An attempt will be made to force a batch build. 
                """
                msg = textwrap.dedent(msg)
                logging_callback(msg, 2)

                # Now force injection into task queue.q

                selection = self.batch_strategy(buffers.cases_buffer, self.batch_size, logging_callback, id)
                buffers.selection_buffer.put_nowait(selection)

            except Exception as e:
                # Handle any potential exceptions
                msg = f"Error in timer_tas for id {id}"
                exception = RuntimeError(msg)
                exception.__cause__ = e
                logging_callback(exception, 0)
                raise exception

        # Register it as a task
        asyncio.create_task(timer_task())

    def __call__(self,
                 buffers: AsyncBuffers,
                 request: ActionRequest,
                 logging_callback: LoggingCallback,
                 )->Future:
        """
        The main registration method for placing requests into the
        processing stream to be handled. We should create a future in response
        to the request, which we return, and insert everything else where it needs
        to go.

        :param request: The request to register
        :param logging_callback: The logging callback
        """

        # Handle insertion into the requests buffering
        # features.

        uuid = request.state_tracker.id
        future, future_callback = self.future_factory(logging_callback)
        buffers.cases_buffer[uuid] = self.data_extractor(request)
        buffers.callbacks_buffer[uuid] = future_callback

        # If we have gone above the threshold, handle a batch
        if len(buffers.cases_buffer) >= self.buffer_threshold:
            msg = f"""
            Buffer threshold reached. Triggering batch build.
            Buffer size: {len(buffers.cases_buffer)}
            batch size: {self.batch_size}
            """
            msg = textwrap.dedent(msg)
            logging_callback(msg, 3)
            selection = self.batch_strategy(buffers.cases_buffer, self.batch_size, logging_callback, None)
            buffers.selection_buffer.put_nowait(selection)

        # Set a timeout to ensure each cases is eventually processed, then return the
        # future

        self.set_timeout_trigger(uuid, request.state_tracker.timeout, buffers, logging_callback)

        return future


class AsyncBatchProcessor(nn.Module):
    """
    The primary process nurse, the async
    batch processor layer is responsible for
    both containing the code that a dispatcher
    can interact with, alongside containing the
    code used for processing individual batches.
    """
    @classmethod
    def setup(cls,
              buffer_threshold: int,
              batch_size: int,
              batch_strategy: BatchStrategy,
              result_processor: ResultsProcessor,
              request_extractor: RequestDataExtractor,
              batch_processor: CoreSyncProcessor,
              logging_callback: LoggingCallback,
              termination_callback: TerminationCallback,
              )->"AsyncBatchProcessor":
        """
        The setup function is the primary expected user
        invoked function used to get an async batch processor
        initialized. It will take over the responsibility
        of binding the user-provided dependencies in their
        correct classes. However, in the case future modifications
        are needed, it is still possible to initialize the class
        directly

        :param buffer_threshold: Accumulate at least this many items before trying to make a batch
        :param batch_size: Target batches of this size.
            WARNING, not all batch strategies will consistently produce batches of this size

        :param batch_strategy: The BatchStrategy instance, which selects uuids to form into batches
        :param result_processor: The processor for results from the main run
        :param request_extractor: The processor that creates data to feed into the main run
        :param batch_processor: The actual main and batching mechanism. It too has builders
        :param logging_callback: The logging callback. Self-explanatory.
        :param termination_callback: A callback that returns true when it is time to end the session
                                     and clean up resources.
        :return: An instanced async batch processor
        """

        # Create the common buffers
        buffers = AsyncBuffers(
            cases_buffer = {},
            callbacks_buffer = {},
            selection_buffer = asyncio.Queue()
        )

        # Create the futures factory
        futures_factory = FuturesFactory(result_processor)

        # Create the inserter
        inserter = RequestInserter(buffer_threshold,
                                   batch_size,
                                   futures_factory,
                                   batch_strategy,
                                   request_extractor
                                   )

        # Instance and return
        return cls(buffers,
                   inserter,
                   batch_processor,
                   logging_callback,
                   termination_callback
                    )

    def __init__(self,
                 buffers: AsyncBuffers,
                 inserter: RequestInserter,
                 processor: CoreSyncProcessor,
                 logging_callback: LoggingCallback,
                 termination_callback: TerminationCallback
                 ):
        """

        :param buffers: The primary data structures we keep informtio nin
        :param inserter: The primary mangement feature used to handle requests and insert into buffers
        :param processor: The primary neural main processing and batching mechanism.
        :param logging_callback: A callback used for logging
        :param termination_callback: A callback. When it returns false, we kill off all
               infinite async loops.
        """
        super().__init__()
        self.buffers = buffers
        self.inserter = inserter
        self.processor = processor
        self.logging_callback = logging_callback
        self.termination_callback = termination_callback

    async def process_loop(self):
        """
        Asyncronous batch processing loop.

        This must be setup in a separate task, or thread, and
        will process batches then split them up and return the
        results to futures using callbacks.
        """
        while not self.termination_callback():
            # Wait until batch is ready to be processed
            selection = await self.buffers.selection_buffer.get()

            # Process batch
            outcome = self.processor(selected_cases = selection,
                                     case_buffer = self.buffers.cases_buffer,
                                     logging_callback = self.logging_callback
                                     )

            # Distribute results to callbacks
            for uuid in outcome:
                if uuid in self.buffers.callbacks_buffer:
                    callback = self.buffers.callbacks_buffer.pop(uuid)
                    callback(outcome[uuid])
                else:
                    msg = f"""
                    Unrecoverable error encountered. Callback for '{uuid}' was not found,
                    which should be impossible.
                    """
                    msg = textwrap.dedent(msg)
                    exception = RuntimeError(msg)
                    self.logging_callback(exception, 0)
                    self.termination_callback(True)
                    raise exception
    def forward(self, request: ActionRequest) -> Future:
        """
        Main forward method. This will append the action
        request to the processing stream, and return an awaitable
        future

        :param request: An action request we wish to see completed
        :return: A future. Await it, and access it's value for an action request.
        """
        return self.inserter(self.buffer,
                             request,
                             self.logging_callback)


