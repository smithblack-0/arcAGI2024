"""
Define the core async batch processor classes we can utilize
"""
import asyncio
import textwrap
import warnings
from dataclasses import dataclass

import torch
from torch import nn
from torch.futures import Future
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Callable, List, Optional
from .types import BatchCaseBuffer, LoggingCallback, TerminationCallback
from .clustering import ClusteringStrategy
from .core_processer import BatchAssembly, BatchDisassembly
from ..data import ActionRequest

## Basic batch strategy classes
#
# These primarily have the responsibility of getting the statistics
# that the above clustering strategies need. At the moment, we base
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
    def get_vital_statistics(self, case_buffer: BatchCaseBuffer)->torch.Tensor:
        """
        An extremely important method, this gives us vital information on
        the various items in the request buffer. It needs to be compatible
        with the clustering mechanism.

        Exact details will vary depending on how vital statistics are extracted
        :return:
        """
        pass

    @abstractmethod
    def batch_strategy_factory(self,
                               **parameters
                               )->Callable[[ClusteringStrategy], "BatchStrategy"]:
        """
        An implementation of a batch strategy factory. It will frequently
        be the case that we would like to inject the clustering strategy
        later on.

        The batch strategy factory should setup a factory that
        can be called with your implementation-specific parameters,
        and will return a partial that can be invoked with a clustering
        strategy.


        **parameters: Your invokation parameters
        :return: A callable for producing a BatchStrategy when given a clustering strategy.
        """
        pass


    def __init__(self,
                 clustering_strategy: ClusteringStrategy
                 ):
        self.clustering_strategy = clustering_strategy



    def __call__(self,
                 batch_case_buffer: BatchCaseBuffer,
                 batch_size: int,
                 logging_callback: LoggingCallback,
                 force_selection: str,
                 )->List[str]:
        vitals = self.get_vital_statistics(batch_case_buffer)
        output = self.clustering_strategy(vitals, batch_size, logging_callback, force_selection)
        return output


###
# Request Receiver
#
# The request processor is responsible for moving requests into
# the batch case buffer. It is a mix of an interface and, if it ever
# matters, a cache in and of itself.
###

class RequestReceiver(ABC):
    """
    Request Receiver

    The request processor is responsible for moving requests into
    the batch case buffer. It is a mix of an interface and, if it ever
    matters, a cache in and of itself.

    It moves and processes an incoming request, getting all callbacks
    and such ready.
    """
    def __init__(self,
                 cases_buffer: BatchCaseBuffer,
                 callbacks_buffer:
                 ):
        self.cases_buffer = cases_buffer

    @abstractmethod
    def process_request(self, request: ActionRequest)->Dict[str, torch.Tensor]:
        """
        The method the implementer must implement for the class to function

        It must take an action request and turn it into a dictionary of tensors.
        These tensors will then be the only ones available downstream in the model.
        :param request: An action request
        :return: A dictionary of tensors, like {"shape" : tensor, "targets" : data}
        """
        pass
    def __call__(self,
                 request: ActionRequest,
                 logging_callback: LoggingCallback
                 )->Future:
        """
        The main registration method for placing requests into the
        processing stream to be handled. We should create a future in response
        to the request,
        :param request:
        :return:
        """


###
# Results processor
#
# The result processor system is designed to take the
# info produced by the model and turn it into another
# action request the system can use.
###
class ResultsProcessor(ABC):
    """
    The results processor is intended to
    take within it the results of executing
    the core neural network model, and
    several important factors, then return
    an ActionRequest letting us know what
    to do next.

    This is an abstract method. The method
    "process_model_results" must be implemented before
    the class can work
    """

    @abstractmethod
    def process_model_results(self,
                              next_destination: str,
                              request: ActionRequest,
                              results: Dict[str, torch.Tensor],
                              ) -> ActionRequest:
        """
        The required abstract method. It should accept
        an action request, the results of processing the request,
        and integrate them into a novel action request. Notably,
        the destination to use is also provided

        :param next_destination: The next action request destination. Send it here!
        :param request: The original action request
        :param results: The results of running the request
        :return: The revised action request that needs to be returned
        """
        pass

    def __call__(self,
                 next_destination: str,
                 request: ActionRequest,
                 results: Dict[str, torch.Tensor],
                 ) -> ActionRequest:
        """
        When invoked, this will use the provided information
        to compute the next action request.

        :param next_destination: The next destination to dispatch to. Make sure to match it!
        :param request: The original action request
        :param results: The results of running the request
        :return: A new action request
        """
        action_request = self.process_model_results(next_destination, request, results)
        assert action_request.state_tracker.destination == next_destination
        return action_request



## The main async machine, and some of the construction dataclasses
#
# This is the actual Async Batch Processor layer that gets
# most of the logic done.

class AsyncBatchProcessor(nn.Module):
    """
    *Purpose**:

    The central orchestrator for managing the batching and processing
    workflow. It is dependency-injected with a Batching Strategy, Batch Assembly,
    Batch Disassembly, and manages the request_buffer and batch_queue. It both
    contains code for the batch processing subloop, and the pseudounbatched main
    processing code

    As far as information flowing in through the main forward method is concerned,
    processing occurs in a linear manner, without batches, but with an indeterminate
    await time. Meanwhile, the processing subloop will await until batches are available
    at which point it will run.

    It contains inside a collection of requests from which batches can be formed to
    help ensure batches of similar length are created.

    **Attributes**:
    - **buffer_threshold**: An integer representing how big the request buffer gets
    before we form and run a batch. It must be greater than or equal to batch size.
    - **batch_size**: An integer representing the size of batches that the system
    will attempt to form. While a batch smaller than this size can be run, a
    batch larger than this size will never be formed.
    - **request_buffer**: A dictionary that maps UUIDs to request tuples
    (`future_callback, ActionRequest`).
    - **batch_queue**: An `asyncio.Queue` that stores batches ready for processing.
    - **batching_strategy**: The Batching Strategy instance for selecting requests
    to form a batch.
    - **batch_assembly**: The Batch Assembly instance for assembling batches from
    selected requests.
    - **batch_disassembly**: The Batch Disassembly instance for disassembling
    processed batches and fulfilling futures/callbacks.
    - **model**: The model that actually processes the batch. It must accept a batched
              dictionary, and return a batched dictionary.
    - **logging_callback**: A callback function used for logging purposes. If left
    as `None`, logging is emitted as a warning.

    **Methods**:
    - **__call__**: Accepts an `ActionRequest` and returns a `Future`. It performs
    the following sequence of actions:
    - Registers the request in the request_buffer.
    - Checks if the request_buffer size exceeds the buffer_threshold.
    - If it does, it forms a batch using the `create_batch` helper method and
    stores it in the batch_queue.
    - Sets up a timer for the request's timeout using the `set_timeout_trigger`
    helper method.
    - Returns the `Future` to the caller.

    - **processing_loop**: This method runs in a separate thread or as an
    asynchronous loop.
    - Waits for batches from the batch_queue.
    - Calls the `model` with the assembled batch.
    - Passes the results to Batch Disassembly for distribution to the original
    requests.


    **Helper Methods**
    - **create_batch**: Accepts the request_buffer, a batch size, and any UUIDs to
    force inclusion. This method first runs the Batching Strategy to select the
    requests, then passes the selected UUIDs to the Batch Assembly to form the
    batch. It returns the resulting batch and metadata list. This method does
    not place the batch into the batch_queue—that is the responsibility of the
    caller.

    - **set_timeout_trigger**: A helper method designed to be registered as part of
    an I/O loop. It should be called with a UUID, the request_buffer, and a
    timeout in milliseconds. Upon invocation, it creates and activates a
    function in `asyncio` that waits for the specified timeout. If the UUID is
    still in the request_buffer after the timeout, it calls `create_batch` with
    the UUID as a forcing entry, then places the completed batch into the
    batch_queue.

    """
    ###
    #
    # Define several very important helper methods
    #
    ###

    def select_batch(self, force_inclusion):

    def create_batch(self,
                     force_inclusion: Optional[str] = None
                     )->Tuple[List[MetadataPayload], Dict[str, torch.Tensor]]:
        """
        Create a batch out of the current resources. If needed, force
        the inclusion of the given case in the batch

        :param force_inclusion: The case to force inclusion on. Optional
        :return: A batch that is ready for processing
        """
        try:
            selection = self.batching_strategy(self.requests_buffer,
                                               self.batch_size,
                                               self.logging_callback,
                                               force_inclusion
                                               )
        except Exception as e:
            msg = f"""

            
            Unable to choose the elements of the buffer to form
            into a batch. An issue occurred. The cases in the 
            reqeust buffer were:            
            {[key for key in self.requests_buffer.keys()]}
            """
            msg = textwrap.dedent(msg)
            raise AsyncTerminalException(msg,
                                 requests_buffer = self.requests_buffer,
                                 ) from e

        try:
            metadata, batch = self.batch_assembly(self.requests_buffer,
                                                  selection,
                                                  self.logging_callback
                                                  )
            return metadata, batch
        except Exception as e:
            msg = f"""
            
            """

        batch = self.batch_assembly(self.requests_buffer, selection, self.logging_callback)
        return batch

    def future_callback_factory(self,
                               future: Future,
                               request: ActionRequest,
                               )->Callable[[Dict[str, torch.Tensor]], None]:
        """
        Creates a callback designed to fufill a future
        with an action request.

        :param future: The future to fufill
        :param request: The request we will need to process
        :return:
        """
        def future_callback(data: Dict[str, torch.Tensor]):
            try:
                # Get the next action request
                new_request = self.action_factory(
                                    request.state_tracker.id,
                                    request,
                                    self.logging_callback,
                                    data
                                    )
                future.set_result(new_request)
            except Exception as e:
                future.set_exception(e)

        return future_assignment_callback


    def set_timeout_trigger(self,
                            id: str,
                            timeout: float):
        """
        Sets a timeout trigger on the async task collection. In the event that
        a case is not handled by the timeout, we force its run.

        :param id: The id to force completion on timeout, if it still has not been handled
        :param timeout: The timeout to wait for, in milliseconds.
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
                if id not in self.requests_buffer:
                    return

                # Create and store batch
                metadata, batch = self.create_batch(id)
                self.batch_queue.put_nowait((metadata, batch))

            except Exception as e:
                # Handle any potential exceptions
                print(f"Error in timer_task for id {id}: {e}")

        # Register it as a task
        asyncio.create_task(timer_task())

    def register_request(self,
                         factory_callback: ActionConstructionCallback,
                         request: ActionRequest):
        """
        Registers an action request in the request buffer and,
        should it be needed, forms a batch. This also includes setting
        up the timeouts

        :param factory_callback: A callback that can be run to construct the next action request
        :param request: The action request to register.
        """
        ##
        # We basically have two jobs to do here.
        #
        # First, we need to actually make sure to register into the buffer
        # the request. Second, if the buffer is full, we need to go make
        # ourselves a batch.
        #
        ##


        # Store the request in the dictionary.
        id = request.state_tracker.id
        self.requests_buffer[id] = (factory_callback, request)

        # If needed, make and store a batch.
        #
        # All code here MUST be syncronous to avoid certain bugs.
        if len(self.requests_buffer) >= self.buffer_threshold:
            metadata, batch = self.create_batch()
            self.batch_queue.put_nowait((metadata, batch))

        # Setup the timeout.
        self.set_timeout_trigger(id, request.state_tracker.timeout)


    ## Init ##
    #
    # The majority of init is dependency-injected to follow
    # best practices.
    def __init(self,
               buffer_threshold: int,
               batch_size: int,
               batching_strategy: BatchStrategy,
               batch_assembly: BatchAssembly,
               model_core: nn.Module,
               batch_disassembly: BatchDisassembly,
               action_factory: ActionFactory,
               termination_callback: TerminationCallback,
               logging_callback: Optional[LoggingCallback] = None
               ):
        """
        Constructor for the AsynBatchProcessor class

        :param buffer_threshold: The buffer threshold. When the buffer goas above this, we make a new batch
        :param batch_size: The target batch size to reach
        :param batching_strategy: The batching strategy dependency
        :param batch_assembly: The batching assembly dependency
        :param model_core: The main machine learing model
        :param batch_disassembly: The batching disassembly dependency
        :param action_factory: The action factory dependency
        :param termination_callback: The termination callback

        """
        # Setup logging if needed
        if logging_callback is None:
            warnings.warn("Logging callback was not provided. Improvising, with verbosity of 1")
            def logging_callback(message, verbosity):
                if verbosity <= 1:
                    print(message)

        # Setup some resources

        self.requests_buffer: RequestBuffer = {}
        self.batch_queue: asyncio.Queue = asyncio.Queue()

        # Store dependencies for later use

        assert buffer_threshold >= batch_size
        self.buffer_threshold = buffer_threshold
        self.batch_size = batch_size
        self.batching_strategy = batching_strategy
        self.batch_assembly = batch_assembly
        self.model_core = model_core
        self.batch_disassembly = batch_disassembly
        self.action_factory = action_factory
        self.logging_callback = logging_callback

    ## Main methods
    #
    # Primary methods are handled here.
    def forward(self, request: ActionRequest) -> Future:
        """
        An invokation of the async batch processor layer,
        requesting the return of the next action request.

        We receive in response a future which is promised
        to contain the next Action Request in the sequence

        :param request:
            The action request we are being asked to fufill
        :return: A future which will contain an action request
        """

        future = Future()
        callback = self.future_callback_factory(future, request)
        self.register_request(callback, request)
        return future

    async def process_loop(self):
        """
        Asyncronous batch processing loop.

        This must be setup in a separate task, or thread, and
        will process batches then split them up and return them
        using callbacks
        """

        while True:
            try:
                # Wait until a batch needs to be processed.
                #
                # Then run the batch, and dispatch the results

                metadata, batch = await self.batch_queue.get()
                batch_output = self.model_core(batch)
                self.batch_disassembly(metadata, batch_output)

            except Exception as e:
                #TODO: propogate exceptions back into the futures that are waiting
                raise e







