# Updated Proposal for Async Batch Processing with Logging Callback

## Key Components and Responsibilities:

### 1. Request Buffer
**Purpose**: The Request Buffer is a dictionary that holds pending requests. It 
allows the system to efficiently select and form batches by maintaining a pool 
of requests from which batches can be assembled. This structure supports 
handling auxiliary tensors, such as targets, alongside mandatory embeddings.

**Type**: `Dict[UUID, Tuple[Future, ActionRequest]]`

- **UUID**: A unique identifier for each request.
- **Tuple Contents**:
  - **future_callback (Future)**: A future or callback function that will be 
    fulfilled once the processing of this request is complete.
  - **ActionRequest**: The request object containing the tensors and other 
    relevant data for processing.

### 2. Batch Queue
**Purpose**: An `asyncio.Queue` instance that manages the queue of batches 
ready for processing. This queue is awaitable, meaning that it can block until 
an item is available for processing, making it suitable for asynchronous 
operations.

### 3. Batching Strategy
**Purpose**: Responsible for selecting which requests from the Request Buffer 
should be included in a batch when a batch is needed.

**Inputs**:
- **request_buffer**: The Request Buffer containing pending requests.
- **batch_size**: The desired size of the batch.
- Optionally, a list of UUIDs that must be included in the next batch.
- **logging_callback**: A callback function for logging, which accepts a 
  message and a verbosity level.

**Output**: A list of UUIDs corresponding to the requests selected for the 
batch.

**Responsibility**: Handles the logic for selecting requests to form a batch 
based on criteria like minimizing padding or balancing load across different 
requests. It does not handle the actual formation of the batch (i.e., combining 
tensors) but only the selection of which requests should be included.

**Implementation**: The Batching Strategy reads the `.subtask_details` 
dictionary field of an `ActionRequest` to access the raw data (tensors) that 
can be used to make batching decisions. By analyzing the shapes of these 
tensors, the strategy can predict which combinations of requests will require 
the least padding or minimize computation issues.

### 4. Batch Assembly
**Purpose**: Responsible for assembling a batch of data from the selected 
requests.

**Inputs**:
- **request_buffer**: The Request Buffer containing pending requests.
- **uuids**: A list of UUIDs selected by the Batching Strategy.
- **logging_callback**: A callback function for logging, which accepts a 
  message and a verbosity level.

**Output**:
- **batch**: A fully formed batch, which could be a tensor (or a set of 
  tensors) ready for processing.
- **metadata**: A list of `List[Tuple[UUID, Callable]]`, where each inner list 
  corresponds to an entry in the batch and associates the UUIDs with their 
  callbacks.

**Responsibility**: Pops the selected elements out of the Request Buffer, 
organizes them into a batch, and prepares them for processing. Different 
implementations might be needed for training (which requires targets) and 
evaluation (which may not). The Batch Assembly is solely responsible for the 
logic of forming batches (e.g., combining tensors into a unified batch with 
common batch indices).

### 5. Batch Disassembly
**Purpose**: Responsible for taking the processed batch results and 
distributing them back to the individual requests.

**Inputs**:
- **metadata**: A list of `List[Tuple[UUID, Callable]]`, which provides the 
  mapping of UUIDs to callbacks for each entry in the batch.
- **batch_result**: A dictionary of batched tensors (`Dict[str, torch.Tensor]`), 
  corresponding to the results of the batch processing.
- **logging_callback**: A callback function for logging, which accepts a 
  message and a verbosity level.

**Output**: No direct output, but it fulfills the futures associated with each 
request, providing the processed results back to the original caller.

**Responsibility**: Ensures that each original request gets its corresponding 
results by disassembling the batch results. It iterates through the metadata 
and extracts individual results from the `batch_result`, creating new 
dictionaries for each request. These dictionaries are then passed to the 
corresponding callback functions, fulfilling the futures.

### 6. Async Batch Processor
**Purpose**: The central orchestrator for managing the batching and processing 
workflow. It is dependency-injected with a Batching Strategy, Batch Assembly, 
Batch Disassembly, and manages the request_buffer and batch_queue.

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

- **create_future**: Creates a `Future` and a callback function. The callback, 
  when invoked, will populate the `Future` with the results. This method allows 
  for easier management of asynchronous request handling and ensures that the 
  Future is fulfilled once the processing is complete.

### 7. Threads, Async, and Thread Safety
This section covers the concurrency aspects of the system, specifically how 
Python's asyncio and threading models are used to ensure smooth operation 
across different components.

**Threading and Concurrency**:
- **Asyncio**: The system heavily relies on Python's `asyncio` library for 
  managing asynchronous operations. The `batch_queue` is an `asyncio.Queue` 
  that allows the system to wait for items in a non-blocking manner, ensuring 
  that the processing loop can efficiently process batches as they become 
  available.

- **Processing Loop and Dispatcher**: While the `AsyncBatchProcessor`'s 
  `processing_loop` runs in one thread, the main dispatcher operates in 
  another. Futures are created and resolved across these threads, with the 
  `AsyncBatchProcessor` thread populating the `Future` that was handed off by 
  the dispatcher thread.

**Thread Safety**:
- **Futures and Thread Safety**: Futures are designed to be thread-safe in 
  Python. One thread can create a `Future` and pass it to another thread, 
  which later resolves it using `set_result` or `set_exception`. As long as 
  only one thread writes to a `Future` (e.g., by resolving it), this operation 
  remains thread-safe.

- **Timeout Handling**: Timeout functionality is managed using `asyncio` 
  timers. When a request is registered, an asynchronous timer is started. If 
  the timer expires before the request is processed, the request is forced 
  into a batch, ensuring that no requests are left unprocessed indefinitely. 
  Frequent timeouts may indicate that the `buffer_threshold` should be 
  adjusted.

This design ensures that all components interact in a thread-safe manner while 
efficiently managing asynchronous operations. The use of `asyncio` and 
threading allows the system to scale and handle high concurrency without 
running into common threading issues.
