from src.main import arcAGI2024
import torch
from torch.profiler import profile, ProfilerActivity
import pynvml
import threading
from contextlib import ContextDecorator
import time
class GPUMonitor(ContextDecorator):
    def __init__(self, interval=5.0):
        """
        Initializes GPU monitoring with a specified interval.

        Args:
            interval (float): Interval in seconds for monitoring.
        """
        self.interval = interval
        self.stop_event = threading.Event()

    def __enter__(self):
        # Initialize NVML and start monitoring thread
        pynvml.nvmlInit()
        self.monitor_thread = threading.Thread(target=self._monitor)
        self.monitor_thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Stop monitoring and shutdown NVML
        self.stop_event.set()
        self.monitor_thread.join()
        pynvml.nvmlShutdown()
        print("GPU monitoring stopped.")

    def _monitor(self):
        while not self.stop_event.is_set():
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Assuming single-GPU instance

            # Query GPU usage stats
            gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

            # Print GPU stats
            print(f"GPU Utilization: {gpu_util.gpu}%")
            print(f"Memory Utilization: {gpu_util.memory}%")
            print(f"Memory Used: {memory_info.used / 1024**2:.2f} MB")
            print(f"Total Memory: {memory_info.total / 1024**2:.2f} MB")
            print(f"Free Memory: {memory_info.free / 1024**2:.2f} MB")
            print("-" * 40)

            # Wait for the specified interval or until the stop event is set
            time.sleep(self.interval)
def explore_models_with_profiling():
    # Setup model core
    model_core = arcAGI2024.CasualLMCore.build_model_on_top_of_pretrained_head(
        head_model_name="gpt2",
        num_layers=10,
        num_read_heads=10,
        num_write_heads=10,
        num_memories=100,
        dropout_rate=0.1,
        auxilary_dropout_rate=0.1
    )

    # Initialize loss functions
    main_loss_fn = arcAGI2024.CrossEntropyLoss(model_core.vocabulary.tokenizer.pad_token_id)
    mem_access_loss_fn = arcAGI2024.UniformMemLoss()

    # Initialize the CausalLMTrainer
    trainer = arcAGI2024.CausalLMTrainer(
        model_core=model_core,
        main_loss_function=main_loss_fn,
        mem_access_loss_function=mem_access_loss_fn,
        verbose=True
    )

    # Create mock training data
    batch_size = 30
    num_tokens = 20
    cache_rate = 50
    tokens = torch.randint(0, model_core.vocabulary.tokenizer.true_vocab_size, (batch_size, num_tokens))
    targets = torch.randint(0, model_core.vocabulary.tokenizer.true_vocab_size, (batch_size, num_tokens))
    masks = (torch.rand(batch_size, num_tokens) > 0.5)

    # Move to cuda
    tokens = tokens.to("cuda")
    targets = targets.to("cuda")
    masks = masks.to("cuda")
    trainer = trainer.to("cuda")

    # Run steps with profiler
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True, profile_memory=True) as prof:
        memories, numeric_metrics, loss = trainer.step(tokens, targets, masks, numerics_cache_rate=cache_rate)

    # Output profiling results
    print(numeric_metrics)
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))

    # Cleanup
    del prof
    del trainer
    del model_core
    del tokens
    del targets
    del masks
    del memories
    del numeric_metrics
    torch.cuda.empty_cache()

explore_models_with_profiling()

