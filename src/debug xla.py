from src.main import arcAGI2024
import torch
from torch.profiler import profile, ProfilerActivity
import pynvml
import threading
from contextlib import ContextDecorator
import time

def explore_models_with_profiling():
    # Setup model core on TPU
    model_core = arcAGI2024.CausalLMCore.build_model_on_top_of_pretrained_head(
        head_model_name="gpt2",
        num_layers=10,
        num_read_heads=10,
        num_write_heads=10,
        num_memories=60,
        dropout_rate=0.1,
        auxilary_dropout_rate=0.1
    ).to(xm.xla_device())  # Send model to XLA device
    torch_xla2.experimental.eager_mode(True)

    # Initialize loss functions
    main_loss_fn = arcAGI2024.CrossEntropyLoss(model_core.vocabulary.tokenizer.pad_token_id)
    mem_access_loss_fn = arcAGI2024.UniformMemLoss()

    # Initialize the CausalLMTrainer on XLA device
    trainer = arcAGI2024.CausalLMTrainer(
        model_core=model_core,
        main_loss_function=main_loss_fn,
        mem_access_loss_function=mem_access_loss_fn,
        rescaler_mode="mean",
        verbose=True,
        empty_cuda_cache=False
    ).to(xm.xla_device())

    # Create mock training data
    batch_size = 3
    num_tokens = 3
    cache_rate = 20
    tokens = torch.randint(0, model_core.vocabulary.tokenizer.true_vocab_size, (batch_size, num_tokens)).to(xm.xla_device())
    targets = torch.randint(0, model_core.vocabulary.tokenizer.true_vocab_size, (batch_size, num_tokens)).to(xm.xla_device())
    masks = (torch.rand(batch_size, num_tokens) > 0.5).to(xm.xla_device())

    # Compute schedule rate
    num_active_tokens = (~masks).sum().to(dtype=torch.float32)
    schedule_rate = 1 / num_active_tokens
    print(schedule_rate.to(torch.device("cpu")))
    schedule_rate = float(schedule_rate)

    try:
        # Run the training step
        memories, metrics = trainer.step(
            tokens, targets, masks,
            numerics_cache_rate=cache_rate,
            save_cached_to_cpu=True,
            scheduling_rates=(schedule_rate, schedule_rate),
        )
        # Output profiling results
        print(metrics)

    finally:
        # Cleanup
        del trainer
        del model_core
        del tokens
        del targets
        del masks
        if "memories" in locals():
            del memories
        if "metrics" in locals():
            del metrics
        xm.mark_step()  # Sync TPU

# Execute with XLA multiprocessing if required
explore_models_with_profiling()
