import functools

import torch
import torch.distributed as dist
from torch import nn
from datasets import load_dataset
from src.main import arcAGI2024
import torch.multiprocessing as mp

# Setup a bunch of hyperparams
total_workers = 1
label_smoothing_rate = 0.7
learning_rate = 0.001
verbose = True

loader_config = arcAGI2024.PretrainingLoaderConfig(huggingface_dataset_name='wikitext',
                                                   huggingface_dataset_version="wikitext-2-raw-v1",
                                                   batch_size=100,
                                                   truncate_length=300,
                                                   )
model_config = arcAGI2024.CoreConfig(
    num_layers=4,
    num_read_heads=10,
    num_write_heads=10,
    num_memories=20,
    dropout_rate=0.1,
    sublayers_dropout_rate=0.04
)

# Create vocabulary. Use it to create model core and the trainer interface
vocabulary = arcAGI2024.load_vocabulary_off_huggingface_model('gpt2')
core = arcAGI2024.CausalLMCore.build_model_using_config(vocabulary, model_config)
trainer_core = arcAGI2024.StandardTrainerCore(core)

# Create rest of the training config, including the link to the datasource
training_config = arcAGI2024.TrainingConfig(
    arcAGI2024.create_dataloader_factory(total_workers, vocabulary.tokenizer, loader_config),
    "debug_run_1",
    metrics_logging_directory="/metrics",
    checkpoint_save_directory="/checkpoints",
    checkpoint_batch_frequency=100,
    num_workers=total_workers,
    num_epochs=3,
    epoch_position=0
)

# Create trainer object

cross_entropy = arcAGI2024.CrossEntropyLoss(vocabulary.tokenizer.pad_token_id,
                                            label_smoothing_rate=label_smoothing_rate)
mem_equality = arcAGI2024.UniformMemLoss()
grad_normalization = arcAGI2024.AutorescaleGradientControl()
trainer = arcAGI2024.CausalLMTrainer(trainer_core, cross_entropy, mem_equality, grad_normalization, verbose=True)


# Create factories for placing model, optim, schedule
def setup_model(device: torch.device) -> torch.nn.Module:
    return trainer.to(device)


def setup_optim(model: nn.Module) -> torch.optim.Optimizer:
    return torch.optim.Adam(model.parameters(), lr=learning_rate)


def setup_scheduler(optim: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler:
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=0.5, patience=5, verbose=True)


run_process = functools.partial(
    arcAGI2024.run_training_process,
    training_config=training_config,
    model_factory=setup_model,
    optim_factory=setup_optim,
    scheduler_factory=setup_scheduler,
)

# Setup process group

dist.init_process_group(
    backend='nccl',  # or 'gloo' for CPU-based training
    init_method='tcp://127.0.0.1:29500',  # URL specifying the initialization method
    world_size=total_workers,  # Total number of processes
    rank=0  # Rank of this process
)

mp.spawn(run_process, nprocs=total_workers, join=True)

dist.destroy_process_group()