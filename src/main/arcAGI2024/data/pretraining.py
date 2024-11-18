import functools
import torch
import datasets
import numpy as np
from dataclasses import dataclass, field
from datasets import DatasetDict, Dataset, load_dataset
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from torch.utils import data
from typing import Dict, List, Tuple, Callable, Union, Optional, Any
from .base import make_buffered_pipeline, LoaderConfig, register_loader_factory, ProcessLoaderBinder

@dataclass
class PretrainingLoaderConfig(LoaderConfig):
    """
    A configuration class for holding configuration
    details for a pretraining loader, sans external
    dependencies

    Fields (required)

    huggingface_dataset_name: The name of the huggingface dataset
    huggingface_dataset_version: The version to use
    dataset_text_name: The schema name of the text in the dataset.
    batch_size: The batch size. Generally quite large
    truncate_length: The truncate length. Longer than this is truncated

    Fields (optional)

    shuffle: Whether to shuffle. Defaults to true
    num_batches_in_buffer: How many batches to keep in the lengths buffer.
                           Since we are dealing with wildly varying lengths,
                           and enormous batches that amplify any issues,
                           some special logic attempts to minimize the
                           amount of needed padding
    num_prefetch_threads: Exactly what it saids
    prefetch_factor: How many batches to keep in a prefetched state as tensors.
    huggingface_loader_kwargs: Any additional arguments the huggingface loader needs
    """
    huggingface_dataset_name: str
    huggingface_dataset_version: str
    dataset_text_name: str
    batch_size: int
    truncate_length: int
    shuffle: bool = True
    num_batches_in_buffer: int = 20
    num_prefetch_threads: int = 4
    prefetch_factor: int = 4
    huggingface_loader_kwargs: Dict[str, Any] = field(default_factory= lambda : {})

def create_pretokenized_datasets(datasets: DatasetDict,
                                 tokenizer: PreTrainedTokenizer,
                                 text_storage_name: str,
                                 use_cache: bool = True,
                                 ) -> DatasetDict:
    """
    Creates a pretokenized datasets collection
    :param datasets: A huggingface datasets collection of text.
    :param tokenizer: The tokenizer to use for this process
    :param text_storage_name: The name text is stored under in the dataset
    :param use_cache: Whether to use the pretokenized cache if available, or force a rebuild
    :return: The pretokenized dataset.
    """

    def batch_tokenize(examples: Dataset):
        texts = examples[text_storage_name]
        encodings = tokenizer(
            text=texts,
            truncation=False,
            padding=False,
            return_attention_mask=False,
        )
        return encodings

    datasets = datasets.map(batch_tokenize, batched=True, remove_columns="text", load_from_cache_file=use_cache)
    return datasets


def data_collator(batch: List[Dict[str, List[int]]],
                  pad_id: int,
                  truncate_length: int,
                  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    A function designed to be bound to using functools.

    It will finish padding and tokenization.

    :param batch: The batch under consideration
    :param pad_id: The integer to use when padding.
    :param truncate_length: What to truncate if we go beyond
    :return: The tokens, targets, and attn mask.
    """
    batch_ids = [item["input_ids"] for item in batch]

    # Figure out the padding/truncate lengt
    target_length = max(len(item) for item in batch_ids)
    target_length = min(truncate_length, target_length)

    # Perform the padding/truncate process
    final_ids = []
    final_masks = []
    for case in batch_ids:
        # Encode the mask and inputs array. Make sure
        # to truncate if needed.
        ids_array = torch.tensor(case[:target_length], dtype=torch.long)
        mask_array = torch.ones_like(ids_array, dtype=torch.bool)

        # Integrate the padding to whatever degree is needed
        ids_padding = torch.full([target_length - len(ids_array)], pad_id, dtype=torch.long)
        mask_padding = torch.zeros_like(ids_padding, dtype=torch.bool)

        # Combine and store
        ids_output = torch.concat([ids_array, ids_padding], dim=-1)
        attn_mask = torch.concat([mask_array, mask_padding], dim=-1)

        final_ids.append(ids_output)
        final_masks.append(attn_mask)

    input_ids = torch.stack(final_ids, dim=0)
    attn_mask = torch.stack(final_masks, dim=0)

    # Produce the targets and inputs
    inputs = input_ids[..., :-1]
    targets = input_ids[..., 1:]
    batch_mask = attn_mask[..., :-1].to(dtype=torch.bool)

    # Pin

    inputs.pin_memory()
    targets.pin_memory()
    batch_mask.pin_memory()

    # Return
    return inputs, targets, batch_mask


def make_dataloaders(worker_rank: int,
                     total_workers: int,
                     config: PretrainingLoaderConfig,
                     padding_id: int,
                     datasets: DatasetDict
                     ) -> Dict[str, data.DataLoader]:
    """
    Creates the dataloaders for the given dataset dictionaries.

    :param worker_rank: The rank of the worker under consideration
    :param total_workers: The total number of workers
    :param config: The config we are working
    :param padding_id: The integer to use when padding.
    :param datasets: The datasets to convert
    :return: The dataloaders
    """

    bound_collate_fn = functools.partial(data_collator, pad_id=padding_id, truncate_length=config.truncate_length)
    loaders: Dict[str, data.DataLoader] = {}
    for name, dataset in datasets.items():
        loader = make_buffered_pipeline(config.batch_size,
                                        total_workers,
                                        worker_rank,
                                        bound_collate_fn,
                                        dataset,
                                        num_batches_in_buffer=config.num_batches_in_buffer,
                                        shuffle=config.shuffle,
                                        num_prefetch_threads=config.num_prefetch_threads,
                                        prefetch_factor=config.prefetch_factor
                                        )
        loaders[name] = loader
    return loaders


def create_dataloader_factory(total_workers: int,
                              tokenizer: PreTrainedTokenizer,
                              config: PretrainingLoaderConfig
                              ) -> ProcessLoaderBinder:
    """
    Creates a multiprocessing factory that is designed to produce
    data loader factories compatible with a particular device
    :param total_workers: The total number of workers in the env
    :param tokenizer: The tokenizer to use
    :param config: The loader config to use
    :return: The loaders. 'train_loader', 'test_loader', 'validation_loader'
    """

    # Setup the tokenized datasets
    raw_dataset = load_dataset(config.huggingface_dataset_name,
                               config.huggingface_dataset_version,
                               **config.huggingface_loader_kwargs)
    pretokenized_dataset = create_pretokenized_datasets(raw_dataset, tokenizer, config.dataset_text_name)

    # Bind the factory. It now is looking only for the
    # worker number in order to finish initializing.

    return functools.partial(make_dataloaders,
                             total_workers=total_workers,
                             config=config,
                             padding_id=tokenizer.pad_token_id,
                             datasets=pretokenized_dataset
                             )

register_loader_factory(LoaderConfig, create_dataloader_factory)