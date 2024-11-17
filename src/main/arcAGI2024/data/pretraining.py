import functools
import torch
import datasets
import numpy as np
from dataclasses import dataclass, field
from datasets import DatasetDict, Dataset, load_dataset
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from torch.utils import data
from typing import Dict, List, Tuple, Callable, Union, Optional, Any
from ..vocabulary import Vocabulary


@dataclass
class PretrainingLoaderConfig:
    """
    A configuration class for holding configuration
    details for a pretraining loader, sans external
    dependencies
    """
    huggingface_dataset_name: str
    huggingface_dataset_version: str
    batch_size: int
    truncate_length: int
    loader_kwargs: Dict[str, Any] = field(default_factory= lambda : {})
def create_pretokenized_datasets(datasets: DatasetDict,
                                 tokenizer: PreTrainedTokenizer,
                                 use_cache: bool = True,
                                 ) -> DatasetDict:
    """
    Creates a pretokenized datasets collection
    :param datasets: A huggingface datasets collection of text.
    :param tokenizer: The tokenizer to use for this process
    :param use_cache: Whether to use the pretokenized cache if available, or force a rebuild
    :return: The pretokenized dataset.
    """

    def batch_tokenize(examples: Dataset):
        texts = examples["text"]
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
                     truncate_length: int,
                     batch_size: int,
                     tokenizer: PreTrainedTokenizer,
                     datasets: DatasetDict
                     ) -> Dict[str, data.DataLoader]:
    """
    Creates the dataloaders for the training, validation, test datasets.

    :param worker_rank: The number associated with this particular worker
    :param total_workers: The total number of workers
    :param truncate_length: The length to truncate to
    :param batch_size: The batch size to use
    :param tokenizer: The tokenizer to use.
    :param datasets: Has a 'test', 'train', 'validation' split with 'tokens' features in the validators.
    :return: The setup dataloaders, one for each split
    """
    collater = functools.partial(data_collator, pad_id=tokenizer.pad_token_id, truncate_length=truncate_length)
    loaders: Dict[str, data.DataLoader] = {}
    for dataset in datasets:
        sampler = data.DistributedSampler(dataset, total_workers, worker_rank)
        loader = data.DataLoader(dataset,
                                 batch_size,
                                 sampler=sampler,
                                 shuffle=True,
                                 collate_fn=collater,
                                 num_workers=1,
                                 prefetch_factor=2,
                                 pin_memory=True,
                                 )
        loaders[dataset] = loader
    return loaders


def create_dataloader_factory(total_workers: int,
                              tokenizer: PreTrainedTokenizer,
                              config: PretrainingLoaderConfig
                              ) -> Callable[[int], Dict[str, data.DataLoader]]:
    """
    Creates a multiprocessing factory that is designed to produce
    data loader factories compatible with a particular device
    :param total_workers: The total number of workers in the env
    :param tokenizer: The tokenizer to use
    :param config: The loader config to use
    :return: The loaders. 'train_loader', 'test_loader', 'validation_loader'
    """
    data = load_dataset(config.huggingface_dataset_name, config.huggingface_dataset_version, **config.loader_kwargs)
    return functools.partial(make_dataloaders,
                             total_workers=total_workers,
                             truncate_length=config.truncate_length,
                             batch_size=config.batch_size,
                             tokenizer=tokenizer,
                             datasets=data
                             )
