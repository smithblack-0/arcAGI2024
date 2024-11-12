import functools
import torch
from datasets import DatasetDict, Dataset
from transformers import PreTrainedTokenizer
from torch.utils import data
from typing import Dict, List, Tuple, Callable


class LoaderConfig
def create_pretokenized_datasets(datasets: DatasetDict,
                                 tokenizer: PreTrainedTokenizer
                                 ) -> DatasetDict:
    """
    Creates a pretokenized datasets collection
    :param datasets: A huggingface datasets collection of text.
    :param tokenizer: The tokenizer to use for this process
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

    datasets = datasets.map(batch_tokenize, batched=True, remove_columns="text")
    return datasets


def data_collator(batch: List[Dict[str, List[int]]],
                  tokenizer: PreTrainedTokenizer,
                  truncate_length: int,
                  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    batch = [[tokenizer.bos_token_id] + item + [tokenizer.eos_token_id] for item in batch]  # ids
    batch = [item[:truncate_length] for item in batch]
    batch = [{"input_ids": item} for item in batch]

    # Produce the encodings
    encodings = tokenizer.pad(batch,
                              padding=True,
                              return_tensors='pt',
                              return_attention_mask=True)

    input_ids = encodings['input_ids']
    batch_mask = encodings['attention_mask']

    # Produce the targets and inputs
    inputs = input_ids[..., :-1]
    targets = input_ids[..., 1:]
    batch_mask = batch_mask[..., :-1].to(dtype=torch.bool)

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
    collater = functools.partial(data_collator, tokenizer=tokenizer, truncate_length=truncate_length)
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
                              truncate_length: int,
                              batch_size: int,
                              tokenizer: PreTrainedTokenizer,
                              datasets: DatasetDict
                              ) -> Callable[[int], Dict[str, data.DataLoader]]:
    """
    Creates a multiprocessing factory that is designed to produce
    data loader factories compatible with a particular device
    :param total_workers: The total number of workers in the env
    :param truncate_length: The length to truncate to
    :param batch_size: The batch size to use
    :param tokenizer: The tokenizer to use
    :param datasets: The datasets to use
    :return: The loaders. 'train_loader', 'test_loader', 'validation_loader'
    """
    return functools.partial(make_dataloaders,
                             total_workers=total_workers,
                             truncate_length=truncate_length,
                             batch_size=batch_size,
                             tokenizer=tokenizer,
                             datasets=datasets
                             )
