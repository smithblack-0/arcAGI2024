from typing import List, Tuple, Dict

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset
from src.main.arcAGI2024.vocabulary import VocabularyStruct

vocab = VocabularyStruct.auto_load_from_pretrained('gpt2')
tokenizer = vocab.tokenizer
max_length = 500
batch_size = 32
num_workers = 4
prefetch_factor = 2
def load_and_tokenize_dataset():
    """
    Loads and tokenizes the WikiText-2 dataset using the specified tokenizer.

    Args:
        model_name (str): Name of the tokenizer model to use.

    Returns:
        tuple: (tokenized_datasets, tokenizer) where tokenized_datasets is the tokenized WikiText-2 dataset.
    """
    # Load WikiText-2 dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    def tokenize_function(examples):
        texts = examples["text"]
        encodings = tokenizer(
            text=texts,
            truncation=True,
            padding=False,
            max_length=max_length,
            return_attention_mask=False,
        )

        return encodings


    # Preprocess dataset. For some reason a significant number
    # of these are empty
    # Tokenize dataset
    dataset = dataset.filter(lambda x: len(x['text']) > 100)
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    return tokenized_datasets, tokenizer


def data_collator(batch: List[Dict[str, List[int]]])->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    """
    Turns the pretraining targets into a batch.
    """
    batch = [item["input_ids"] for item in batch]
    batch = [[tokenizer.bos_token_id] + item + [tokenizer.eos_token_id] for item in batch]
    batch = {"input_ids" : item for item in batch}
    encodings = tokenizer.pad(batch, padding=True, return_tensors='pt', return_attention_mask=True)

    input_ids= encodings['input_ids']
    batch_mask = encodings['attention_mask']

    inputs = input_ids[..., :-1]
    targets = input_ids[..., 1:]
    batch_mask = batch_mask[..., :-1]
    return inputs, targets, batch_mask

def create_data_loaders(tokenized_datasets):
    """
    Creates DataLoaders for training and validation datasets with length-based batching and optimizations.

    Args:
        tokenized_datasets (DatasetDict): The tokenized WikiText-2 dataset.
        tokenizer (AutoTokenizer): Tokenizer for padding and masking.
        batch_size (int): Batch size for training and validation.
        num_workers (int): Number of workers for data loading.
        prefetch_factor (int): Number of batches to prefetch.

    Returns:
        tuple: (train_loader, val_loader) DataLoaders for training and validation.
    """

    # Train DataLoader
    train_loader = DataLoader(
        tokenized_datasets["train"],
        batch_size=batch_size,
        sampler=RandomSampler(tokenized_datasets["train"]),
        collate_fn=data_collator,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
    )

    # Validation DataLoader
    val_loader = DataLoader(
        tokenized_datasets["validation"],
        batch_size=batch_size,
        sampler=SequentialSampler(tokenized_datasets["validation"]),
        collate_fn=data_collator,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
    )

    return train_loader, val_loader


def prepare_wikitext2_dataloaders():
    """
    Prepares the WikiText-2 dataset DataLoaders for training and validation.

    Args:
        model_name (str): Name of the tokenizer model to use.
        batch_size (int): Batch size for training and validation.
        num_workers (int): Number of workers for data loading.
        prefetch_factor (int): Number of batches to prefetch.

    Returns:
        tuple: (train_loader, val_loader) for the WikiText-2 dataset.
    """
    # Load and preprocess the dataset
    tokenized_datasets, tokenizer = load_and_tokenize_dataset()

    # Create DataLoaders
    train_loader, val_loader = create_data_loaders(tokenized_datasets)

    return train_loader, val_loader

if __name__ == '__main__':
    # Usage example
    train_loader, val_loader = prepare_wikitext2_dataloaders()
    for item in train_loader:
        print(item)
        break