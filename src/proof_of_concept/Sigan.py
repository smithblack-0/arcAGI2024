"""

"""


import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset


def download_and_preprocess_data(train_size=1000, test_size=10000):
    # Define a transform to normalize the data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalizing to [-1, 1]
    ])

    # Download and transform the CIFAR-10 dataset
    full_train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    full_test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Combine the datasets
    full_dataset = torch.utils.data.ConcatDataset([full_train_set, full_test_set])

    # Shuffle the dataset
    indices = torch.randperm(len(full_dataset))

    # Select the training and test subsets
    train_indices = indices[:train_size]
    test_indices = indices[train_size:train_size + test_size]

    train_subset = Subset(full_dataset, train_indices)
    test_subset = Subset(full_dataset, test_indices)

    # Create data loaders
    train_loader = DataLoader(train_subset, batch_size=train_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=test_size, shuffle=False)

    return train_loader, test_loader


# Example usage:
train_loader, test_loader = download_and_preprocess_data()
print(f"Training set size: {len(train_loader.dataset)}")
print(f"Testing set size: {len(test_loader.dataset)}")
