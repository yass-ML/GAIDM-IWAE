import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def get_dataloaders(batch_size=128, data_dir='./data'):
    """
    Returns (train_loader, val_loader, test_loader) for MNIST.

    Preprocessing:
    - Train: Dynamic Binarization (sampling from pixel intensities)
    - Val/Test: Fixed Binarization (rounding at 0.5)

    Split:
    - Train: 50,000 samples
    - Val: 10,000 samples (from the original training set)
    - Test: 10,000 samples (original test set)
    """

    # Transforms
    # Dynamic binarization: interpret pixel value as probability p, sample x ~ Bern(p)
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.bernoulli(x))
    ])

    # Fixed binarization: round to nearest integer (0 or 1)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.round(x))
    ])

    # Download and load datasets
    # We load train dataset twice to apply different transforms for train vs val
    full_train_dynamic = datasets.MNIST(data_dir, train=True, download=True, transform=train_transform)
    full_train_fixed = datasets.MNIST(data_dir, train=True, download=True, transform=test_transform)
    test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=test_transform)

    # Create Split Indices (fixed seed for reproducibility of split)
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(full_train_dynamic), generator=generator)

    train_indices = indices[:50000]
    val_indices = indices[50000:]

    # Create Subsets
    train_dataset = Subset(full_train_dynamic, train_indices)
    val_dataset = Subset(full_train_fixed, val_indices)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Data Loaded: Train {len(train_dataset)}, Val {len(val_dataset)}, Test {len(test_dataset)}")

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Test the loader
    tr, va, te = get_dataloaders()
    x, _ = next(iter(tr))
    print(f"Batch shape: {x.shape}")
    print(f"Min: {x.min()}, Max: {x.max()}") # Should be 0 and 1
