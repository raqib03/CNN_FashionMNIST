import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def load_fashion_mnist(batch_size, validation_split=0.2):
    """
    Load the Fashion-MNIST dataset and return DataLoaders for training, validation and testing.

    Args:
        batch_size (int): Number of samples per batch.
        validation_split (float): Fraction of the training set used for validation.
    
    Returns:
        train_loader, val_loader, test_loader (DataLoader): PyTorch DataLoaders for datasets.
    """

    # Define transformations: Convert to tensor and normalize to [0, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) # Normalize to [-1, 1]
    ])

    # Download and load the datasets
    train_dataset = datasets.FashionMNIST(
        root='../data',
        train=True,
        transform=transform,
        download=True
    )
    test_dataset = datasets.FashionMNIST(
        root="../data",
        train=False,
        transform=transform,
        download=True
    )

    # Split the training dataset into traning and validation sets
    val_size = int(len(train_dataset) * validation_split)
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    # Create Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Print dataset sizes
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    return train_loader, val_loader, test_loader
