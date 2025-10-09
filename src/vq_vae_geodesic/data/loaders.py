"""
Data loaders for MNIST and CIFAR-10 datasets.
"""
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10


def get_cifar_loaders(batch_size=64, root='data/raw', num_workers=0, shuffle_train_set=True):
    """Get CIFAR-10 data loaders (train, val, test)."""
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = CIFAR10(root=root, train=True, transform=transform, download=True)
    # split train into train and validation
    train_ds, val_ds = random_split(train_ds, [45000, 5000])
    test_ds = CIFAR10(root=root, train=False, transform=transform, download=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train_set,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader


def get_MNIST_loaders(batch_size=64, root='data/raw', num_workers=0, shuffle_train_set=True):
    """Get MNIST data loaders (train, val, test)."""
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = MNIST(root=root, train=True, transform=transform, download=True)
    test_ds = MNIST(root=root, train=False, transform=transform, download=True)
    # split train into train and validation
    train_ds, val_ds = random_split(train_ds, [55000, 5000])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train_set,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader
