"""
Data loaders for MNIST and CIFAR-10 datasets, and discrete latent codes.
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10


def get_cifar_loaders(batch_size=64, root='data/raw', num_workers=0, shuffle_train_set=True):
    """Get CIFAR-10 data loaders (train, val, test)."""
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = CIFAR10(root=root, train=True, transform=transform, download=True)
    # split train into train and validation WITH FIXED SEED FOR REPRODUCIBILITY
    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(train_ds, [45000, 5000], generator=generator)
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
    # split train into train and validation WITH FIXED SEED FOR REPRODUCIBILITY
    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(train_ds, [55000, 5000], generator=generator)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train_set,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader


class CodesDataset(Dataset):
    """
    Dataset of discrete latent codes for training PixelCNN autoregressive prior.
    
    Loads quantized latent codes from .pt files saved after VAE quantization.
    Codes should be in grid format (N, H, W) where each value is an integer
    index into the codebook [0, K-1].
    
    Args:
        pt_path: Path to .pt file containing codes
        split: 'train' or 'val' to load specific split if available
        grid_shape: Optional (H, W) to reshape flat codes, if needed
    """
    
    def __init__(self, pt_path, split='train', grid_shape=None):
        data = torch.load(pt_path)

        # Try different keys in order of preference
        # First check for pre-split train/val codes
        if f'{split}_codes' in data:
            self.codes = data[f'{split}_codes']
        # Then try codes_per_image (flat codes that need reshaping)
        elif 'codes_per_image' in data:
            arr = data['codes_per_image']
            if grid_shape is None:
                # Try to infer from metadata or n_chunks
                if 'n_chunks' in data:
                    n_chunks = int(data['n_chunks'])
                    # Default: assume 2x4 grid for 8 chunks
                    H = 2
                    W = n_chunks // H
                else:
                    H = int(data.get('grid_h', 2))
                    W = int(data.get('grid_w', 4))
                grid_shape = (H, W)
            self.codes = arr.reshape(-1, *grid_shape)
        # Finally try codes_grid (but check it's not None/object)
        elif 'codes_grid' in data and data['codes_grid'].dtype != object:
            self.codes = data['codes_grid']
        else:
            raise RuntimeError(
                f"Could not find valid codes in {pt_path}. "
                f"Expected keys: '{split}_codes', 'codes_per_image', or 'codes_grid'. "
                f"Available keys: {list(data.keys())}"
            )
        
        self.codes = self.codes.cpu().long()
        # Reshape to (N, H, W) if needed
        if self.codes.ndim == 2 and grid_shape is not None:
            self.codes = self.codes.view(-1, *grid_shape)
        
    def __len__(self):
        return len(self.codes)
    
    def __getitem__(self, idx):
        """Return codes grid as long tensor (H, W)."""
        return self.codes[idx].long()


def get_codes_loaders(pt_path, batch_size=128, val_split=0.1, 
                      num_workers=0, grid_shape=None):
    """
    Get data loaders for discrete latent codes.
    
    Args:
        pt_path: Path to .pt file with codes
        batch_size: Batch size for training
        val_split: Fraction of data to use for validation (if not pre-split)
        num_workers: Number of data loading workers
        grid_shape: Optional (H, W) to reshape codes
        
    Returns:
        train_loader: DataLoader for training
        val_loader: DataLoader for validation (or None)
    """
    # Try to load with split
    try:
        train_ds = CodesDataset(pt_path, split='train', grid_shape=grid_shape)
        try:
            val_ds = CodesDataset(pt_path, split='val', grid_shape=grid_shape)
        except:
            val_ds = None
    except:
        # No split, create one
        ds = CodesDataset(pt_path, grid_shape=grid_shape)
        if val_split is not None and val_split > 0:
            N = len(ds)
            n_val = int(N * val_split)
            n_train = N - n_val
            train_ds, val_ds = random_split(ds, [n_train, n_val])
        else:
            train_ds = ds
            val_ds = None
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    return train_loader, val_loader
