"""
Data loaders
"""
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10, CelebA


def get_celeba_loaders(batch_size=128, num_workers=0, data_dir='data/raw', shuffle_train_set=True, img_size=32):
    transform = transforms.Compose([
        transforms.CenterCrop(140),  # Crop to centered face region
        transforms.Resize(img_size),  # Resize to target size (32 or 64)
        transforms.ToTensor()  # [0, 1] range
    ])
    
    root = data_dir
    
    # CelebA has official train/val/test splits
    train_ds = CelebA(root=root, split='train', transform=transform, download=True)
    val_ds = CelebA(root=root, split='valid', transform=transform, download=True)
    test_ds = CelebA(root=root, split='test', transform=transform, download=True)
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=shuffle_train_set,
        num_workers=num_workers, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers, 
        pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, 
        batch_size=batch_size * 2, 
        shuffle=False,
        num_workers=num_workers, 
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def get_cifar_loaders(batch_size=100, num_workers=2, data_dir='data/raw', shuffle_train_set=True):
    """
    Returns train, val, test loaders for CIFAR-10 (32x32x3 RGB images).
    Images are in [0, 1] range.
    """
    transform = transforms.Compose([
        transforms.ToTensor()  # [0, 1]
    ])
    root = data_dir
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
    Dataset of discrete latent codes (indices) for training PixelCNN autoregressive prior.

    Loads quantized latent codes indices from .pt files saved after VAE quantization.
    Codes should be in grid format (N, H, W) where each value is an integer
    index into the codebook [0, K-1].
    """

    def __init__(self, pt_path, split='train', grid_shape=None):
        data = torch.load(pt_path)
        # Only support current workflow: codes saved as 'train_codes', 'val_codes', etc.
        key = f'{split}_codes'
        if key not in data:
            raise RuntimeError(f"Could not find '{key}' in {pt_path}. Available keys: {list(data.keys())}")
        codes = data[key]
        codes = codes.cpu().long()
        # Reshape to (N, H, W) if needed
        if codes.ndim == 2 and grid_shape is not None:
            codes = codes.view(-1, *grid_shape)
        self.codes = codes

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, idx):
        """Return codes (indices) grid as long tensor (H, W)."""
        return self.codes[idx]

def get_codes_loaders(pt_path, batch_size=128, num_workers=0, grid_shape=None):
    """
    Get data loaders for discrete latent codes (indices).
    """
    train_ds = CodesDataset(pt_path, split='train', grid_shape=grid_shape)
    val_ds = CodesDataset(pt_path, split='val', grid_shape=grid_shape)
    test_ds = CodesDataset(pt_path, split='test', grid_shape=grid_shape)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader
