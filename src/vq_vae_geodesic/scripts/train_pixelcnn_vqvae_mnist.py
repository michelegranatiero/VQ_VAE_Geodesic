"""
Train PixelCNN autoregressive prior on VQ-VAE discrete codes.

This learns p(z) where z are the discrete latent codes from VQ-VAE.
The PixelCNN models the joint distribution over the latent code grid
using an autoregressive factorization.

This allows sampling new codes from the learned distribution, which
can then be decoded to generate new images.

Usage:
    python -m vq_vae_geodesic.scripts.train_pixelcnn_vqvae_mnist
"""
import torch
import numpy as np
from pathlib import Path

from vq_vae_geodesic.config import checkpoint_dir, latents_dir
from vq_vae_geodesic.hyperparameters import get_mnist_config
from vq_vae_geodesic.utils import set_seed
from vq_vae_geodesic.data.loaders import get_MNIST_loaders
from vq_vae_geodesic.models.modules.vqvae import build_vqvae_from_config
from vq_vae_geodesic.models.modules.pixelCNN import PixelCNN
from vq_vae_geodesic.training.train import fit_pixelcnn

import wandb


def extract_vqvae_codes(vqvae, data_loader, device):
    """
    Extract discrete codes from VQ-VAE for all images in the dataset.
    
    Args:
        vqvae: Trained VQ-VAE model
        data_loader: DataLoader for the dataset
        device: Device to run on
        
    Returns:
        codes: Discrete codes (N, H, W) where H, W is the latent grid shape
    """
    vqvae.eval()
    all_codes = []
    
    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)
            
            # Encode to spatial features
            z_e = vqvae.encoder(x)  # (B, embedding_dim, H, W)
            
            # Quantize to get codes
            _, _, codes = vqvae.vq(z_e)  # codes: (B, H, W)
            
            all_codes.append(codes.cpu().numpy())
    
    all_codes = np.concatenate(all_codes, axis=0)
    return all_codes


def launch_train_pixelcnn_vqvae(
    resume_path=None,
    n_epochs=20,
    learning_rate=1e-3,
    use_wandb=True
):
    """
    Main training function for PixelCNN on VQ-VAE codes.
    
    Args:
        resume_path: Path to checkpoint to resume from
        n_epochs: Number of training epochs
        learning_rate: Learning rate for Adam optimizer
        use_wandb: Whether to log to Weights & Biases
    """
    config = get_mnist_config()
    set_seed(config.seed)
    
    device = torch.device("cuda:0" if config.use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # ============================================================
    # Load VQ-VAE and Extract Codes
    # ============================================================
    print("=" * 60)
    print("Loading VQ-VAE and Extracting Codes")
    print("=" * 60)
    
    # Load trained VQ-VAE
    vqvae_path = checkpoint_dir() / "vqvae_mnist_final.pt"
    if not vqvae_path.exists():
        vqvae_path = checkpoint_dir() / "vqvae_mnist.pt"
    
    if not vqvae_path.exists():
        raise FileNotFoundError(
            f"VQ-VAE checkpoint not found at {vqvae_path}. "
            "Train VQ-VAE first with: python -m vq_vae_geodesic.scripts.train_vqvae_mnist"
        )
    
    print(f"\nLoading VQ-VAE from {vqvae_path}")
    vqvae = build_vqvae_from_config(config.arch_params, config.vqvae_params)
    vqvae.load_state_dict(torch.load(vqvae_path, map_location=device))
    vqvae.to(device)
    vqvae.eval()
    
    # Get data loaders
    train_loader, val_loader, _ = get_MNIST_loaders(
        batch_size=config.training_params.batch_size,
        root='data/raw'
    )
    
    # Extract codes from VQ-VAE
    print("\nExtracting discrete codes from training set...")
    train_codes = extract_vqvae_codes(vqvae, train_loader, device)
    print(f"Train codes shape: {train_codes.shape}")
    
    print("\nExtracting discrete codes from validation set...")
    val_codes = extract_vqvae_codes(vqvae, val_loader, device)
    print(f"Val codes shape: {val_codes.shape}")
    
    # Save codes for later use
    codes_dir = latents_dir()
    codes_dir.mkdir(parents=True, exist_ok=True)
    
    train_codes_path = codes_dir / "vqvae_train_codes.pt"
    val_codes_path = codes_dir / "vqvae_val_codes.pt"

    torch.save({'codes': torch.from_numpy(train_codes)}, train_codes_path)
    torch.save({'codes': torch.from_numpy(val_codes)}, val_codes_path)

    print(f"\nSaved train codes to {train_codes_path}")
    print(f"Saved val codes to {val_codes_path}")
    
    # ============================================================
    # Initialize PixelCNN
    # ============================================================
    print("\n" + "=" * 60)
    print("Initializing PixelCNN")
    print("=" * 60)
    
    # Grid shape from VQ-VAE codes
    grid_h, grid_w = train_codes.shape[1], train_codes.shape[2]
    num_tokens = vqvae.vq.num_embeddings
    
    print(f"\nPixelCNN Configuration:")
    print(f"  Grid shape: {grid_h}x{grid_w}")
    print(f"  Vocabulary size (K): {num_tokens}")
    print(f"  Embedding dim: {config.pixelcnn_params.embed_dim}")
    print(f"  Hidden channels: {config.pixelcnn_params.hidden_channels}")
    print(f"  Number of layers: {config.pixelcnn_params.n_layers}")
    
    model = PixelCNN(
        num_tokens=num_tokens,
        embed_dim=config.pixelcnn_params.embed_dim,
        hidden_channels=config.pixelcnn_params.hidden_channels,
        n_layers=config.pixelcnn_params.n_layers
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    # ============================================================
    # Setup Training
    # ============================================================
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    start_epoch = 1
    if resume_path and Path(resume_path).exists():
        print(f"\nResuming from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")
    
    # Initialize wandb
    if use_wandb:
        wandb.init(
            project="vq_vae_geodesic",
            name="pixelcnn-vqvae-mnist",
            config={
                "model": "PixelCNN",
                "approach": "VQ-VAE prior",
                "dataset": "MNIST",
                "num_tokens": num_tokens,
                "grid_shape": f"{grid_h}x{grid_w}",
                "embed_dim": config.pixelcnn_params.embed_dim,
                "hidden_channels": config.pixelcnn_params.hidden_channels,
                "n_layers": config.pixelcnn_params.n_layers,
                "learning_rate": learning_rate,
                "n_epochs": n_epochs,
                "batch_size": config.training_params.batch_size,
            }
        )
    
    # ============================================================
    # Train PixelCNN
    # ============================================================
    print("\n" + "=" * 60)
    print("Training PixelCNN on VQ-VAE Codes")
    print("=" * 60)
    
    # Convert codes to torch tensors and create datasets
    train_codes_t = torch.from_numpy(train_codes).long()
    val_codes_t = torch.from_numpy(val_codes).long()
    
    # Create simple dataset that returns tensors directly (not tuples)
    class SimpleCodesDataset(torch.utils.data.Dataset):
        def __init__(self, codes):
            self.codes = codes
        def __len__(self):
            return len(self.codes)
        def __getitem__(self, idx):
            return self.codes[idx]
    
    train_dataset = SimpleCodesDataset(train_codes_t)
    val_dataset = SimpleCodesDataset(val_codes_t)
    
    train_loader_codes = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.training_params.batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_loader_codes = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.training_params.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    print(f"\nTrain batches: {len(train_loader_codes)}")
    print(f"Val batches: {len(val_loader_codes)}")
    print(f"\nTraining from epoch {start_epoch} to {n_epochs}...")
    
    # Train
    checkpoint_path = checkpoint_dir() / "pixelcnn_vqvae_mnist.pt"
    
    fit_pixelcnn(
        model=model,
        train_loader=train_loader_codes,
        val_loader=val_loader_codes,
        optimizer=optimizer,
        device=device,
        num_epochs=n_epochs,
        start_epoch=start_epoch,
        checkpoint_path=checkpoint_path
    )
    
    # Save final model
    final_path = checkpoint_dir() / "pixelcnn_vqvae_mnist_final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"\nSaved final model to {final_path}")
    
    if use_wandb:
        wandb.finish()
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nCheckpoints saved to:")
    print(f"  - {checkpoint_path} (best validation)")
    print(f"  - {final_path} (final epoch)")


if __name__ == "__main__":
    launch_train_pixelcnn_vqvae(
        resume_path=None,
        n_epochs=20,
        learning_rate=1e-3,
        use_wandb=True
    )
