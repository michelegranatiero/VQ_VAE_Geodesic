"""
Train VQ-VAE with end-to-end learned codebook on MNIST.

This serves as a baseline comparison against the VAE + geodesic quantization approach.
VQ-VAE learns the codebook jointly during training via vector quantization.

Pipeline:
1. Encoder: Image → Latent grid (2x4)
2. Vector Quantizer: Continuous latents → Discrete codes via nearest-neighbor
3. Decoder: Quantized latents → Reconstructed image

Loss:
- Reconstruction loss (MSE)
- VQ loss (codebook + commitment)

Usage:
    python -m vq_vae_geodesic.scripts.train_vqvae_mnist
"""
import torch
import wandb

from vq_vae_geodesic.config import checkpoint_dir
from vq_vae_geodesic.hyperparameters import get_mnist_config
from vq_vae_geodesic.utils import set_seed
from vq_vae_geodesic.data.loaders import get_MNIST_loaders
from vq_vae_geodesic.models.modules.vqvae import build_vqvae_from_config
from vq_vae_geodesic.training.train import fit_vqvae

RESUME = False  # Set to True to resume from checkpoint


def launch_train_vqvae(resume=False):
    """Train VQ-VAE with end-to-end learned codebook on MNIST."""
    config = get_mnist_config()
    set_seed(config.seed)
    
    device = torch.device("cuda:0" if config.use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get data loaders
    train_loader, val_loader, _ = get_MNIST_loaders(
        batch_size=config.vqvae_params.batch_size,
        shuffle_train_set=True
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Build model
    model = build_vqvae_from_config(config.arch_params, config.vqvae_params)
    model = model.to(device)
    
    # Count parameters
    # total_params = sum(p.numel() for p in model.parameters())
    # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"Total parameters: {total_params:,}")
    # print(f"Trainable parameters: {trainable_params:,}")
    
    # Optimizer
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=config.vqvae_params.lr,
        weight_decay=config.vqvae_params.weight_decay
    )
    
    # Resume logic
    start_epoch = 1
    train_loss_history = []
    val_loss_history = []
    checkpoint_path = checkpoint_dir() / "vqvae_mnist.pt"
    
    if resume and checkpoint_path.exists():
        print(f"Resuming training from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        train_loss_history = checkpoint.get('train_loss_history', [])
        val_loss_history = checkpoint.get('val_loss_history', [])
        print(f"Resumed from epoch {start_epoch}")
    else:
        print("Starting training from scratch")
    
    # Initialize wandb
    wandb.init(
        project="vq_vae_geodesic",
        name="vqvae-mnist",
        config=config.to_dict()
    )
    
    # Train
    # print("\nVQ-VAE Configuration:")
    # print(f"Codebook size: {config.vqvae_params.num_embeddings}")
    # print(f"Embedding dim: {config.vqvae_params.embedding_dim}")
    # print(f"Commitment cost: {config.vqvae_params.commitment_cost}")
    # print(f"Latent grid: {config.vqvae_params.grid_h}x{config.vqvae_params.grid_w}")
    # print()
    
    train_loss_avg, val_loss_avg = fit_vqvae(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        num_epochs=config.vqvae_params.num_epochs,
        device=device,
        start_epoch=start_epoch,
        checkpoint_path=str(checkpoint_path),
        train_loss_history=train_loss_history,
        val_loss_history=val_loss_history,
        save_checkpoint_every=10
    )
    
    # Save final model
    final_path = checkpoint_dir() / "vqvae_mnist_final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"\nFinal model saved to {final_path}")
    wandb.save(str(final_path))
    
    # Save best model info
    best_path = checkpoint_dir() / "vqvae_mnist_best.pt"
    if best_path.exists():
        print(f"Best model saved at {best_path}")
        wandb.save(str(best_path))
    
    print(f"\nFinal Training Loss: {train_loss_avg[-1]:.4f}")
    if val_loss_avg:
        print(f"Final Validation Loss: {val_loss_avg[-1]:.4f}")
        print(f"Best Validation Loss: {min(val_loss_avg):.4f}")
    
    wandb.finish()
    
    print("\nNext steps:")
    print("1. Evaluate: python -m vq_vae_geodesic.scripts.evaluate_vqvae_mnist")
    print("2. Train PixelCNN on VQ-VAE codes")
    print("3. Compare with geodesic quantization approach")


if __name__ == "__main__":
    launch_train_vqvae(resume=RESUME)
