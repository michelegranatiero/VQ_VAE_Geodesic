"""
Train PixelCNN autoregressive prior on quantized MNIST latent codes.

After training a VAE and quantizing latents, this script trains a PixelCNN
to model the distribution p(z) over discrete latent codes. The learned prior
can then be used for unconditional generation.

Pipeline:
1. Load quantized codes from pt file (output of quantize_mnist.py)
2. Build PixelCNN model
3. Train with cross-entropy loss
4. Save trained model for sampling

Usage:
    python -m vq_vae_geodesic.scripts.train_pixelcnn_mnist
"""
import torch
import wandb

from vq_vae_geodesic.config import checkpoint_dir, latents_dir
from vq_vae_geodesic.hyperparameters import get_mnist_config
from vq_vae_geodesic.utils import set_seed
from vq_vae_geodesic.data.loaders import get_codes_loaders
from vq_vae_geodesic.models.modules.pixelCNN import PixelCNN
from vq_vae_geodesic.training.train import fit_pixelcnn

RESUME = False  # Set to True to resume from checkpoint


def launch_train_pixelcnn(resume=False):
    """Train PixelCNN autoregressive prior on quantized MNIST codes."""
    config = get_mnist_config()
    set_seed(config.seed)
    
    device = torch.device("cuda:0" if config.use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load assigned codes (discrete codes per image)
    codes_path = latents_dir() / "assigned_codes.pt"
    if not codes_path.exists():
        raise FileNotFoundError(
            f"Assigned codes not found at {codes_path}\n"
            "Run quantization first: python -m vq_vae_geodesic.scripts.quantize_mnist"
        )

    print(f"Loading assigned codes from {codes_path}")

    # Get data loaders
    grid_shape = (config.quant_params.grid_h, config.quant_params.grid_w)
    train_loader, val_loader = get_codes_loaders(
        pt_path=codes_path,
        batch_size=config.pixelcnn_params.batch_size,
        val_split=config.pixelcnn_params.val_split,
        num_workers=config.data_params.num_workers,
        grid_shape=grid_shape
    )
    
    print(f"Train batches: {len(train_loader)}")
    if val_loader:
        print(f"Val batches: {len(val_loader)}")
    
    # Build model
    model = PixelCNN(
        num_tokens=config.quant_params.n_codewords,
        embed_dim=config.pixelcnn_params.embed_dim,
        hidden_channels=config.pixelcnn_params.hidden_channels,
        n_layers=config.pixelcnn_params.n_layers,
        kernel_size=config.pixelcnn_params.kernel_size
    )
    model = model.to(device)
    
    print(f"PixelCNN parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=config.pixelcnn_params.lr
    )
    
    # Resume logic
    start_epoch = 1
    train_loss_history = []
    val_loss_history = []
    checkpoint_path = checkpoint_dir() / "pixelcnn_mnist.pt"
    
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
        name="pixelcnn-geodesic-mnist",
        config=config.to_dict()
    )
    
    # Train
    train_loss_avg, val_loss_avg = fit_pixelcnn(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        num_epochs=config.pixelcnn_params.num_epochs,
        device=device,
        start_epoch=start_epoch,
        checkpoint_path=str(checkpoint_path),
        train_loss_history=train_loss_history,
        val_loss_history=val_loss_history,
        save_checkpoint_every=1
    )
    
    # Save final model
    final_path = checkpoint_dir() / "pixelcnn_mnist_final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"\nFinal model saved to {final_path}")
    wandb.save(str(final_path))
    
    print(f"Final Training Loss: {train_loss_avg[-1]:.4f}")
    if val_loss_avg:
        print(f"Final Validation Loss: {val_loss_avg[-1]:.4f}")
    
    wandb.finish()
    
    print("\nNext steps:")
    print("  1. Sample new codes: python -m vq_vae_geodesic.scripts.sample_pixelcnn_mnist")
    print("  2. Generate images from codes using VAE decoder")


if __name__ == "__main__":
    launch_train_pixelcnn(resume=RESUME)
