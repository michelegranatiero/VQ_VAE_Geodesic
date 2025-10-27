import torch
import numpy as np
import wandb
from vq_vae_geodesic.config import checkpoint_dir
from vq_vae_geodesic.hyperparameters import get_cifar10_config
from vq_vae_geodesic.utils import set_seed
from vq_vae_geodesic.training.losses import vqvae_loss_mse
from vq_vae_geodesic.data.loaders import get_cifar_loaders
from vq_vae_geodesic.models.modules.vqvae import build_vqvae_from_config
from vq_vae_geodesic.training.train import fit_vqvae
from torchvision.datasets import CIFAR10

RESUME = False  # Set to True to resume from checkpoint

"""Train VQ-VAE with end-to-end learned codebook on CIFAR-10."""
def launch_train_vqvae(resume=False):
    config = get_cifar10_config()
    set_seed(config.seed)
    device = torch.device("cuda:0" if config.use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Calculate data variance
    print("Calculating data variance...")
    from vq_vae_geodesic.config import data_dir
    train_data_raw = CIFAR10(root=data_dir() / 'raw', train=True, download=False)
    data_variance = np.var(train_data_raw.data / 255.0)
    print(f"Data variance: {data_variance:.6f}")
    
    # Load data
    train_loader, val_loader, _ = get_cifar_loaders(
        batch_size=config.vqvae_params.batch_size
    )
    
    # Build model
    model = build_vqvae_from_config(config.arch_params, config.vqvae_params, dataset="cifar")
    model = model.to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=config.vqvae_params.lr,
    )

    # Loss function
    loss_fn = vqvae_loss_mse

    # Resume logic
    start_epoch = 1
    train_loss_history = []
    train_recon_history = []
    val_loss_history = []
    val_recon_history = []
    checkpoint_path = checkpoint_dir('cifar10') / "vqvae_cifar10.pt"
    
    if resume and checkpoint_path.exists():
        print(f"Resuming training from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        train_loss_history = checkpoint.get('train_loss_history', [])
        train_recon_history = checkpoint.get('train_recon_history', [])
        val_loss_history = checkpoint.get('val_loss_history', [])
        val_recon_history = checkpoint.get('val_recon_history', [])
        print(f"Resumed from epoch {start_epoch}")
    else:
        print("Starting training from scratch")
    
    wandb.init(
        project="vq_vae_geodesic",
        name="vqvae-cifar10",
        config=config.to_dict()
    )
    
    # Train
    train_loss_avg, val_loss_avg = fit_vqvae(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        num_epochs=config.vqvae_params.num_epochs,
        device=device,
        checkpoint_path=checkpoint_path,
        start_epoch=start_epoch,
        train_loss_history=train_loss_history,
        train_recon_history=train_recon_history,
        val_loss_history=val_loss_history,
        val_recon_history=val_recon_history,
        save_checkpoint_every=config.training_params.save_checkpoint_every
    )
    
    wandb.finish()
    
    print(f"\nModel and training state saved in {checkpoint_dir('cifar10')}")

if __name__ == "__main__":
    launch_train_vqvae(resume=RESUME)
