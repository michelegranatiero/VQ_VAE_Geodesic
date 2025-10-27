import torch
import numpy as np
import wandb
from vq_vae_geodesic.config import checkpoint_dir
from vq_vae_geodesic.hyperparameters import get_celeba_config
from vq_vae_geodesic.utils import set_seed
from vq_vae_geodesic.training.losses import vqvae_loss_mse
from vq_vae_geodesic.data.loaders import get_celeba_loaders
from vq_vae_geodesic.models.modules.vqvae import build_vqvae_from_config
from vq_vae_geodesic.training.train import fit_vqvae

RESUME = False  # Set to True to resume from checkpoint
IMG_SIZE = 32   # 32 or 64

"""Train VQ-VAE with end-to-end learned codebook on CelebA."""
def launch_train_vqvae(resume=False, img_size=32):
    config = get_celeba_config(img_size=img_size)
    set_seed(config.seed)
    device = torch.device("cuda:0" if config.use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Image size: {img_size}×{img_size}")
    
    # Calculate data variance (using a batch from CelebA)
    print("Calculating data variance...")
    from vq_vae_geodesic.data.loaders import get_celeba_loaders
    temp_loader, _, _ = get_celeba_loaders(batch_size=1000, num_workers=0, img_size=img_size)
    sample_batch, _ = next(iter(temp_loader))
    data_variance = np.var(sample_batch.numpy())
    print(f"Data variance: {data_variance:.6f}")
    
    # Load CelebA data
    train_loader, val_loader, _ = get_celeba_loaders(
        batch_size=config.vqvae_params.batch_size,
        num_workers=config.data_params.num_workers,
        img_size=img_size
    )
    
    # Build model
    model = build_vqvae_from_config(config.arch_params, config.vqvae_params, dataset="celeba")
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
    checkpoint_path = checkpoint_dir('celeba') / "vqvae_celeba.pt"
    
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
        name="vqvae-celeba",
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
    
    print(f"\nModel and training state saved in {checkpoint_dir('celeba')}")

if __name__ == "__main__":
    launch_train_vqvae(resume=RESUME, img_size=IMG_SIZE)
