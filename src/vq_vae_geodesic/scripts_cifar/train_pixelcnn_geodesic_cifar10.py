import torch
import wandb
from vq_vae_geodesic.config import checkpoint_dir, latents_dir
from vq_vae_geodesic.hyperparameters import get_cifar10_config
from vq_vae_geodesic.utils import set_seed
from vq_vae_geodesic.data.loaders import get_codes_loaders
from vq_vae_geodesic.models.modules.pixelCNN import build_pixelcnn_from_config
from vq_vae_geodesic.training.train import fit_pixelcnn

RESUME = False  # Set to True to resume from checkpoint

"""Train PixelCNN autoregressive prior on VAE + Geodesic quantized CIFAR-10 codes."""
def launch_train_pixelcnn(resume=False):
    config = get_cifar10_config()
    set_seed(config.seed)
    
    device = torch.device("cuda:0" if config.use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load assigned codes (discrete codes indices per image)
    codes_path = latents_dir('cifar10') / "assigned_codes_cifar10.pt"
    print(f"Loading assigned codes from {codes_path}")

    # Get data loaders (of codes indices)
    grid_shape = (config.quant_params.grid_h, config.quant_params.grid_w)
    train_loader, val_loader, _ = get_codes_loaders(
        pt_path=codes_path,
        batch_size=config.pixelcnn_params.batch_size,
        num_workers=config.data_params.num_workers,
        grid_shape=grid_shape
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Build model
    model = build_pixelcnn_from_config(config, for_vqvae=False)
    model = model.to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=config.pixelcnn_params.lr
    )
    
    # Resume logic
    start_epoch = 1
    train_loss_history = []
    val_loss_history = []
    checkpoint_path = checkpoint_dir('cifar10') / "pixelcnn_geodesic_cifar10.pt"
    
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
        name="pixelcnn-geodesic-cifar10",
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
        checkpoint_path=checkpoint_path,
        train_loss_history=train_loss_history,
        val_loss_history=val_loss_history,
        save_checkpoint_every=1
    )

    wandb.finish()

    print(f"\nFinal model saved in {checkpoint_dir('cifar10')}")
    print(f"Final Training Loss: {train_loss_avg[-1]:.4f}")
    print(f"Final Validation Loss: {val_loss_avg[-1]:.4f}")


if __name__ == "__main__":
    launch_train_pixelcnn(resume=RESUME)
