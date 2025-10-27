import torch
import wandb
from vq_vae_geodesic.config import checkpoint_dir, latents_dir
from vq_vae_geodesic.hyperparameters import get_cifar10_config
from vq_vae_geodesic.utils import set_seed
from vq_vae_geodesic.evaluation.utils import load_model_vqvae_cifar10
from vq_vae_geodesic.data.loaders import get_codes_loaders
from vq_vae_geodesic.models.modules.pixelCNN import build_pixelcnn_from_config
from vq_vae_geodesic.training.train import fit_pixelcnn

RESUME = False  # Set to True to resume from checkpoint

"""Train PixelCNN autoregressive prior on VQ-VAE quantized CIFAR-10 codes."""
def launch_train_pixelcnn_vqvae(resume=False):
    config = get_cifar10_config()
    set_seed(config.seed)

    device = torch.device("cuda:0" if config.use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load VQ-VAE checkpoint
    vqvae = load_model_vqvae_cifar10(config.arch_params, config.vqvae_params, device)
    print("Loaded VQ-VAE model for CIFAR-10.")


    # Use pre-extracted VQ-VAE assigned-like codes saved by extract_vqvae_codes_cifar10
    codes_dir = latents_dir('cifar10')
    assigned_codes_path = codes_dir / 'vqvae_assigned_codes_cifar10.pt'
    if not assigned_codes_path.exists():
        raise FileNotFoundError(f"Expected {assigned_codes_path} (run extract_vqvae_codes_cifar10 first)")

    grid_shape = (config.vqvae_params.grid_h, config.vqvae_params.grid_w)
    train_loader_codes, val_loader_codes, _ = get_codes_loaders(
        pt_path=assigned_codes_path,
        batch_size=config.pixelcnn_params.batch_size,
        num_workers=config.data_params.num_workers,
        grid_shape=grid_shape
    )

    print(f"Train batches: {len(train_loader_codes)}")
    print(f"Val batches: {len(val_loader_codes)}")

    # Build PixelCNN model
    model = build_pixelcnn_from_config(config, for_vqvae=True)
    model = model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=config.pixelcnn_vqvae_params.lr
    )

    # Resume logic
    start_epoch = 1
    train_loss_history = []
    val_loss_history = []
    checkpoint_path = checkpoint_dir('cifar10') / "pixelcnn_vqvae_cifar10.pt"

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
        name="pixelcnn-vqvae-cifar10",
        config=config.to_dict()
    )

    # Train
    train_loss_avg, val_loss_avg = fit_pixelcnn(
        model=model,
        train_loader=train_loader_codes,
        val_loader=val_loader_codes,
        optimizer=optimizer,
        num_epochs=config.pixelcnn_vqvae_params.num_epochs,
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
    launch_train_pixelcnn_vqvae(resume=RESUME)
