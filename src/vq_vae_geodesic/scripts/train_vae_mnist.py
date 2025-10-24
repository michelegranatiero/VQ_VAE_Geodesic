import torch
import wandb
from vq_vae_geodesic.hyperparameters import get_mnist_config
from vq_vae_geodesic.utils import set_seed
from functools import partial
from vq_vae_geodesic.data.loaders import get_MNIST_loaders
from vq_vae_geodesic.models.modules.vae import build_vae_from_config
from vq_vae_geodesic.training.losses import vae_loss_bce
from vq_vae_geodesic.config import checkpoint_dir
from vq_vae_geodesic.training.train import fit_vae

RESUME = False  # Set to True to resume from checkpoint

"""Train VAE on MNIST dataset."""
def launch_train(resume=False):
    config = get_mnist_config()
    set_seed(config.seed)
    device = torch.device("cuda:0" if config.use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    train_loader, val_loader, _ = get_MNIST_loaders(
        batch_size=config.data_params.batch_size,
        num_workers=config.data_params.num_workers
    )

    # Build model
    model = build_vae_from_config(config.arch_params)
    model = model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=config.training_params.lr,
        weight_decay=config.training_params.weight_decay
    )

    # Loss function con beta fissato tramite functools.partial
    loss_fn = partial(vae_loss_bce, beta=config.training_params.variational_beta)

    # Resume logic
    start_epoch = 1
    train_loss_history = []
    train_recon_history = []
    val_loss_history = []
    val_recon_history = []
    checkpoint_path = checkpoint_dir() / "checkpoint_mnist.pt"

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
        name="vae-mnist",
        config=config.to_dict()
    )

    # Train
    train_loss_avg, val_loss_avg = fit_vae(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        num_epochs=config.training_params.num_epochs,
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

    print(f"\nModel and training state saved in {checkpoint_dir()}")
    print("\nNext step: Extract latents")
    print("uv run -m src.vq_vae_geodesic.scripts.extract_mnist_latents")


if __name__ == "__main__":
    launch_train(resume=RESUME)
