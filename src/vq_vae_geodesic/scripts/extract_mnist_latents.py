"""
Extract latents from trained MNIST VAE.
"""
import torch

from vq_vae_geodesic.config import checkpoint_dir, latents_dir
from vq_vae_geodesic.hyperparameters import get_mnist_config
from vq_vae_geodesic.data.loaders import get_MNIST_loaders
from vq_vae_geodesic.models.modules.vae import build_vae_from_config
from vq_vae_geodesic.evaluation import extract_and_save_latents

from vq_vae_geodesic.utils import set_seed


def launch_extraction():
    """Extract latents from trained MNIST VAE."""
    config = get_mnist_config()
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load trained model
    checkpoint_path = checkpoint_dir() / "main_checkpoint_mnist.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    model = build_vae_from_config(config.arch_params)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Get data loaders
    train_loader, val_loader, _ = get_MNIST_loaders(
        batch_size=config.training_params.batch_size, shuffle_train_set=False
    )

    # Extract and save latents
    save_dir = latents_dir()

    print("Extracting train latents...")
    extract_and_save_latents(model, train_loader, device, save_dir, "train")

    print("Extracting validation latents...")
    extract_and_save_latents(model, val_loader, device, save_dir, "val")

    print(f"\nLatents saved to {save_dir}")
    print("\nNext step: Run geodesic quantization")
    print("python -m vq_vae_geodesic.scripts.quantize_mnist")


if __name__ == "__main__":
    launch_extraction()
