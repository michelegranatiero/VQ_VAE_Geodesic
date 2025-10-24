"""
Extract latents from trained MNIST VAE.
"""
import torch

from vq_vae_geodesic.config import checkpoint_dir, latents_dir
from vq_vae_geodesic.hyperparameters import get_mnist_config
from vq_vae_geodesic.data.loaders import get_MNIST_loaders
from vq_vae_geodesic.evaluation.utils import load_model_vae_mnist
from vq_vae_geodesic.evaluation.extract_latents import extract_and_save_latents

from vq_vae_geodesic.utils import set_seed


def launch_extraction():
    """Extract latents from trained MNIST VAE."""
    config = get_mnist_config()
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load trained model
    model = load_model_vae_mnist(config.arch_params, device)

    # Get data loaders
    train_loader, val_loader, test_loader = get_MNIST_loaders(
        batch_size=config.training_params.batch_size, shuffle_train_set=False
    )

    # Extract and save latents
    save_dir = latents_dir()
    save_dir.mkdir(exist_ok=True, parents=True)

    print("Extracting train latents...")
    train_path = save_dir / "train_latents.pt"
    extract_and_save_latents(model, train_loader, device, train_path)

    print("Extracting validation latents...")
    val_path = save_dir / "val_latents.pt"
    extract_and_save_latents(model, val_loader, device, val_path)

    print("Extracting test latents...")
    test_path = save_dir / "test_latents.pt"
    extract_and_save_latents(model, test_loader, device, test_path)

    print(f"\nLatents saved to {save_dir}")
    print("\nNext step: Run geodesic quantization")
    print("uv run -m src.vq_vae_geodesic.scripts.quantize_mnist")


if __name__ == "__main__":
    launch_extraction()
