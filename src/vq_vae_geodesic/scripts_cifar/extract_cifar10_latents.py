"""
Extract latents from trained CIFAR-10 VAE.
"""
import torch

from vq_vae_geodesic.config import checkpoint_dir, latents_dir
from vq_vae_geodesic.hyperparameters import get_cifar10_config
from vq_vae_geodesic.data.loaders import get_cifar_loaders
from vq_vae_geodesic.evaluation.utils import load_model_vae_cifar10
from vq_vae_geodesic.evaluation.extract_latents import extract_and_save_latents

from vq_vae_geodesic.utils import set_seed


def launch_extraction():
    """Extract latents from trained CIFAR-10 VAE."""
    config = get_cifar10_config()
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    # Load trained VAE model
    print("Loading trained VAE model...")
    model = load_model_vae_cifar10(config.arch_params, device)

    # Get data loaders (NOT shuffled for consistent extraction)
    train_loader, val_loader, test_loader = get_cifar_loaders(
        batch_size=config.data_params.batch_size,
        shuffle_train_set=False
    )

    # Extract and save latents
    save_dir = latents_dir('cifar10')
    save_dir.mkdir(exist_ok=True, parents=True)

    print("\n=== Extracting train latents ===")
    train_path = save_dir / "train_latents_cifar10.pt"
    extract_and_save_latents(model, train_loader, device, train_path)

    print("\n=== Extracting validation latents ===")
    val_path = save_dir / "val_latents_cifar10.pt"
    extract_and_save_latents(model, val_loader, device, val_path)

    print("\n=== Extracting test latents ===")
    test_path = save_dir / "test_latents_cifar10.pt"
    extract_and_save_latents(model, test_loader, device, test_path)

    print(f"\n All latents saved to {save_dir}")
    print("\n Next step: Run geodesic quantization")
    print(" uv run python -m vq_vae_geodesic.scripts_cifar.quantize_cifar10")


if __name__ == "__main__":
    launch_extraction()
