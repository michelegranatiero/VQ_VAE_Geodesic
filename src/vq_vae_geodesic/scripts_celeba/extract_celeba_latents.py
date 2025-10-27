"""
Extract latents from trained CelebA VAE.
"""
import torch

from vq_vae_geodesic.config import checkpoint_dir, latents_dir
from vq_vae_geodesic.hyperparameters import get_celeba_config
from vq_vae_geodesic.data.loaders import get_celeba_loaders
from vq_vae_geodesic.evaluation.extract_latents import extract_and_save_latents
from vq_vae_geodesic.utils import set_seed

IMG_SIZE = 32 # Must match training


def launch_extraction(img_size=64):
    """Extract latents from trained CelebA VAE."""
    config = get_celeba_config(img_size=img_size)
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"Image size: {img_size}×{img_size}")

    # Load trained VAE model
    print("Loading trained VAE model...")
    checkpoint_path = checkpoint_dir('celeba') / "vae_celeba_best.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    from vq_vae_geodesic.models.modules.vae import build_vae_from_config
    model = build_vae_from_config(config.arch_params, dataset="celeba")  # Uses CelebA architecture (RGB)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Get CelebA data loaders (NOT shuffled for consistent extraction)
    train_loader, val_loader, test_loader = get_celeba_loaders(
        batch_size=config.data_params.batch_size,
        shuffle_train_set=False,
        num_workers=config.data_params.num_workers,
        img_size=img_size
    )

    # Extract and save latents
    save_dir = latents_dir('celeba')
    save_dir.mkdir(exist_ok=True, parents=True)

    print("\n=== Extracting train latents ===")
    train_path = save_dir / "train_latents_celeba.pt"
    extract_and_save_latents(model, train_loader, device, train_path)

    print("\n=== Extracting validation latents ===")
    val_path = save_dir / "val_latents_celeba.pt"
    extract_and_save_latents(model, val_loader, device, val_path)

    print("\n=== Extracting test latents ===")
    test_path = save_dir / "test_latents_celeba.pt"
    extract_and_save_latents(model, test_loader, device, test_path)

    print(f"\n All latents saved to {save_dir}")
    print("\n Next step: Run geodesic quantization")
    print(" uv run python -m vq_vae_geodesic.scripts_celeba.quantize_celeba")

if __name__ == "__main__":
    launch_extraction(img_size=IMG_SIZE)
