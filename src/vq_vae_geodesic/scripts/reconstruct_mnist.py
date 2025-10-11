"""
Evaluate reconstructions on MNIST using geodesic quantization.
"""
import torch
import numpy as np

from vq_vae_geodesic.config import checkpoint_dir, latents_dir, recons_dir
from vq_vae_geodesic.hyperparameters import get_mnist_config
from vq_vae_geodesic.data.loaders import get_MNIST_loaders
from vq_vae_geodesic.models.modules.vae import build_vae_from_config
from vq_vae_geodesic.models.quantization.geodesic import GeodesicQuantizer
from vq_vae_geodesic.evaluation import recon_from_mu, reconstruct_from_chunk_codebook
from vq_vae_geodesic.utils import set_seed


def launch_reconstruction():
    """Evaluate reconstructions on MNIST using geodesic quantization."""
    config = get_mnist_config()
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    checkpoint_path = checkpoint_dir() / "main_checkpoint_mnist.pth"
    model = build_vae_from_config(config.arch_params)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Load codebook
    codebook_path = latents_dir() / "chunk_codebook.npz"
    if not codebook_path.exists():
        raise FileNotFoundError(
            f"Codebook not found at {codebook_path}\n"
            "Run quantization first: python -m vq_vae_geodesic.scripts.quantize_mnist"
        )

    quantizer = GeodesicQuantizer.load(codebook_path)
    print(f"Loaded codebook: {quantizer.codebook_chunks.shape}")

    # Load train codes
    codebook_data = np.load(codebook_path)
    train_codes = codebook_data['codes_per_image']
    print(f"Loaded train codes: {train_codes.shape}")

    # Load val codes
    val_codes_path = latents_dir() / "val_codes.npz"
    if not val_codes_path.exists():
        raise FileNotFoundError(
            f"Val codes not found at {val_codes_path}\n"
            "Run quantization first: python -m vq_vae_geodesic.scripts.quantize_mnist"
        )
    val_codes_data = np.load(val_codes_path)
    val_codes = val_codes_data['codes_per_image']
    print(f"Loaded val codes: {val_codes.shape}")

    # Get data loaders
    train_loader, val_loader, _ = get_MNIST_loaders(
        batch_size=config.training_params.batch_size, shuffle_train_set=False
    )

    # Reconstruction output directories
    recons_train_dir = recons_dir() / "train"
    recons_val_dir = recons_dir() / "val"
    recons_train_dir.mkdir(parents=True, exist_ok=True)
    recons_val_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate on train set
    print("\n=== Train Set ===")
    print("Baseline reconstruction (from mu)...")
    mse_mu_train = recon_from_mu(
        model, train_loader, device, str(recons_train_dir)
    )

    print("\nReconstruction with geodesic quantization...")
    mse_quantized_train = reconstruct_from_chunk_codebook(
        model, train_loader, quantizer.codebook_chunks, train_codes,
        device, str(recons_train_dir)
    )

    loss_increase_train = ((mse_quantized_train - mse_mu_train) / mse_mu_train) * 100
    print(f"\nTrain MSE (baseline): {mse_mu_train:.6f}")
    print(f"Train MSE (quantized): {mse_quantized_train:.6f}")
    print(f"Quantization loss increase: {loss_increase_train:.2f}%")

    # Evaluate on validation set
    print("\n=== Validation Set ===")
    print("Baseline reconstruction (from mu)...")
    mse_mu_val = recon_from_mu(
        model, val_loader, device, str(recons_val_dir)
    )

    print("\nReconstruction with geodesic quantization...")
    mse_quantized_val = reconstruct_from_chunk_codebook(
        model, val_loader, quantizer.codebook_chunks, val_codes,
        device, str(recons_val_dir)
    )

    loss_increase_val = ((mse_quantized_val - mse_mu_val) / mse_mu_val) * 100
    print(f"\nValidation MSE (baseline): {mse_mu_val:.6f}")
    print(f"Validation MSE (quantized): {mse_quantized_val:.6f}")
    print(f"Quantization loss increase: {loss_increase_val:.2f}%")

    print(f"\n Reconstructions saved to {recons_dir()}")


if __name__ == "__main__":
    launch_reconstruction()
