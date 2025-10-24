"""
Perform geodesic quantization on extracted MNIST latents.
"""

import torch

from vq_vae_geodesic.config import latents_dir
from vq_vae_geodesic.hyperparameters import get_mnist_config
from vq_vae_geodesic.models.quantization.geodesic import GeodesicQuantizer
from vq_vae_geodesic.utils import set_seed
from vq_vae_geodesic.evaluation.utils import load_latents_mnist

def launch_quantization():
    """Perform geodesic quantization on extracted MNIST latents."""
    config = get_mnist_config()
    set_seed(config.seed)
    quant_params = config.quant_params

    # -------------- LATENTS LOADING --------------
    train_latents_path = latents_dir() / "train_latents.pt"
    val_latents_path = latents_dir() / "val_latents.pt"
    test_latents_path = latents_dir() / "test_latents.pt"

    train_latents, train_logvars = load_latents_mnist(train_latents_path)
    print(f"Loaded train latents: mu={train_latents.shape}, logvar={train_logvars.shape}")

    val_latents, val_logvars = load_latents_mnist(val_latents_path)
    print(f"Loaded val latents: mu={val_latents.shape}, logvar={val_logvars.shape}")

    test_latents, test_logvars = load_latents_mnist(test_latents_path)
    print(f"Loaded test latents: mu={test_latents.shape}, logvar={test_logvars.shape}")

    # -------------- QUANTIZATION --------------
    # Create quantizer
    print("\nBuilding geodesic quantizer...")
    quantizer = GeodesicQuantizer(
        n_codewords=quant_params.n_codewords,
        n_chunks=quant_params.n_chunks,
        random_state=quant_params.random_state,
    )

    # Fit codebook on TRAIN latents (generate codebook)
    print("Fitting codebook with geodesic distances...")
    quantizer.fit(train_latents, train_logvars)
    print(f"Codebook shape: {quantizer.codebook_chunks.shape}")

    # -------------- CODE ASSIGNMENT --------------
    # Assign train latents to codebook (get assigned codes)
    print("Assigning train latents to codebook...")
    train_codes = quantizer.assign(train_latents, train_logvars)  # (N_train, n_chunks)
    print(f"Train codes shape: {train_codes.shape}")

    # Assign val latents to codebook
    print("Assigning val latents to codebook...")
    val_codes = quantizer.assign(val_latents, val_logvars)  # (N_val, n_chunks)
    print(f"Val codes shape: {val_codes.shape}")

    # Assign test latents to codebook
    print("Assigning test latents to codebook...")
    test_codes = quantizer.assign(test_latents, test_logvars)  # (N_test, n_chunks)
    print(f"Test codes shape: {test_codes.shape}")

    # -------------- SAVING CODEBOOK AND CODES --------------
    # Save codebook
    save_path = latents_dir() / "chunk_codebook.pt"
    quantizer.save(save_path)
    print(f"\nCodebook saved to {save_path}")

    # Save all codes together for convenience
    all_codes_path = latents_dir() / "assigned_codes.pt"
    torch.save({
        'train_codes': train_codes,
        'val_codes': val_codes,
        'test_codes': test_codes
    }, all_codes_path)

    print(f"Codes saved to {latents_dir()}")


if __name__ == "__main__":
    launch_quantization()
