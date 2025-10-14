"""
Perform geodesic quantization on extracted MNIST latents.
"""

import torch

from vq_vae_geodesic.config import latents_dir
from vq_vae_geodesic.hyperparameters import get_mnist_config
from vq_vae_geodesic.models.quantization.geodesic import GeodesicQuantizer
from vq_vae_geodesic.utils import set_seed

def launch_quantization():
    """Perform geodesic quantization on extracted MNIST latents."""
    config = get_mnist_config()
    set_seed(config.seed)
    quant_params = config.quant_params

    # -------------- LATENTS LOADING --------------
    # Load train latents (mu and logvar)
    train_latents_path = latents_dir() / "train_latents.pt"
    if not train_latents_path.exists():
        raise FileNotFoundError(
            f"Latents not found at {train_latents_path}\n"
            "Run extraction first: python -m vq_vae_geodesic.scripts.extract_mnist_latents"
        )

    train_data = torch.load(train_latents_path, map_location='cpu')
    train_latents = train_data['mu']  # torch.Tensor
    train_variances = train_data['logvar']
    print(f"Loaded train latents: mu={train_latents.shape}, logvar={train_variances.shape}")

    # Load val latents (mu and logvar)
    val_latents_path = latents_dir() / "val_latents.pt"
    if not val_latents_path.exists():
        raise FileNotFoundError(
            f"Latents not found at {val_latents_path}\n"
            "Run extraction first: python -m vq_vae_geodesic.scripts.extract_mnist_latents"
        )

    val_data = torch.load(val_latents_path, map_location='cpu')
    val_latents = val_data['mu']
    val_variances = val_data['logvar']
    print(f"Loaded val latents: mu={val_latents.shape}, logvar={val_variances.shape}")

    # Load test latents (mu and logvar)
    test_latents_path = latents_dir() / "test_latents.pt"
    if not test_latents_path.exists():
        raise FileNotFoundError(
            f"Latents not found at {test_latents_path}\n"
            "Run extraction first: python -m vq_vae_geodesic.scripts.extract_mnist_latents"
        )

    test_data = torch.load(test_latents_path, map_location='cpu')
    test_latents = test_data['mu']
    test_variances = test_data['logvar']
    print(f"Loaded test latents: mu={test_latents.shape}, logvar={test_variances.shape}")

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
    quantizer.fit(train_latents, train_variances)
    print(f"Codebook shape: {quantizer.codebook_chunks.shape}")

    # -------------- CODE ASSIGNMENT --------------
    # Assign train latents to codebook (get assigned codes)
    print("Assigning train latents to codebook...")
    train_codes = quantizer.assign(train_latents, train_variances)  # (N_train, n_chunks)
    print(f"Train codes shape: {train_codes.shape}")

    # Assign val latents to codebook
    print("Assigning val latents to codebook...")
    val_codes = quantizer.assign(val_latents, val_variances)  # (N_val, n_chunks)
    print(f"Val codes shape: {val_codes.shape}")

    # Assign test latents to codebook
    print("Assigning test latents to codebook...")
    test_codes = quantizer.assign(test_latents, test_variances)  # (N_test, n_chunks)
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
    print("\nNext step: Evaluate reconstructions")
    print("python -m vq_vae_geodesic.scripts.reconstruct_mnist")


if __name__ == "__main__":
    launch_quantization()
