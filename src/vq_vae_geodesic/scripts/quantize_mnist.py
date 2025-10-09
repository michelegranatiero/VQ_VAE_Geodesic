"""
Perform geodesic quantization on extracted MNIST latents.
"""
import numpy as np

from vq_vae_geodesic.config import latents_dir
from vq_vae_geodesic.hyperparameters import get_mnist_config
from vq_vae_geodesic.models.quantization.geodesic import GeodesicQuantizer

from vq_vae_geodesic.utils import set_seed

def launch_quantization():
    """Perform geodesic quantization on extracted MNIST latents."""
    config = get_mnist_config()
    set_seed(config.seed)
    quant_params = config.quant_params

    # Load train latents
    train_latents_path = latents_dir() / "train_latents.npz"
    if not train_latents_path.exists():
        raise FileNotFoundError(
            f"Latents not found at {train_latents_path}\n"
            "Run extraction first: python -m vq_vae_geodesic.scripts.extract_mnist_latents"
        )

    train_data = np.load(train_latents_path)
    train_latents = train_data['mu']  # Use 'mu' key (mean of latent distribution)
    if quant_params.use_var_in_features:
        train_variances = train_data['logvar']  # Also load logvar if needed
        print(f"Loaded train latents: mu={train_latents.shape}, logvar={train_variances.shape}")
    else:
        train_variances = None
        print(f"Loaded train latents: {train_latents.shape}")

    # Load val latents
    val_latents_path = latents_dir() / "val_latents.npz"
    if not val_latents_path.exists():
        raise FileNotFoundError(
            f"Latents not found at {val_latents_path}\n"
            "Run extraction first: python -m vq_vae_geodesic.scripts.extract_mnist_latents"
        )

    val_data = np.load(val_latents_path)
    val_latents = val_data['mu']
    if quant_params.use_var_in_features:
        val_variances = val_data['logvar']
        print(f"Loaded val latents: mu={val_latents.shape}, logvar={val_variances.shape}")
    else:
        val_variances = None
        print(f"Loaded val latents: {val_latents.shape}")

    # Create quantizer
    print("\nBuilding geodesic quantizer...")
    quantizer = GeodesicQuantizer(
        n_codewords=quant_params.n_codewords,
        n_chunks=quant_params.n_chunks,
        use_var=quant_params.use_var_in_features
    )

    # Fit codebook on train latents
    print("Fitting codebook with geodesic distances...")
    quantizer.fit(train_latents)
    print(f"Codebook shape: {quantizer.codebook_chunks.shape}")

    # Assign train latents to codebook (get assigned codes)
    print("Assigning train latents to codebook...")
    train_codes = quantizer.assign(train_latents)  # (N_train, n_chunks)
    print(f"Train codes shape: {train_codes.shape}")

    # Assign val latents to codebook
    print("Assigning val latents to codebook...")
    val_codes = quantizer.assign(val_latents)  # (N_val, n_chunks)
    print(f"Val codes shape: {val_codes.shape}")

    # Save codebook and codes
    save_path = latents_dir() / "chunk_codebook.npz"
    quantizer.save(save_path, codes_per_image=train_codes)
    print(f"\nCodebook and train codes saved to {save_path}")

    # Save val codes separately
    val_codes_path = latents_dir() / "val_codes.npz"
    np.savez_compressed(val_codes_path, codes_per_image=val_codes)
    print(f"Val codes saved to {val_codes_path}")

    print("\nNext step: Evaluate reconstructions")
    print("  python -m vq_vae_geodesic.scripts.reconstruct_mnist")


if __name__ == "__main__":
    launch_quantization()
