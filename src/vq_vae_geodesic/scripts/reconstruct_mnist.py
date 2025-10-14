"""
Evaluate reconstructions on MNIST using geodesic quantization.
"""
import torch

from vq_vae_geodesic.config import checkpoint_dir, latents_dir, recons_dir
from vq_vae_geodesic.hyperparameters import get_mnist_config
from vq_vae_geodesic.data.loaders import get_MNIST_loaders
from vq_vae_geodesic.models.modules.vae import build_vae_from_config
from vq_vae_geodesic.evaluation.reconstruct_codebook import (
    evaluate_vae_reconstruction,
    evaluate_geodesic_reconstruction,
    save_reconstruction_plot
)
from vq_vae_geodesic.utils import set_seed


def launch_reconstruction():
    """Evaluate reconstructions on MNIST using geodesic quantization."""
    config = get_mnist_config()
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    checkpoint_path = checkpoint_dir() / "main_checkpoint_mnist.pt"
    model = build_vae_from_config(config.arch_params)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Load codebook
    codebook_path = latents_dir() / "chunk_codebook.pt"
    if not codebook_path.exists():
        raise FileNotFoundError(
            f"Codebook not found at {codebook_path}\n"
            "Run quantization first: python -m vq_vae_geodesic.scripts.quantize_mnist"
        )
    codebook_data = torch.load(codebook_path, map_location=device)
    codebook_chunks = codebook_data['codebook_chunks']

    # Load assigned codes (torch tensor)
    codes_path = latents_dir() / "assigned_codes.pt"
    if not codes_path.exists():
        raise FileNotFoundError(
            f"Codes not found at {codes_path}\n"
            "Run quantization first: python -m vq_vae_geodesic.scripts.quantize_mnist"
        )
    codes_data = torch.load(codes_path, map_location="cpu")
    train_codes = codes_data['train_codes']
    val_codes = codes_data['val_codes']
    test_codes = codes_data['test_codes']
    print(f"Loaded codes: train={tuple(train_codes.shape)}, val={tuple(val_codes.shape)}, test={tuple(test_codes.shape)}")

    # Get data loaders
    train_loader, val_loader, test_loader = get_MNIST_loaders(
        batch_size=config.training_params.batch_size, shuffle_train_set=False
    )

    # Reconstruction output directories
    recons_train_dir = recons_dir() / "train"
    recons_val_dir = recons_dir() / "val"
    recons_test_dir = recons_dir() / "test"
    recons_train_dir.mkdir(parents=True, exist_ok=True)
    recons_val_dir.mkdir(parents=True, exist_ok=True)
    recons_test_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate on train set
    print("\n=== Train Set ===")
    print("VAE reconstruction (with sampling)...")
    mse_vae_train, orig_train, recon_vae_train = evaluate_vae_reconstruction(
        model, train_loader, device, n_show=8
    )

    print("Reconstruction with geodesic quantization...")
    mse_geodesic_train, _, recon_geodesic_train = evaluate_geodesic_reconstruction(
        model, train_loader, codebook_chunks, train_codes,
        device, n_show=8
    )

    # Save single plot with all reconstructions
    save_reconstruction_plot(
        orig_train, recon_vae_train, recon_geodesic_train,
        recons_train_dir / "recon_all.png",
        title_orig="Original", title_vae="VAE Recon", title_geodesic="VAE + Geodesic Recon"
    )

    loss_increase_train = ((mse_geodesic_train - mse_vae_train) / mse_vae_train) * 100
    print(f"\nTrain MSE (VAE): {mse_vae_train:.6f}")
    print(f"Train MSE (Geodesic): {mse_geodesic_train:.6f}")
    print(f"Geodesic degradation: {loss_increase_train:.2f}%")

    # Evaluate on validation set
    print("\n=== Validation Set ===")
    print("VAE reconstruction (with sampling)...")
    mse_vae_val, orig_val, recon_vae_val = evaluate_vae_reconstruction(
        model, val_loader, device, n_show=8
    )

    print("Reconstruction with geodesic quantization...")
    mse_geodesic_val, _, recon_geodesic_val = evaluate_geodesic_reconstruction(
        model, val_loader, codebook_chunks, val_codes,
        device, n_show=8
    )

    # Save single plot with all reconstructions
    save_reconstruction_plot(
        orig_val, recon_vae_val, recon_geodesic_val,
        recons_val_dir / "recon_all.png",
        title_orig="Original", title_vae="VAE Recon", title_geodesic="VAE + Geodesic Recon"
    )

    loss_increase_val = ((mse_geodesic_val - mse_vae_val) / mse_vae_val) * 100
    print(f"\nValidation MSE (VAE): {mse_vae_val:.6f}")
    print(f"Validation MSE (Geodesic): {mse_geodesic_val:.6f}")
    print(f"Geodesic degradation: {loss_increase_val:.2f}%")

    # Evaluate on test set
    print("\n=== Test Set ===")
    print("VAE reconstruction (with sampling)...")
    mse_vae_test, orig_test, recon_vae_test = evaluate_vae_reconstruction(
        model, test_loader, device, n_show=8
    )

    print("Reconstruction with geodesic quantization...")
    mse_geodesic_test, _, recon_geodesic_test = evaluate_geodesic_reconstruction(
        model, test_loader, codebook_chunks, test_codes,
        device, n_show=8
    )

    # Save single plot with all reconstructions
    save_reconstruction_plot(
        orig_test, recon_vae_test, recon_geodesic_test,
        recons_test_dir / "recon_all.png",
        title_orig="Original", title_vae="VAE Recon", title_geodesic="VAE + Geodesic Recon"
    )

    loss_increase_test = ((mse_geodesic_test - mse_vae_test) / mse_vae_test) * 100
    print(f"\nTest MSE (VAE): {mse_vae_test:.6f}")
    print(f"Test MSE (Geodesic): {mse_geodesic_test:.6f}")
    print(f"Geodesic degradation: {loss_increase_test:.2f}%")

    print(f"\nReconstructions saved to {recons_dir()}")


if __name__ == "__main__":
    launch_reconstruction()
