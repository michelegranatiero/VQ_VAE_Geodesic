import torch
import wandb
from vq_vae_geodesic.utils import set_seed
from vq_vae_geodesic.hyperparameters import get_cifar10_config
from vq_vae_geodesic.data.loaders import get_codes_loaders

from vq_vae_geodesic.evaluation.evaluate import (
    evaluate_pixelcnn_geodesic_cifar10,
    evaluate_pixelcnn_vqvae_cifar10,
)
from vq_vae_geodesic.config import latents_dir
from vq_vae_geodesic.evaluation.utils import load_pixelcnn_checkpoint


def launch_evaluation():
    config = get_cifar10_config()
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Geodesic PixelCNN: use assigned_codes_cifar10.pt (has train/val/test splits)
    codes_path = latents_dir('cifar10') / "assigned_codes_cifar10.pt"
    grid_shape = (config.quant_params.grid_h, config.quant_params.grid_w)
    _, _, test_loader_geo = get_codes_loaders(
        pt_path=codes_path,
        batch_size=config.pixelcnn_params.batch_size,
        num_workers=config.data_params.num_workers,
        grid_shape=grid_shape
    )

    # --- VQ-VAE PixelCNN: use the assigned-like file produced by extract_vqvae_codes_cifar10
    vqvae_assigned_path = latents_dir('cifar10') / 'vqvae_assigned_codes_cifar10.pt'
    if not vqvae_assigned_path.exists():
        raise FileNotFoundError(f"Expected {vqvae_assigned_path}; run extract_vqvae_codes_cifar10 first")

    grid_shape_vqvae = (config.vqvae_params.grid_h, config.vqvae_params.grid_w)
    _, _, test_loader_vqvae = get_codes_loaders(
        pt_path=vqvae_assigned_path,
        batch_size=config.pixelcnn_params.batch_size,
        num_workers=config.data_params.num_workers,
        grid_shape=grid_shape_vqvae
    )

    # Load PixelCNN models
    pixelcnn_geodesic = load_pixelcnn_checkpoint("pixelcnn_geodesic_cifar10_best.pt", config, device, is_vqvae=False)
    pixelcnn_vqvae = load_pixelcnn_checkpoint("pixelcnn_vqvae_cifar10_best.pt", config, device, is_vqvae=True)

    # Evaluate: geodesic uses test_loader_geo, vqvae uses test_loader_vqvae
    metrics_geo = evaluate_pixelcnn_geodesic_cifar10(pixelcnn_geodesic, test_loader_geo, device)
    metrics_vq = evaluate_pixelcnn_vqvae_cifar10(pixelcnn_vqvae, test_loader_vqvae, device)

    print("\n=== PixelCNN Test Metrics (CIFAR-10) ===")
    print(f"Geodesic PixelCNN - Loss: {metrics_geo['loss']:.6f}, Perplexity: {metrics_geo['perplexity']:.4f} ({metrics_geo['perplexity_pct']:.1f}%)")
    print(f"VQ-VAE PixelCNN  - Loss: {metrics_vq['loss']:.6f}, Perplexity: {metrics_vq['perplexity']:.4f} ({metrics_vq['perplexity_pct']:.1f}%)")


if __name__ == "__main__":
    launch_evaluation()
