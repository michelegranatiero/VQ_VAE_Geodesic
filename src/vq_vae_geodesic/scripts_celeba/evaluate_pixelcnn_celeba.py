import torch
from vq_vae_geodesic.utils import set_seed
from vq_vae_geodesic.hyperparameters import get_celeba_config
from vq_vae_geodesic.data.loaders import get_codes_loaders
from vq_vae_geodesic.config import latents_dir, checkpoint_dir
from vq_vae_geodesic.models.modules.pixelCNN import build_pixelcnn_from_config

# Import evaluation functions (may need adaptation)
from vq_vae_geodesic.evaluation.evaluate import (
    evaluate_pixelcnn_geodesic_cifar10,
    evaluate_pixelcnn_vqvae_cifar10,
)


def launch_evaluation():
    config = get_celeba_config(img_size=32)
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Geodesic PixelCNN
    codes_path = latents_dir('celeba') / "assigned_codes_celeba.pt"
    grid_shape = (config.quant_params.grid_h, config.quant_params.grid_w)
    _, _, test_loader_geo = get_codes_loaders(
        pt_path=codes_path,
        batch_size=config.pixelcnn_params.batch_size,
        num_workers=config.data_params.num_workers,
        grid_shape=grid_shape
    )

    # --- VQ-VAE PixelCNN
    vqvae_assigned_path = latents_dir('celeba') / 'vqvae_assigned_codes_celeba.pt'
    if not vqvae_assigned_path.exists():
        raise FileNotFoundError(f"Expected {vqvae_assigned_path}; run extract_vqvae_codes_celeba first")

    grid_shape_vqvae = (config.vqvae_params.grid_h, config.vqvae_params.grid_w)
    _, _, test_loader_vqvae = get_codes_loaders(
        pt_path=vqvae_assigned_path,
        batch_size=config.pixelcnn_params.batch_size,
        num_workers=config.data_params.num_workers,
        grid_shape=grid_shape_vqvae
    )

    # Load PixelCNN models
    print("Loading Geodesic PixelCNN...")
    pixelcnn_geo_checkpoint = torch.load(
        checkpoint_dir('celeba') / "pixelcnn_geodesic_celeba_best.pt",
        map_location=device
    )
    pixelcnn_geodesic = build_pixelcnn_from_config(config, for_vqvae=False)
    pixelcnn_geodesic.load_state_dict(pixelcnn_geo_checkpoint['model_state_dict'])
    pixelcnn_geodesic = pixelcnn_geodesic.to(device)
    pixelcnn_geodesic.eval()

    print("Loading VQ-VAE PixelCNN...")
    pixelcnn_vq_checkpoint = torch.load(
        checkpoint_dir('celeba') / "pixelcnn_vqvae_celeba_best.pt",
        map_location=device
    )
    pixelcnn_vqvae = build_pixelcnn_from_config(config, for_vqvae=True)
    pixelcnn_vqvae.load_state_dict(pixelcnn_vq_checkpoint['model_state_dict'])
    pixelcnn_vqvae = pixelcnn_vqvae.to(device)
    pixelcnn_vqvae.eval()

    # Evaluate (using CIFAR-10 evaluation functions - they should work generically)
    metrics_geo = evaluate_pixelcnn_geodesic_cifar10(pixelcnn_geodesic, test_loader_geo, device)
    metrics_vq = evaluate_pixelcnn_vqvae_cifar10(pixelcnn_vqvae, test_loader_vqvae, device)

    print("\n=== PixelCNN Test Metrics (CelebA) ===")
    print(f"Geodesic PixelCNN - Loss: {metrics_geo['loss']:.6f}, Perplexity: {metrics_geo['perplexity']:.4f} ({metrics_geo['perplexity_pct']:.1f}%)")
    print(f"VQ-VAE PixelCNN  - Loss: {metrics_vq['loss']:.6f}, Perplexity: {metrics_vq['perplexity']:.4f} ({metrics_vq['perplexity_pct']:.1f}%)")


if __name__ == "__main__":
    launch_evaluation()
