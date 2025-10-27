"""
Executable scripts for the pipelines.

Run scripts as modules:

VAE + Geodesic Quantization training on CIFAR10:
    uv run -m src.vq_vae_geodesic.scripts_cifar.train_vae_cifar10
    uv run -m src.vq_vae_geodesic.scripts_cifar.extract_cifar10_latents
    uv run -m src.vq_vae_geodesic.scripts_cifar.quantize_cifar10

VQ-VAE training on CIFAR10:
    uv run -m src.vq_vae_geodesic.scripts_cifar.train_vqvae_cifar10
    uv run -m src.vq_vae_geodesic.scripts_cifar.extract_vqvae_codes_cifar10 (for pixelCNN prior)

Evaluate reconstructions and metrics for VAE, VAE + Geodesic, and VQ-VAE on CIFAR10:
    uv run -m src.vq_vae_geodesic.scripts_cifar.evaluate_recon_cifar10

Interpolation comparison between VAE + Geodesic and VQ-VAE:
    uv run -m src.vq_vae_geodesic.scripts_cifar.interpolate_latent_codes_cifar10

VAE + Geodesic Quantization + PixelCNN Prior on CIFAR10 (training):
    uv run -m src.vq_vae_geodesic.scripts_cifar.train_pixelcnn_geodesic_cifar10

    uv run -m src.vq_vae_geodesic.scripts_cifar.sample_geodesic_pixelcnn_cifar10
    uv run -m src.vq_vae_geodesic.scripts_cifar.compare_geodesic_sampling_cifar10
    uv run -m src.vq_vae_geodesic.scripts_cifar.compare_vae_vs_geodesic_cifar10
    uv run -m src.vq_vae_geodesic.scripts_cifar.compare_geodesic_temperatures_cifar10

VQ-VAE + PixelCNN Prior on CIFAR10 (training):
    uv run -m src.vq_vae_geodesic.scripts_cifar.train_pixelcnn_vqvae_cifar10

    uv run -m src.vq_vae_geodesic.scripts_cifar.compare_vqvae_sampling_cifar10


Evaluate pixelCNN for VAE + Geodesic and VQ-VAE on CIFAR10:
    uv run -m src.vq_vae_geodesic.scripts_cifar.evaluate_pixelcnn_cifar10


"""
# Note: Scripts are meant to be run as modules, not imported.
