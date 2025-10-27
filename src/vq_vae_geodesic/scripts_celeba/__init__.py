"""
Executable scripts for the pipelines.

Run scripts as modules:

VAE + Geodesic Quantization training on CelebA:
    uv run -m src.vq_vae_geodesic.scripts_celeba.train_vae_celeba
    uv run -m src.vq_vae_geodesic.scripts_celeba.extract_celeba_latents
    uv run -m src.vq_vae_geodesic.scripts_celeba.quantize_celeba

VQ-VAE training on CelebA:
    uv run -m src.vq_vae_geodesic.scripts_celeba.train_vqvae_celeba
    uv run -m src.vq_vae_geodesic.scripts_celeba.extract_vqvae_codes_celeba (for pixelCNN prior)

Evaluate reconstructions and metrics for VAE, VAE + Geodesic, and VQ-VAE on CelebA:
    uv run -m src.vq_vae_geodesic.scripts_celeba.evaluate_recon_celeba

Interpolation comparison between VAE + Geodesic and VQ-VAE:
    uv run -m src.vq_vae_geodesic.scripts_celeba.interpolate_latent_codes_celeba

VAE + Geodesic Quantization + PixelCNN Prior on CelebA (training):
    uv run -m src.vq_vae_geodesic.scripts_celeba.train_pixelcnn_geodesic_celeba

    uv run -m src.vq_vae_geodesic.scripts_celeba.sample_geodesic_pixelcnn_celeba
    uv run -m src.vq_vae_geodesic.scripts_celeba.compare_geodesic_sampling_celeba
    uv run -m src.vq_vae_geodesic.scripts_celeba.compare_vae_vs_geodesic_celeba
    uv run -m src.vq_vae_geodesic.scripts_celeba.compare_geodesic_temperatures_celeba

VQ-VAE + PixelCNN Prior on CelebA (training):
    uv run -m src.vq_vae_geodesic.scripts_celeba.train_pixelcnn_vqvae_celeba

    uv run -m src.vq_vae_geodesic.scripts_celeba.compare_vqvae_sampling_celeba


Evaluate pixelCNN for VAE + Geodesic and VQ-VAE on CelebA:
    uv run -m src.vq_vae_geodesic.scripts_celeba.evaluate_pixelcnn_celeba


"""
# Note: Scripts are meant to be run as modules, not imported.
