"""
Executable scripts for the pipelines.

Run scripts as modules:

VAE + Geodesic Quantization training on MNIST:
    uv run -m src.vq_vae_geodesic.scripts.train_vae_mnist
    uv run -m src.vq_vae_geodesic.scripts.extract_mnist_latents
    uv run -m src.vq_vae_geodesic.scripts.quantize_mnist

VQ-VAE training on MNIST:
    uv run -m src.vq_vae_geodesic.scripts.train_vqvae_mnist
    uv run -m src.vq_vae_geodesic.scripts.extract_vqvae_codes_mnist (for pixelCNN prior)

Evaluate reconstructions and metrics for VAE, VAE + Geodesic, and VQ-VAE on MNIST:
    uv run -m src.vq_vae_geodesic.scripts.evaluate_recon_mnist

Interpolation comparison between VAE + Geodesic and VQ-VAE:
    uv run -m src.vq_vae_geodesic.scripts.interpolate_latent_codes
    
VAE + Geodesic Quantization + PixelCNN Prior on MNIST (training):
    uv run -m src.vq_vae_geodesic.scripts.train_pixelcnn_geodesic_mnist

    uv run -m src.vq_vae_geodesic.scripts.sample_geodesic_pixelcnn_mnist
    uv run -m src.vq_vae_geodesic.scripts.compare_geodesic_sampling
    uv run -m src.vq_vae_geodesic.scripts.compare_geodesic_temperatures

VQ-VAE + PixelCNN Prior on MNIST (training):
    uv run -m src.vq_vae_geodesic.scripts.train_pixelcnn_vqvae_mnist

    uv run -m src.vq_vae_geodesic.scripts.compare_vqvae_sampling


Evaluate pixelCNN for VAE + Geodesic and VQ-VAE on MNIST:
    uv run -m src.vq_vae_geodesic.scripts.evaluate_pixelcnn_mnist


"""
# Note: Scripts are meant to be run as modules, not imported.
