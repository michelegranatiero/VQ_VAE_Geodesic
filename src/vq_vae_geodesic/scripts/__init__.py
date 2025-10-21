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

VAE + Geodesic Quantization + PixelCNN Prior on MNIST (training):
    uv run -m src.vq_vae_geodesic.scripts.train_pixelcnn_mnist
    uv run -m src.vq_vae_geodesic.scripts.sample_pixelcnn_mnist ----------

    uv run -m src.vq_vae_geodesic.scripts.compare_pixelcnn_vs_random ----------
    uv run -m src.vq_vae_geodesic.scripts.compare_temperatures ----------

VQ-VAE + PixelCNN Prior on MNIST (training):
    uv run -m src.vq_vae_geodesic.scripts.train_pixelcnn_vqvae_mnist

Evaluate pixelCNN for VAE + Geodesic and VQ-VAE on MNIST:
    uv run -m src.vq_vae_geodesic.scripts.evaluate_pixelcnn_mnist

Comparison
    uv run -m src.vq_vae_geodesic.scripts.compare_sampling_methods ----------

    uv run -m src.vq_vae_geodesic.scripts.print_final_results ----------


"""
# Note: Scripts are meant to be run as modules, not imported.
