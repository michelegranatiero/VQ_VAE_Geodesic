"""
Executable scripts for the pipelines.

Run scripts as modules:

VQ-VAE + Geodesic Quantization + PixelCNN Prior on MNIST:
    uv run -m src.vq_vae_geodesic.scripts.train_vae_mnist
    uv run -m src.vq_vae_geodesic.scripts.extract_mnist_latents
    uv run -m src.vq_vae_geodesic.scripts.quantize_mnist
    uv run -m src.vq_vae_geodesic.scripts.reconstruct_mnist

    uv run -m src.vq_vae_geodesic.scripts.train_pixelcnn_mnist
    uv run -m src.vq_vae_geodesic.scripts.sample_pixelcnn_mnist
    uv run -m src.vq_vae_geodesic.scripts.compare_pixelcnn_vs_random
    uv run -m src.vq_vae_geodesic.scripts.compare_temperatures

    uv run -m src.vq_vae_geodesic.scripts.train_vqvae_mnist
    uv run -m src.vq_vae_geodesic.scripts.compare_vae_vs_vqvae (reconstructions + metrics)

    uv run -m src.vq_vae_geodesic.scripts.train_pixelcnn_vqvae_mnist
    uv run -m src.vq_vae_geodesic.scripts.compare_sampling_methods

    uv run -m src.vq_vae_geodesic.scripts.print_final_results


"""
# Note: Scripts are meant to be run as modules, not imported.
