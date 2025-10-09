"""
Executable scripts for the VQ-VAE pipeline.

Run scripts as modules:
    uv run -m src.vq_vae_geodesic.scripts.train_vae_mnist
    uv run -m src.vq_vae_geodesic.scripts.extract_mnist_latents
    uv run -m src.vq_vae_geodesic.scripts.quantize_mnist
    uv run -m src.vq_vae_geodesic.scripts.reconstruct_mnist
"""
# Note: Scripts are meant to be run as modules, not imported
# So we don't import them here to avoid slow sklearn imports
