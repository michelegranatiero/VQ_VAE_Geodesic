"""
CIFAR-10 Training and Evaluation Scripts

This package contains all scripts for running the VQ-VAE + Geodesic Quantization
experiments on the CIFAR-10 dataset.

Training Pipeline:
    1. train_vae_cifar10.py - Train VAE on CIFAR-10
    2. extract_cifar10_latents.py - Extract latent representations
    3. quantize_cifar10.py - Perform geodesic quantization
    4. train_vqvae_cifar10.py - Train VQ-VAE end-to-end
    5. extract_vqvae_codes_cifar10.py - Extract VQ-VAE discrete codes
    6. train_pixelcnn_geodesic_cifar10.py - Train PixelCNN on geodesic codes
    7. train_pixelcnn_vqvae_cifar10.py - Train PixelCNN on VQ-VAE codes

Evaluation Scripts:
    - evaluate_recon_cifar10.py - Reconstruction quality metrics
    - evaluate_pixelcnn_cifar10.py - PixelCNN log-likelihood
    - sample_geodesic_pixelcnn_cifar10.py - Generate samples
    - compare_geodesic_sampling_cifar10.py - Compare sampling methods
    - compare_vqvae_sampling_cifar10.py - VQ-VAE sampling comparison
    - compare_geodesic_temperatures_cifar10.py - Temperature effects
    - interpolate_latent_codes_cifar10.py - Latent interpolation

Architecture Differences from MNIST:
    - Input: 32×32×3 (vs 28×28×1)
    - Latent grid: 8×8 (vs 4×4)
    - VAE latent_dim: 512 (vs 128)
    - PixelCNN layers: 12 (vs 7)
    - Residual blocks in encoder/decoder
"""

__all__ = [
    # Training pipeline
    'train_vae_cifar10',
    'extract_cifar10_latents',
    'quantize_cifar10',
    'train_vqvae_cifar10',
    'extract_vqvae_codes_cifar10',
    'train_pixelcnn_geodesic_cifar10',
    'train_pixelcnn_vqvae_cifar10',
    
    # Evaluation
    'evaluate_recon_cifar10',
    'evaluate_pixelcnn_cifar10',
    'sample_geodesic_pixelcnn_cifar10',
    'compare_geodesic_sampling_cifar10',
    'compare_vqvae_sampling_cifar10',
    'compare_geodesic_temperatures_cifar10',
    'interpolate_latent_codes_cifar10',
]
