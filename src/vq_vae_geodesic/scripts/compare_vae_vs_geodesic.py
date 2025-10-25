"""
Compare sampling between:
1. VAE pure (sampling from continuous latent space)
2. VAE + Geodesic quantization (sampling with PixelCNN on discrete codebook)

This shows the effect of discretization on sample quality.
"""
import torch
from pathlib import Path

from vq_vae_geodesic.config import data_dir, samples_dir
from vq_vae_geodesic.hyperparameters import get_mnist_config
from vq_vae_geodesic.utils import set_seed
from vq_vae_geodesic.evaluation.sample import sample_pixelcnn
from vq_vae_geodesic.evaluation.utils import (
    load_pixelcnn_checkpoint,
    load_model_vae_mnist,
    load_codebook_mnist,
    codes_to_images_via_codebook
)
from vq_vae_geodesic.evaluation.visualize import plot_comparison_2rows

N_SAMPLES = 16
TEMPERATURE = 1.0


def sample_vae_pure(vae, n_samples, latent_dim, device):
    """
    Sample from VAE by sampling from standard normal in latent space.
    
    Args:
        vae: Trained VAE model
        n_samples: Number of samples to generate
        latent_dim: Dimension of latent space
        device: Device to use
        
    Returns:
        Generated images as numpy array (N, C, H, W)
    """
    vae.eval()
    with torch.no_grad():
        # Sample from standard normal distribution
        z = torch.randn(n_samples, latent_dim).to(device)
        # Decode to images
        samples = vae.decoder(z)
        samples = torch.clamp(samples, 0, 1)
    
    return samples.cpu().numpy()


def compare_vae_vs_geodesic(n_samples=16, temperature=1.0):
    """
    Compare sampling quality between VAE pure and VAE+Geodesic.
    """
    config = get_mnist_config()
    set_seed(config.seed)
    
    device = torch.device('cuda' if config.use_gpu and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Get configuration
    arch_params = config.arch_params
    latent_dim = arch_params.latent_dim
    grid_shape = (config.quant_params.grid_h, config.quant_params.grid_w)
    
    print("Loading models...")
    # Load VAE
    vae = load_model_vae_mnist(arch_params, device)
    
    # Load PixelCNN (geodesic)
    pixelcnn = load_pixelcnn_checkpoint("pixelcnn_geodesic_mnist_best.pt", config, device)
    
    # Load geodesic codebook
    codebook_chunks = load_codebook_mnist(device)
    
    print("All models loaded successfully\n")
    
    # Sample from VAE pure (continuous latent space)
    print(f"Sampling {n_samples} images from VAE (continuous latent space)...")
    vae_samples = sample_vae_pure(vae, n_samples, latent_dim, device)
    print(f"VAE samples shape: {vae_samples.shape}\n")
    
    # Sample from VAE+Geodesic (PixelCNN on discrete codes)
    print(f"Sampling {n_samples} images from VAE+Geodesic (PixelCNN, T={temperature})...")
    geodesic_codes = sample_pixelcnn(
        pixelcnn,
        device=device,
        img_size=grid_shape,
        temperature=temperature,
        batch_size=n_samples,
        progress=True
    )
    geodesic_samples = codes_to_images_via_codebook(
        geodesic_codes, 
        codebook_chunks, 
        vae.decoder, 
        device
    )
    print(f"Geodesic samples shape: {geodesic_samples.shape}\n")
    
    # Create comparison visualization
    save_path = samples_dir('mnist') / "vae_vs_geodesic_mnist.png"
    
    plot_comparison_2rows(
        top_imgs=vae_samples,
        bottom_imgs=geodesic_samples,
        top_label="VAE Pure",
        bottom_label="VAE+Geodesic",
        save_path=save_path,
        title="VAE Pure vs VAE+Geodesic Quantization (MNIST)",
        n_show=min(16, n_samples)
    )
    
    # Save data
    data_path = save_path.parent / save_path.name.replace('.png', '.pt')
    torch.save({
        'vae_samples': vae_samples,
        'geodesic_samples': geodesic_samples,
        'temperature': temperature,
        'n_samples': n_samples
    }, data_path)
    print(f"Saved comparison data to {data_path}")
    
    print("\nComparison complete!")
    print("VAE Pure: Samples directly from continuous N(0,I) latent space")
    print("VAE+Geodesic: Samples from PixelCNN prior over discrete geodesic codes")


if __name__ == "__main__":
    compare_vae_vs_geodesic(
        n_samples=N_SAMPLES,
        temperature=TEMPERATURE
    )
