"""
Compare PixelCNN sampling vs random sampling.

Generates samples using:
1. PixelCNN prior (learned autoregressive distribution)
2. Random uniform sampling from codebook (baseline)

This comparison shows the benefit of learning the prior distribution.

Usage:
    python -m vq_vae_geodesic.scripts.compare_pixelcnn_vs_random
"""
import torch
import numpy as np
import matplotlib.pyplot as plt

from vq_vae_geodesic.config import checkpoint_dir, latents_dir, data_dir
from vq_vae_geodesic.hyperparameters import get_mnist_config
from vq_vae_geodesic.utils import set_seed
from vq_vae_geodesic.models.modules.pixelCNN import PixelCNN
from vq_vae_geodesic.models.modules.vae import build_vae_from_config
from vq_vae_geodesic.evaluation.sample import sample_pixelcnn


def sample_random_codes(n_samples, grid_shape, num_tokens, device):
    """
    Sample random discrete codes uniformly from codebook.
    
    This is the baseline: no learned prior, just uniform random sampling.
    
    Args:
        n_samples: Number of samples to generate
        grid_shape: (H, W) shape of code grid
        num_tokens: K, vocabulary size
        device: Device for computation
        
    Returns:
        codes: Random codes (n_samples, H, W)
    """
    H, W = grid_shape
    # Uniform random sampling from [0, K-1]
    codes = torch.randint(0, num_tokens, (n_samples, H, W), device=device)
    return codes.cpu().numpy()


def codes_to_images(codes, quantizer, vae, device):
    """
    Convert discrete codes to images via codebook lookup and decoder.
    
    Args:
        codes: Discrete codes (B, H, W)
        quantizer: GeodesicQuantizer with codebook
        vae: VAE model with decoder
        device: Device for computation
        
    Returns:
        images: Generated images (B, C, H, W)
    """
    n_samples = codes.shape[0]
    latents_list = []

    # Accept quantizer as codebook_chunks tensor or GeodesicQuantizer
    if hasattr(quantizer, 'codebook_chunks'):
        codebook_chunks = quantizer.codebook_chunks
    else:
        codebook_chunks = quantizer

    for b in range(n_samples):
        code_indices = codes[b].flatten()
        chunks = codebook_chunks[torch.from_numpy(code_indices).long()]
        latent = chunks.flatten()
        latents_list.append(latent.cpu().numpy())

    latents = np.stack(latents_list)
    latents_t = torch.from_numpy(latents).float().to(device)

    # Decode
    with torch.no_grad():
        images = vae.decoder(latents_t)

    return images.cpu().numpy()


def plot_comparison_grid(pixelcnn_imgs, random_imgs, save_path, n_show=16):
    """
    Plot side-by-side comparison of PixelCNN vs random samples.
    
    Args:
        pixelcnn_imgs: Images from PixelCNN (N, C, H, W)
        random_imgs: Images from random sampling (N, C, H, W)
        save_path: Path to save figure
        n_show: Number of samples to show
    """
    n = min(n_show, len(pixelcnn_imgs), len(random_imgs))
    
    fig, axes = plt.subplots(2, n, figsize=(2*n, 5))
    
    # PixelCNN samples (top row)
    for i in range(n):
        ax = axes[0, i]
        ax.imshow(pixelcnn_imgs[i, 0], cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
        if i == 0:
            ax.text(-0.1, 0.5, "PixelCNN\n(Autoregressive\nLearned Prior)", 
                   fontsize=11, fontweight='bold', rotation=0, ha='right', 
                   va='center', transform=ax.transAxes, color='darkgreen')
    
    # Random samples (bottom row)
    for i in range(n):
        ax = axes[1, i]
        ax.imshow(random_imgs[i, 0], cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
        if i == 0:
            ax.text(-0.1, 0.5, "Random\nSampling\n(Uniform Baseline)", 
                   fontsize=11, fontweight='bold', rotation=0, ha='right', 
                   va='center', transform=ax.transAxes, color='darkred')
    
    fig.suptitle("PixelCNN Autoregressive Prior vs Random Uniform Sampling", 
                fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved comparison to {save_path}")


def compare_pixelcnn_vs_random(n_samples=16, temperature=1.0):
    """
    Compare PixelCNN sampling vs random uniform sampling.
    
    Args:
        n_samples: Number of samples to generate for each method
        temperature: Temperature for PixelCNN sampling
    """
    config = get_mnist_config()
    set_seed(config.seed)
    
    device = torch.device("cuda:0" if config.use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    grid_shape = (config.quant_params.grid_h, config.quant_params.grid_w)
    num_tokens = config.quant_params.n_codewords
    
    # Load models
    print("\nLoading models...")
    
    # PixelCNN
    pixelcnn_path = checkpoint_dir() / "pixelcnn_geodesic_mnist_best.pt"
    
    pixelcnn = PixelCNN(
        num_tokens=num_tokens,
        embed_dim=config.pixelcnn_params.embed_dim,
        hidden_channels=config.pixelcnn_params.hidden_channels,
        n_layers=config.pixelcnn_params.n_layers,
        kernel_size=config.pixelcnn_params.kernel_size
    )
    
    checkpoint = torch.load(pixelcnn_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        pixelcnn.load_state_dict(checkpoint['model_state_dict'])
    else:
        pixelcnn.load_state_dict(checkpoint)
    
    pixelcnn = pixelcnn.to(device)
    pixelcnn.eval()
    print(f"Loaded PixelCNN from {pixelcnn_path}")
    
    # Codebook (torch tensor)
    codebook_path = latents_dir() / "chunk_codebook.pt"
    codebook_data = torch.load(codebook_path, map_location=device)
    codebook_chunks = codebook_data['codebook_chunks'] if isinstance(codebook_data, dict) and 'codebook_chunks' in codebook_data else codebook_data
    print(f"Loaded codebook: {tuple(codebook_chunks.shape)}")
    
    # VAE
    vae_path = checkpoint_dir() / "main_checkpoint_mnist.pt"
    vae = build_vae_from_config(config.arch_params)
    checkpoint = torch.load(vae_path, map_location=device)
    vae.load_state_dict(checkpoint['model_state_dict'])
    vae = vae.to(device)
    vae.eval()
    print(f"Loaded VAE from {vae_path}")
    
    # Sample with PixelCNN
    print(f"\nSampling {n_samples} codes with PixelCNN (T={temperature})...")
    pixelcnn_codes = sample_pixelcnn(
        pixelcnn,
        device=device,
        img_size=grid_shape,
        temperature=temperature,
        batch_size=n_samples,
        progress=True
    )
    print(f"PixelCNN codes shape: {pixelcnn_codes.shape}")
    
    # Sample randomly
    print(f"\nSampling {n_samples} codes randomly (uniform)...")
    random_codes = sample_random_codes(n_samples, grid_shape, num_tokens, device)
    print(f"Random codes shape: {random_codes.shape}")
    
    # Generate images
    print("\nGenerating images from PixelCNN codes...")
    pixelcnn_images = codes_to_images(pixelcnn_codes, codebook_chunks, vae, device)

    print("Generating images from random codes...")
    random_images = codes_to_images(random_codes, codebook_chunks, vae, device)
    
    print(f"\nPixelCNN images shape: {pixelcnn_images.shape}")
    print(f"Random images shape: {random_images.shape}")
    
    # Save comparison
    save_dir = data_dir() / "samples"
    save_dir.mkdir(exist_ok=True, parents=True)
    
    save_path = save_dir / "pixelcnn_vs_random.png"
    plot_comparison_grid(pixelcnn_images, random_images, save_path, n_show=16)
    
    # Save data as torch
    data_path = save_dir / "pixelcnn_vs_random.pt"
    torch.save({
        'pixelcnn_images': pixelcnn_images,
        'random_images': random_images,
        'pixelcnn_codes': pixelcnn_codes,
        'random_codes': random_codes,
        'temperature': temperature
    }, data_path)
    print(f"Saved data to {data_path}")
    
    print("\n" + "="*60)
    print("Comparison complete!")
    print("="*60)
    print("\nExpected results:")
    print("  - PixelCNN: Realistic digits, learned structure")
    print("  - Random: Noisy, unrealistic, no structure")
    print("\nThis shows the benefit of learning p(z) with PixelCNN")
    print("instead of assuming uniform random latent codes.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare PixelCNN vs random sampling")
    parser.add_argument("--n_samples", type=int, default=16,
                       help="Number of samples to generate")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Temperature for PixelCNN sampling")
    
    args = parser.parse_args()
    
    compare_pixelcnn_vs_random(
        n_samples=args.n_samples,
        temperature=args.temperature
    )
