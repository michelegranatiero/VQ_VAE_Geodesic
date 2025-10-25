"""
Interpolate between latent codes of two images and compare reconstruction quality.

Compares interpolation behavior between:
- VAE + Geodesic (post-hoc quantization)
- VQ-VAE (end-to-end learned codebook)

Uses probabilistic blending: at each alpha, randomly picks codes from img1 or img2.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt

from vq_vae_geodesic.config import data_dir, samples_dir
from vq_vae_geodesic.hyperparameters import get_mnist_config
from vq_vae_geodesic.utils import set_seed
from vq_vae_geodesic.data.loaders import get_MNIST_loaders
from vq_vae_geodesic.evaluation.utils import (
    load_model_vae_mnist,
    load_model_vqvae_mnist,
    load_codebook_mnist,
    lookup_codewords
)


def encode_geodesic_to_codes(vae, codebook_chunks, image, device):
    """Encode image to discrete codes using VAE + Geodesic quantization."""
    vae.eval()
    with torch.no_grad():
        mu, _ = vae.encoder(image)
        
        # Get config
        config = get_mnist_config()
        grid_h = config.quant_params.grid_h
        grid_w = config.quant_params.grid_w
        latent_dim = config.arch_params.latent_dim
        chunk_size = config.quant_params.chunk_size(latent_dim)
        
        # Reshape to chunks
        latent_chunks = mu.view(1, grid_h * grid_w, chunk_size)
        
        # Find nearest codewords
        distances = torch.cdist(latent_chunks[0], codebook_chunks, p=2)
        codes = torch.argmin(distances, dim=1).view(grid_h, grid_w)
        
        return codes


def decode_geodesic_codes(codes, codebook_chunks, decoder, device):
    """Decode geodesic codes to image."""
    codes_flat = codes.flatten().unsqueeze(0)  # (1, H*W)
    latents = lookup_codewords(codebook_chunks, codes_flat)  # (1, latent_dim)
    
    decoder.eval()
    with torch.no_grad():
        image = decoder(latents)
    
    return image


def encode_vqvae_to_codes(vqvae, image, device):
    """Encode image to discrete codes using VQ-VAE."""
    vqvae.eval()
    with torch.no_grad():
        z_e = vqvae.encoder(image)
        _, _, codes = vqvae.vq(z_e)
        return codes[0]  # (H, W)


def decode_vqvae_codes(codes, vqvae, device):
    """Decode VQ-VAE codes to image."""
    # codes: (H, W)
    codes_batch = codes.unsqueeze(0)  # (1, H, W)
    
    # Map to embeddings
    embeddings = vqvae.vq.embeddings.weight
    quantized = embeddings[codes_batch]  # (1, H, W, embed_dim)
    quantized = quantized.permute(0, 3, 1, 2).contiguous()
    
    vqvae.eval()
    with torch.no_grad():
        image = vqvae.decoder(quantized)
    
    return image


def interpolate_codes(codes1, codes2, n_steps=8):
    """
    Probabilistic interpolation between two code grids.
    
    For each alpha in [0, 1], randomly pick codes from codes1 or codes2
    at each spatial position, with probability alpha of picking codes2.
    
    Args:
        codes1: First code grid (H, W) - tensor
        codes2: Second code grid (H, W) - tensor
        n_steps: Number of interpolation steps
        
    Returns:
        List of interpolated code grids
    """
    interpolated = []
    
    for alpha in torch.linspace(0, 1, steps=n_steps):
        # For each location, randomly pick from codes1 or codes2 based on alpha
        mask = (torch.rand_like(codes1.float()) < alpha).long()
        blended = codes1 * (1 - mask) + codes2 * mask
        interpolated.append(blended)
    
    return interpolated


def run_interpolation():
    """Main interpolation comparison."""
    config = get_mnist_config()
    set_seed(config.seed)
    
    device = torch.device("cuda:0" if config.use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Load models
    print("Loading models...")
    vae = load_model_vae_mnist(config.arch_params, device)
    codebook_chunks = load_codebook_mnist(device)
    vqvae = load_model_vqvae_mnist(config.arch_params, config.vqvae_params, device)
    print("Models loaded\n")
    
    # Get data
    print("Loading data...")
    train_loader, _, _ = get_MNIST_loaders(batch_size=64, shuffle_train_set=False)
    images, _ = next(iter(train_loader))
    images = images.to(device)
    
    img1 = images[0:1]
    img2 = images[10:11]
    print(f"Selected images: shape={img1.shape}\n")
    
    # Encode both images to codes
    print("Encoding to codes...")
    geodesic_codes1 = encode_geodesic_to_codes(vae, codebook_chunks, img1, device)
    geodesic_codes2 = encode_geodesic_to_codes(vae, codebook_chunks, img2, device)
    
    vqvae_codes1 = encode_vqvae_to_codes(vqvae, img1, device)
    vqvae_codes2 = encode_vqvae_to_codes(vqvae, img2, device)
    print(f"Geodesic codes: {geodesic_codes1.shape}")
    print(f"VQ-VAE codes: {vqvae_codes1.shape}\n")
    
    # Interpolate
    n_steps = 16
    print(f"Interpolating ({n_steps} steps)...")
    
    geodesic_interpolated_codes = interpolate_codes(geodesic_codes1, geodesic_codes2, n_steps)
    vqvae_interpolated_codes = interpolate_codes(vqvae_codes1, vqvae_codes2, n_steps)
    
    # Decode to images
    print("Decoding interpolated codes...")
    geodesic_images = [decode_geodesic_codes(c, codebook_chunks, vae.decoder, device) 
                       for c in geodesic_interpolated_codes]
    vqvae_images = [decode_vqvae_codes(c, vqvae, device) 
                    for c in vqvae_interpolated_codes]
    
    # Plot
    print("Plotting...")
    fig, axes = plt.subplots(3, n_steps, figsize=(2*n_steps, 7))
    
    # Row 0: Original images
    for i in range(n_steps):
        ax = axes[0, i]
        if i == 0:
            ax.imshow(img1[0, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
            ax.set_title('Image 1', fontsize=9)
            ax.text(-0.1, 0.5, 'Original', fontsize=11, fontweight='bold',
                   rotation=90, ha='right', va='center', transform=ax.transAxes)
        elif i == n_steps - 1:
            ax.imshow(img2[0, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
            ax.set_title('Image 2', fontsize=9)
        else:
            # Empty middle cells
            ax.imshow(np.ones((28, 28)) * 0.5, cmap='gray', vmin=0, vmax=1)
            ax.set_title(f'α={i/(n_steps-1):.2f}', fontsize=9)
        ax.axis('off')
    
    # Row 1: VAE + Geodesic
    for i, img in enumerate(geodesic_images):
        ax = axes[1, i]
        ax.imshow(img[0, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
        
        if i == 0:
            ax.text(-0.1, 0.5, 'VAE+Geodesic', fontsize=11, fontweight='bold',
                   rotation=90, ha='right', va='center', transform=ax.transAxes)
    
    # Row 2: VQ-VAE
    for i, img in enumerate(vqvae_images):
        ax = axes[2, i]
        ax.imshow(img[0, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
        
        if i == 0:
            ax.text(-0.1, 0.5, 'VQ-VAE', fontsize=11, fontweight='bold',
                   rotation=90, ha='right', va='center', transform=ax.transAxes)
    
    plt.suptitle('Latent Code Interpolation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save
    save_dir = samples_dir("mnist")
    save_dir.mkdir(exist_ok=True, parents=True)
    
    plot_path = save_dir / "interpolation_data.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {plot_path}")
    
    # Save data
    data_path = save_dir / "interpolation_data.pt"
    torch.save({
        'img1': img1.cpu(),
        'img2': img2.cpu(),
        'geodesic_codes1': geodesic_codes1.cpu(),
        'geodesic_codes2': geodesic_codes2.cpu(),
        'vqvae_codes1': vqvae_codes1.cpu(),
        'vqvae_codes2': vqvae_codes2.cpu(),
        'geodesic_images': [img.cpu() for img in geodesic_images],
        'vqvae_images': [img.cpu() for img in vqvae_images],
    }, data_path)
    print(f"Saved data: {data_path}")

if __name__ == "__main__":
    run_interpolation()
