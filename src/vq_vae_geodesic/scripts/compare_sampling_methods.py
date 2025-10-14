"""
Compare different sampling methods for generating new images.

Compares three approaches:
1. VQ-VAE random sampling (sample codes → decode)
2. VAE + Geodesic with PixelCNN (learned prior)
3. VAE + Geodesic random sampling (uniform baseline)

This shows the impact of:
- End-to-end learning (VQ-VAE) vs post-hoc quantization (Geodesic)
- Learned prior (PixelCNN) vs uniform random sampling

Usage:
    python -m vq_vae_geodesic.scripts.compare_sampling_methods
"""
import torch
import numpy as np
import matplotlib.pyplot as plt

from vq_vae_geodesic.config import checkpoint_dir, data_dir
from vq_vae_geodesic.hyperparameters import get_mnist_config
from vq_vae_geodesic.utils import set_seed
from vq_vae_geodesic.models.modules.vqvae import build_vqvae_from_config
from vq_vae_geodesic.models.modules.vae import build_vae_from_config
from vq_vae_geodesic.models.modules.pixelCNN import PixelCNN
from vq_vae_geodesic.evaluation.sample import sample_pixelcnn


def sample_vqvae_pixelcnn(vqvae, pixelcnn, n_samples, device, temperature=1.0):
    """
    Sample from VQ-VAE using PixelCNN learned prior.
    
    This is VQ-VAE with a learned autoregressive prior over codes.
    
    Args:
        vqvae: Trained VQ-VAE model
        pixelcnn: Trained PixelCNN prior for VQ-VAE codes
        n_samples: Number of samples to generate
        device: Device for computation
        temperature: Sampling temperature
        
    Returns:
        images: Generated images (n_samples, C, H, W)
    """
    vqvae.eval()
    pixelcnn.eval()
    
    # VQ-VAE uses spatial latent codes (H, W)
    latent_h = 7
    latent_w = 7
    
    # Sample codes using PixelCNN
    codes = sample_pixelcnn(
        model=pixelcnn,
        device=device,
        img_size=(latent_h, latent_w),
        temperature=temperature,
        batch_size=int(n_samples),
        progress=True
    )
    
    # Convert to torch tensor
    codes_t = torch.from_numpy(codes).long().to(device)
    
    # Map codes to embeddings
    embeddings = vqvae.vq.embeddings.weight
    quantized_latents = embeddings[codes_t]
    quantized_latents = quantized_latents.permute(0, 3, 1, 2).contiguous()
    
    # Decode
    with torch.no_grad():
        generated_images = vqvae.decoder(quantized_latents)
    
    return generated_images.cpu().numpy()


def sample_vqvae_random(vqvae, n_samples, device):
    """
    Sample from VQ-VAE using random discrete codes (no prior).
    
    This is the baseline for VQ-VAE: uniform random sampling from codebook.
    
    Args:
        vqvae: Trained VQ-VAE model
        n_samples: Number of samples to generate
        device: Device for computation
        
    Returns:
        images: Generated images (n_samples, C, H, W)
    """
    vqvae.eval()
    # VQ-VAE uses spatial latent codes (H, W)
    latent_h = 7
    latent_w = 7
    num_embeddings = vqvae.vq.num_embeddings

    # Trova i codeword effettivamente usati dall'encoder sui dati reali (train set)
    from vq_vae_geodesic.data.loaders import get_MNIST_loaders
    config = get_mnist_config()
    _, train_loader, _ = get_MNIST_loaders(batch_size=256, shuffle_train_set=False)
    used_codes_set = set()
    for i, (data, _) in enumerate(train_loader):
        data = data.to(device)
        with torch.no_grad():
            z = vqvae.encoder(data)
        # z: (B, D, 7, 7)
        z_flat = z.permute(0, 2, 3, 1).contiguous().view(-1, vqvae.vq.embedding_dim)
        embeddings = vqvae.vq.embeddings.weight
        distances = (
            torch.sum(z_flat**2, dim=1, keepdim=True)
            + torch.sum(embeddings**2, dim=1)
            - 2 * torch.matmul(z_flat, embeddings.t())
        )
        encoding_indices = torch.argmin(distances, dim=1)
        used_codes_set.update(encoding_indices.cpu().numpy())
        if i > 10:
            break  # Solo alcune batch per velocità
    used_codes = torch.tensor(list(used_codes_set), device=device)
    print(f"[VQ-VAE random] Found {len(used_codes)} used codes out of {num_embeddings} total codes.")

    # Campiona solo dai codeword usati
    random_indices = torch.randint(0, len(used_codes), (n_samples, latent_h, latent_w), device=device)
    random_codes = used_codes[random_indices]

    # Map codes to embeddings
    embeddings = vqvae.vq.embeddings.weight
    quantized_latents = embeddings[random_codes]
    quantized_latents = quantized_latents.permute(0, 3, 1, 2).contiguous()

    # Decode
    with torch.no_grad():
        generated_images = vqvae.decoder(quantized_latents)

    return generated_images.cpu().numpy()


def sample_geodesic_pixelcnn(vae, quantizer, pixelcnn, n_samples, grid_shape, device, temperature=1.0):
    """
    Sample from VAE + Geodesic using PixelCNN learned prior.
    
    This is the full pipeline with learned autoregressive prior.
    
    Args:
        vae: Trained VAE model (decoder only)
        quantizer: GeodesicQuantizer with codebook
        pixelcnn: Trained PixelCNN prior
        n_samples: Number of samples to generate
        grid_shape: (H, W) shape of code grid
        device: Device for computation
        temperature: Sampling temperature
        
    Returns:
        images: Generated images (n_samples, C, H, W)
    """
    vae.eval()
    pixelcnn.eval()
    
    # Sample codes using PixelCNN (correct signature)
    codes = sample_pixelcnn(
        model=pixelcnn,
        device=device,
        img_size=grid_shape,
        temperature=temperature,
        batch_size=int(n_samples),
        progress=True
    )
    
    # Convert codes to images via codebook lookup and decoder
    images = codes_to_images(codes, quantizer, vae, device)
    
    return images


def sample_geodesic_random(vae, quantizer, n_samples, grid_shape, device):
    """
    Sample from VAE + Geodesic using random codes (no prior).
    
    This is the baseline: uniform random sampling from codebook.
    
    Args:
        vae: Trained VAE model (decoder only)
        quantizer: GeodesicQuantizer with codebook
        n_samples: Number of samples to generate
        grid_shape: (H, W) shape of code grid
        device: Device for computation
        
    Returns:
        images: Generated images (n_samples, C, H, W)
    """
    vae.eval()
    H, W = grid_shape
    num_tokens = quantizer.shape[0]

    # Trova i codeword effettivamente usati dal quantizer sui dati reali (train set)
    from vq_vae_geodesic.data.loaders import get_MNIST_loaders
    config = get_mnist_config()
    _, train_loader, _ = get_MNIST_loaders(batch_size=256, shuffle_train_set=False)
    used_codes_set = set()
    # Carica i codici assegnati dal quantizer (train set)
    import os
    import torch
    latents_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), '../../data/latents')
    assigned_codes_path = os.path.join(latents_dir, 'assigned_codes.pt')
    assigned_codes = torch.load(assigned_codes_path, map_location=device)
    train_codes = assigned_codes['train_codes'] if 'train_codes' in assigned_codes else assigned_codes['codes_per_image']
    used_codes_set.update(np.unique(train_codes.cpu().numpy()))
    used_codes = torch.tensor(list(used_codes_set), device=device)
    print(f"[Geodesic random] Found {len(used_codes)} used codes out of {num_tokens} total codes.")

    # Campiona solo dai codeword usati
    random_indices = torch.randint(0, len(used_codes), (n_samples, H, W), device=device)
    random_codes = used_codes[random_indices]
    codes_np = random_codes.cpu().numpy()

    # Convert codes to images
    images = codes_to_images(codes_np, quantizer, vae, device)

    return images


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


def plot_sampling_comparison(vqvae_pixelcnn_imgs, vqvae_imgs, geodesic_pixelcnn_imgs, geodesic_random_imgs, save_path, n_show=8):
    """
    Plot comparison of different sampling methods.
    
    Args:
        vqvae_pixelcnn_imgs: Images from VQ-VAE + PixelCNN (N, C, H, W)
        vqvae_imgs: Images from VQ-VAE random sampling (N, C, H, W)
        geodesic_pixelcnn_imgs: Images from Geodesic + PixelCNN (N, C, H, W)
        geodesic_random_imgs: Images from Geodesic random sampling (N, C, H, W)
        save_path: Path to save figure
        n_show: Number of samples to show per method
    """
    n = min(n_show, len(vqvae_pixelcnn_imgs), len(vqvae_imgs), len(geodesic_pixelcnn_imgs), len(geodesic_random_imgs))
    
    fig, axes = plt.subplots(4, n, figsize=(2*n, 9))
    
    # Row 1: VQ-VAE + PixelCNN
    for i in range(n):
        ax = axes[0, i]
        ax.imshow(vqvae_pixelcnn_imgs[i, 0], cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
        if i == 0:
            ax.text(-0.1, 0.5, "VQ-VAE\n+ PixelCNN\n(E2E + Prior)", 
                   fontsize=11, fontweight='bold', rotation=0, ha='right', 
                   va='center', transform=ax.transAxes, color='darkgreen')
    
    # Row 2: VQ-VAE random sampling
    for i in range(n):
        ax = axes[1, i]
        ax.imshow(vqvae_imgs[i, 0], cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
        if i == 0:
            ax.text(-0.1, 0.5, "VQ-VAE\n(Random Codes\nE2E)", 
                   fontsize=11, fontweight='bold', rotation=0, ha='right', 
                   va='center', transform=ax.transAxes, color='green')
    
    # Row 3: Geodesic + PixelCNN
    for i in range(n):
        ax = axes[2, i]
        ax.imshow(geodesic_pixelcnn_imgs[i, 0], cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
        if i == 0:
            ax.text(-0.1, 0.5, "VAE + Geodesic\n+ PixelCNN\n(Post + Prior)", 
                   fontsize=11, fontweight='bold', rotation=0, ha='right', 
                   va='center', transform=ax.transAxes, color='blue')
    
    # Row 4: Geodesic random sampling
    for i in range(n):
        ax = axes[3, i]
        ax.imshow(geodesic_random_imgs[i, 0], cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
        if i == 0:
            ax.text(-0.1, 0.5, "VAE + Geodesic\n(Random Codes\nPost)", 
                   fontsize=11, fontweight='bold', rotation=0, ha='right', 
                   va='center', transform=ax.transAxes, color='darkred')
    
    fig.suptitle("Sampling Methods Comparison: Generated Images (Not Reconstructions)", 
                fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved comparison to {save_path}")


def compare_sampling_methods(n_samples=16, temperature=1.0):
    """
    Compare different sampling methods for generation.
    
    Args:
        n_samples: Number of samples to generate for each method
        temperature: Temperature for PixelCNN sampling
    """
    config = get_mnist_config()
    set_seed(config.seed)
    
    device = torch.device("cuda:0" if config.use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    grid_shape = (config.quant_params.grid_h, config.quant_params.grid_w)
    num_tokens = config.quant_params.n_codewords
    
    # ============================================================
    # Load Models
    # ============================================================
    print("=" * 60)
    print("Loading Models")
    print("=" * 60)
    
    # 1. VQ-VAE
    print("\n1. Loading VQ-VAE...")
    vqvae_path = checkpoint_dir() / "vqvae_mnist_best.pt"
    
    vqvae = build_vqvae_from_config(config.arch_params, config.vqvae_params)
    checkpoint_vqvae = torch.load(vqvae_path, map_location=device)
    vqvae.load_state_dict(checkpoint_vqvae['model_state_dict'])
    vqvae.to(device)
    print(f"   Loaded from {vqvae_path}")
    
    # 2. VAE
    print("\n2. Loading VAE...")
    vae_path = checkpoint_dir() / "main_checkpoint_mnist.pt"
    vae = build_vae_from_config(config.arch_params)
    checkpoint = torch.load(vae_path, map_location=device)
    # Handle both checkpoint formats
    if 'model_state_dict' in checkpoint:
        vae.load_state_dict(checkpoint['model_state_dict'])
    else:
        vae.load_state_dict(checkpoint)
    vae.to(device)
    print(f"   Loaded from {vae_path}")
    
    # 3. Geodesic Quantizer (torch tensor)
    print("\n3. Loading Geodesic Quantizer...")
    quantizer_path = checkpoint_dir().parent / "latents" / "chunk_codebook.pt"
    codebook_data = torch.load(quantizer_path, map_location=device)
    codebook_chunks = codebook_data['codebook_chunks'] if isinstance(codebook_data, dict) and 'codebook_chunks' in codebook_data else codebook_data
    print(f"   Loaded from {quantizer_path}")
    print(f"   Codebook shape: {tuple(codebook_chunks.shape)}")
    
    # 4. PixelCNN (for Geodesic codes)
    print("\n4. Loading PixelCNN (for Geodesic)...")
    pixelcnn_path = checkpoint_dir() / "pixelcnn_mnist_best.pt"
    
    pixelcnn = PixelCNN(
        num_tokens=num_tokens,
        embed_dim=config.pixelcnn_params.embed_dim,
        hidden_channels=config.pixelcnn_params.hidden_channels,
        n_layers=config.pixelcnn_params.n_layers
    )
    pixelcnn.load_state_dict(torch.load(pixelcnn_path, map_location=device))
    pixelcnn.to(device)
    print(f"   Loaded from {pixelcnn_path}")
    
    # 5. PixelCNN (for VQ-VAE codes)
    print("\n5. Loading PixelCNN (for VQ-VAE)...")
    pixelcnn_vqvae_path = checkpoint_dir() / "pixelcnn_vqvae_mnist_best.pt"
    
    pixelcnn_vqvae = PixelCNN(
        num_tokens=num_tokens,
        embed_dim=config.pixelcnn_params.embed_dim,
        hidden_channels=config.pixelcnn_params.hidden_channels,
        n_layers=config.pixelcnn_params.n_layers
    )
    checkpoint_vqvae = torch.load(pixelcnn_vqvae_path, map_location=device)
    # Handle checkpoint format (might have optimizer, history, etc.)
    if 'model_state_dict' in checkpoint_vqvae:
        pixelcnn_vqvae.load_state_dict(checkpoint_vqvae['model_state_dict'])
    else:
        pixelcnn_vqvae.load_state_dict(checkpoint_vqvae)
    pixelcnn_vqvae.to(device)
    print(f"   Loaded from {pixelcnn_vqvae_path}")
    
    # ============================================================
    # Generate Samples
    # ============================================================
    print("\n" + "=" * 60)
    print("Generating Samples")
    print("=" * 60)
    
    # 1. VQ-VAE + PixelCNN
    print(f"\n1. Sampling {n_samples} images from VQ-VAE + PixelCNN (T={temperature})...")
    vqvae_pixelcnn_samples = sample_vqvae_pixelcnn(
        vqvae, pixelcnn_vqvae, n_samples, device, temperature
    )
    print(f"   Generated shape: {vqvae_pixelcnn_samples.shape}")
    
    # 2. VQ-VAE random sampling
    print(f"\n2. Sampling {n_samples} images from VQ-VAE (random codes)...")
    vqvae_samples = sample_vqvae_random(vqvae, n_samples, device)
    print(f"   Generated shape: {vqvae_samples.shape}")
    
    # 3. Geodesic + PixelCNN
    print(f"\n3. Sampling {n_samples} images from Geodesic + PixelCNN (T={temperature})...")
    geodesic_pixelcnn_samples = sample_geodesic_pixelcnn(
        vae, codebook_chunks, pixelcnn, n_samples, grid_shape, device, temperature
    )
    print(f"   Generated shape: {geodesic_pixelcnn_samples.shape}")

    # 4. Geodesic random sampling
    print(f"\n4. Sampling {n_samples} images from Geodesic (random codes)...")
    geodesic_random_samples = sample_geodesic_random(
        vae, codebook_chunks, n_samples, grid_shape, device
    )
    print(f"   Generated shape: {geodesic_random_samples.shape}")
    
    # ============================================================
    # Plot and Save
    # ============================================================
    print("\n" + "=" * 60)
    print("Saving Results")
    print("=" * 60)
    
    save_dir = data_dir() / "samples"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot comparison
    save_path = save_dir / "sampling_methods_comparison.png"
    plot_sampling_comparison(
        vqvae_pixelcnn_samples,
        vqvae_samples,
        geodesic_pixelcnn_samples,
        geodesic_random_samples,
        save_path,
        n_show=8
    )
    
    # Save raw data as torch
    data_path = save_dir / "sampling_methods_comparison.pt"
    torch.save({
        'vqvae_pixelcnn_samples': vqvae_pixelcnn_samples,
        'vqvae_samples': vqvae_samples,
        'geodesic_pixelcnn_samples': geodesic_pixelcnn_samples,
        'geodesic_random_samples': geodesic_random_samples
    }, data_path)
    print(f"Saved data to {data_path}")
    
    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print("\nFour Sampling Methods Compared:")
    print("  1. VQ-VAE + PixelCNN:")
    print("     - End-to-end learned codebook")
    print("     - Learned autoregressive prior")
    print("     - Best of both worlds (E2E + Prior)")
    print("\n  2. VQ-VAE (Random):")
    print("     - End-to-end learned codebook")
    print("     - No prior (uniform random)")
    print("     - Tests codebook quality alone")
    print("\n  3. Geodesic + PixelCNN:")
    print("     - Post-hoc geodesic quantization")
    print("     - Learned autoregressive prior")
    print("     - Learned prior on geometric codes")
    print("\n  4. Geodesic (Random):")
    print("     - Post-hoc geodesic quantization")
    print("     - No prior (uniform random)")
    print("     - Baseline comparison")
    print("\nExpected Quality Ranking:")
    print("  Best  → VQ-VAE + PixelCNN (E2E + learned prior)")
    print("  Good  → Geodesic + PixelCNN (geometric + learned prior)")
    print("  Mid   → VQ-VAE Random (learned codebook only)")
    print("  Worst → Geodesic Random (no structure)")


if __name__ == "__main__":
    compare_sampling_methods(n_samples=16, temperature=1.0)
