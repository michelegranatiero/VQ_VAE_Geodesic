"""
Sample new discrete codes from trained PixelCNN and generate images.

Uses ancestral sampling to generate new latent code grids from the learned
autoregressive prior, then decodes them to images using the VAE decoder.

Usage:
    python -m vq_vae_geodesic.scripts.sample_pixelcnn_mnist
"""
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from vq_vae_geodesic.config import checkpoint_dir, latents_dir, data_dir
from vq_vae_geodesic.hyperparameters import get_mnist_config
from vq_vae_geodesic.utils import set_seed
from vq_vae_geodesic.models.modules.pixelCNN import PixelCNN
from vq_vae_geodesic.models.modules.vae import build_vae_from_config
from vq_vae_geodesic.evaluation.sample import sample_pixelcnn


def plot_samples_grid(samples, out_path, n_rows=4, n_cols=4, title="PixelCNN Samples"):
    """
    Plot a grid of generated samples.
    
    Args:
        samples: Generated images (N, C, H, W) or (N, H, W)
        out_path: Path to save the plot
        n_rows: Number of rows in grid
        n_cols: Number of columns in grid
        title: Title for the figure
    """
    n_samples = min(len(samples), n_rows * n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2*n_cols, 2*n_rows))
    
    for idx, ax in enumerate(axes.flat):
        if idx < n_samples:
            img = samples[idx].squeeze()  # Remove channel dim if present
            ax.imshow(img, cmap='gray', vmin=0, vmax=1)
            ax.axis('off')
            ax.set_title(f"Sample {idx+1}", fontsize=8)
        else:
            ax.axis('off')
    
    fig.suptitle(title, fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved sample grid to {out_path}")


def launch_sample_pixelcnn(n_samples=16, temperature=1.0, save_images=True):
    """
    Sample new images from trained PixelCNN prior + VAE decoder.
    
    Args:
        n_samples: Number of images to generate
        temperature: Sampling temperature (>1 more diverse, <1 sharper)
        save_images: Whether to save generated images
    """
    config = get_mnist_config()
    set_seed(config.seed)
    
    device = torch.device("cuda:0" if config.use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load PixelCNN
    pixelcnn_path = checkpoint_dir() / "pixelcnn_geodesic_mnist_best.pt"
    
    if not pixelcnn_path.exists():
        raise FileNotFoundError(
            f"PixelCNN model not found at {pixelcnn_path}\n"
            "Train PixelCNN first: python -m vq_vae_geodesic.scripts.train_pixelcnn_mnist"
        )
    
    print(f"Loading PixelCNN from {pixelcnn_path}")
    pixelcnn = PixelCNN(
        num_tokens=config.quant_params.n_codewords,
        embed_dim=config.pixelcnn_params.embed_dim,
        hidden_channels=config.pixelcnn_params.hidden_channels,
        n_layers=config.pixelcnn_params.n_layers,
        kernel_size=config.pixelcnn_params.kernel_size
    )
    
    # Load weights
    if pixelcnn_path.suffix == '.pt':
        # Try loading as full checkpoint first
        try:
            checkpoint = torch.load(pixelcnn_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                pixelcnn.load_state_dict(checkpoint['model_state_dict'])
            else:
                pixelcnn.load_state_dict(checkpoint)
        except:
            pixelcnn.load_state_dict(torch.load(pixelcnn_path, map_location=device))
    
    pixelcnn = pixelcnn.to(device)
    pixelcnn.eval()
    
    # Sample codes from PixelCNN
    print(f"\nSampling {n_samples} code grids with temperature {temperature}...")
    grid_shape = (config.quant_params.grid_h, config.quant_params.grid_w)
    
    codes = sample_pixelcnn(
        pixelcnn,
        device=device,
        img_size=grid_shape,
        temperature=temperature,
        batch_size=n_samples,
        progress=True
    )
    
    print(f"Sampled codes shape: {codes.shape}")
    print(f"Code range: [{codes.min()}, {codes.max()}]")
    
    # Load codebook (torch tensor)
    codebook_path = latents_dir() / "chunk_codebook.pt"
    if not codebook_path.exists():
        raise FileNotFoundError(f"Codebook not found at {codebook_path}")

    print(f"Loading codebook from {codebook_path}")
    codebook_data = torch.load(codebook_path, map_location=device)
    codebook_chunks = codebook_data['codebook_chunks'] if isinstance(codebook_data, dict) and 'codebook_chunks' in codebook_data else codebook_data

    # Convert codes to continuous latents
    print("Converting codes to latent vectors...")
    latents_list = []
    for b in range(n_samples):
        code_indices = codes[b].flatten()
        chunks = codebook_chunks[torch.from_numpy(code_indices).long()]
        latent = chunks.flatten()
        latents_list.append(latent.cpu().numpy())
    latents = np.stack(latents_list)
    latents_t = torch.from_numpy(latents).float().to(device)
    print(f"Latents shape: {latents_t.shape}")
    
    # Load VAE decoder
    vae_path = checkpoint_dir() / "main_checkpoint_mnist.pt"
    if not vae_path.exists():
        raise FileNotFoundError(f"VAE checkpoint not found at {vae_path}")
    
    print(f"Loading VAE from {vae_path}")
    vae = build_vae_from_config(config.arch_params)
    checkpoint = torch.load(vae_path, map_location=device)
    vae.load_state_dict(checkpoint['model_state_dict'])
    vae = vae.to(device)
    vae.eval()
    
    # Decode latents to images
    print("Decoding latents to images...")
    with torch.no_grad():
        images = vae.decoder(latents_t)
    
    images = images.cpu().numpy()
    print(f"Generated images shape: {images.shape}")
    
    # Save images using the same style as reconstruction
    if save_images:
        # Use data_dir for consistency with other evaluation scripts
        save_dir = data_dir() / "samples"
        save_dir.mkdir(exist_ok=True, parents=True)

        # Create grid visualization
        save_path = save_dir / f"pixelcnn_samples_grid_t{temperature:.1f}.png"
        n_rows = int(np.ceil(np.sqrt(n_samples)))
        n_cols = int(np.ceil(n_samples / n_rows))
        plot_samples_grid(
            images, 
            save_path, 
            n_rows=n_rows, 
            n_cols=n_cols,
            title=f"PixelCNN Samples (T={temperature})"
        )

        # Save codes and latents as torch
        codes_path = save_dir / f"sampled_codes_t{temperature:.1f}.pt"
        torch.save({
            'codes': codes,
            'latents': latents,
            'images': images,
            'temperature': temperature,
            'n_samples': n_samples
        }, codes_path)
        print(f"Saved codes, latents, and images to {codes_path}")
    
    print("\nSampling complete!")
    print(f"Results saved to {save_dir if save_images else 'not saved'}")
    return images, codes, latents


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Sample from trained PixelCNN")
    parser.add_argument("--n_samples", type=int, default=16, help="Number of samples to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--no_save", action="store_true", help="Don't save images")
    
    args = parser.parse_args()
    
    launch_sample_pixelcnn(
        n_samples=args.n_samples,
        temperature=args.temperature,
        save_images=not args.no_save
    )
