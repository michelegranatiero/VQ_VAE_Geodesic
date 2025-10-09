
"""
Reconstruction evaluation utilities.

Functions to reconstruct images from latent codes or codebooks
and visualize/evaluate the reconstruction quality.
"""
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import numpy as np


def plot_recon_grid(orig, recon, out_path, n=8, title_orig="Original", title_recon="Reconstr."):
    """
    Plot a 2xN grid comparing original and reconstructed images.

    Args:
        orig: Original images (N, C, H, W) or (N, H, W)
        recon: Reconstructed images (N, C, H, W) or (N, H, W)
        out_path: Path to save the plot
        n: Number of images to show
        title_orig: Title for original images row
        title_recon: Title for reconstructed images row
    """
    orig = orig[:n]
    recon = recon[:n]
    fig, axs = plt.subplots(2, n, figsize=(2*n, 4))
    for i in range(n):
        # Original
        ax = axs[0, i]
        ax.imshow(orig[i].squeeze(), cmap='gray')
        ax.axis('off')
        if i == 0:
            ax.set_title(title_orig)
        # Recon
        ax = axs[1, i]
        ax.imshow(recon[i].squeeze(), cmap='gray')
        ax.axis('off')
        if i == 0:
            ax.set_title(title_recon)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def reconstruct_with_codebook(vae, data_loader, codebook, assigned_idx_global, device, out_dir="data/recons"):
    """
    Reconstruct images using a full-dimensional codebook.

    Uses pre-assigned codebook indices to replace latent vectors
    with their quantized versions, then decodes to reconstruct images.

    Args:
        vae: VAE model with decoder
        data_loader: DataLoader (must be in same order as assigned indices)
        codebook: Codebook vectors (K, D)
        assigned_idx_global: Pre-assigned codebook indices (N,)
        device: Device for computation
        out_dir: Directory to save reconstruction grid

    Returns:
        mean_mse: Average MSE across all images
    """
    vae.eval()
    os.makedirs(out_dir, exist_ok=True)
    all_mse = []
    ptr = 0
    codebook_t = torch.from_numpy(codebook).float().to(device)
    orig_imgs = []
    recon_imgs = []
    n_show = 8

    for x, _ in tqdm(data_loader):
        bs = x.size(0)
        x = x.to(device)

        # Get assigned codebook indices for this batch
        assigned_batch = assigned_idx_global[ptr:ptr+bs]
        ptr += bs

        # Lookup quantized latent vectors
        z_replacements = codebook_t[assigned_batch]  # (bs, D)

        # Decode quantized latents
        with torch.no_grad():
            recon = vae.decoder(z_replacements)

        # Compute MSE
        batch_mse = torch.mean((recon - x)**2).item()
        all_mse.append(batch_mse)

        # Collect images for visualization
        if len(orig_imgs) < n_show:
            to_take = min(n_show - len(orig_imgs), bs)
            orig_imgs.append(x[:to_take].cpu())
            recon_imgs.append(recon[:to_take].cpu())

    # Plot reconstruction grid
    orig_imgs = torch.cat(orig_imgs, dim=0).numpy()
    recon_imgs = torch.cat(recon_imgs, dim=0).numpy()
    plot_recon_grid(orig_imgs, recon_imgs, os.path.join(out_dir, "recon_grid_codebook.png"),
                    n=n_show, title_orig="Original", title_recon="Recon codebook")
    return np.mean(all_mse)


def recon_from_mu(vae, data_loader, device, out_dir="data/recons"):
    """
    Reconstruct images from continuous latent means (baseline).

    Encodes images to get mu (latent mean), then decodes directly
    without any quantization. This serves as a baseline to compare
    against codebook-based reconstruction.

    Args:
        vae: VAE model with encoder and decoder
        data_loader: DataLoader for the dataset
        device: Device for computation
        out_dir: Directory to save reconstruction grid

    Returns:
        mean_mse: Average MSE across all images
    """
    os.makedirs(out_dir, exist_ok=True)
    all_mse = []
    orig_imgs = []
    recon_imgs = []
    n_show = 8

    for x, _ in tqdm(data_loader):
        bs = x.size(0)
        x = x.to(device)

        # Encode and decode with continuous latents
        with torch.no_grad():
            mu, logvar = vae.encoder(x)
            recon = vae.decoder(mu)

        # Compute MSE
        batch_mse = torch.mean((recon - x)**2).item()
        all_mse.append(batch_mse)

        # Collect images for visualization
        if len(orig_imgs) < n_show:
            to_take = min(n_show - len(orig_imgs), bs)
            orig_imgs.append(x[:to_take].cpu())
            recon_imgs.append(recon[:to_take].cpu())

    # Plot reconstruction grid
    orig_imgs = torch.cat(orig_imgs, dim=0).numpy()
    recon_imgs = torch.cat(recon_imgs, dim=0).numpy()
    plot_recon_grid(orig_imgs, recon_imgs, os.path.join(out_dir, "recon_grid_mu.png"),
                    n=n_show, title_orig="Original", title_recon="Recon mu")
    return np.mean(all_mse)


def reconstruct_from_chunk_codebook(vae, data_loader, codebook_chunks, codes_per_image, device, out_dir="data/recons"):
    """
    Reconstruct images using a chunked codebook.

    Each latent vector is split into L chunks, and each chunk is
    independently quantized. The quantized chunks are concatenated
    and decoded to reconstruct the image.

    Args:
        vae: VAE model with decoder
        data_loader: DataLoader (must be in same order as codes_per_image)
        codebook_chunks: Chunk codebook (K, chunk_size)
        codes_per_image: Assigned chunk codes (N, L) with indices 0..K-1
        device: Device for computation
        out_dir: Directory to save reconstruction grid

    Returns:
        mean_mse: Average MSE across all images
    """
    vae.eval()
    os.makedirs(out_dir, exist_ok=True)
    all_mse = []
    ptr = 0
    orig_imgs = []
    recon_imgs = []
    n_show = 8

    # Ensure numpy arrays
    codebook_chunks = np.asarray(codebook_chunks)
    codes_per_image = np.asarray(codes_per_image)

    for x, _ in tqdm(data_loader):
        bs = x.size(0)
        x = x.to(device)

        # Get codes for this batch: (bs, L)
        batch_codes = codes_per_image[ptr:ptr+bs]
        ptr += bs

        # Lookup chunk centroids: (bs, L, chunk_size)
        z_chunks = codebook_chunks[batch_codes]  # NumPy advanced indexing

        # Reshape to full latent vectors: (bs, D) where D = L * chunk_size
        z_recon = z_chunks.reshape(bs, -1)
        z_recon_t = torch.from_numpy(z_recon).float().to(device)

        # Decode quantized latents
        with torch.no_grad():
            recon = vae.decoder(z_recon_t)

        # Compute MSE
        batch_mse = torch.mean((recon - x)**2).item()
        all_mse.append(batch_mse)

        # Collect images for visualization
        if len(orig_imgs) < n_show:
            to_take = min(n_show - len(orig_imgs), bs)
            orig_imgs.append(x[:to_take].cpu())
            recon_imgs.append(recon[:to_take].cpu())

    # Plot reconstruction grid
    if len(orig_imgs) > 0:
        orig_imgs = torch.cat(orig_imgs, dim=0).numpy()
        recon_imgs = torch.cat(recon_imgs, dim=0).numpy()
        plot_recon_grid(orig_imgs, recon_imgs, os.path.join(out_dir, "recon_grid_chunk_codebook.png"),
                        n=min(n_show, orig_imgs.shape[0]),
                        title_orig="Original", title_recon="Recon chunk-codebook")
    return np.mean(all_mse)
