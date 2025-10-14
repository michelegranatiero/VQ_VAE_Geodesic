
"""
Reconstruction evaluation utilities.

Functions to reconstruct images from latent codes or codebooks
and visualize/evaluate the reconstruction quality.
"""
import torch
import matplotlib.pyplot as plt
from pathlib import Path 
from tqdm import tqdm

def save_reconstruction_plot(orig_imgs, vae_imgs, geodesic_imgs, out_path, title_orig="Original", title_vae="VAE", title_geodesic="Geodesic"):
    """
    Save a reconstruction comparison plot with original, VAE, and geodesic reconstructions.

    Args:
        orig_imgs: Original images (N, C, H, W) as tensor
        vae_imgs: VAE reconstructions (N, C, H, W) as tensor
        geodesic_imgs: Geodesic reconstructions (N, C, H, W) as tensor
        out_path: Path to save the plot (str or Path)
        title_orig: Title for original images row
        title_vae: Title for VAE reconstructions row
        title_geodesic: Title for geodesic reconstructions row
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_show = min(8, orig_imgs.shape[0], vae_imgs.shape[0], geodesic_imgs.shape[0])
    orig = orig_imgs[:n_show]
    vae = vae_imgs[:n_show]
    geodesic = geodesic_imgs[:n_show]
    fig, axs = plt.subplots(3, n_show, figsize=(2*n_show, 6))
    for i in range(n_show):
        # Original
        ax = axs[0, i]
        ax.imshow(orig[i].squeeze().cpu().numpy(), cmap='gray')
        ax.axis('off')
        if i == 0:
            ax.set_title(title_orig)
        # VAE
        ax = axs[1, i]
        ax.imshow(vae[i].squeeze().cpu().numpy(), cmap='gray')
        ax.axis('off')
        if i == 0:
            ax.set_title(title_vae)
        # Geodesic
        ax = axs[2, i]
        ax.imshow(geodesic[i].squeeze().cpu().numpy(), cmap='gray')
        ax.axis('off')
        if i == 0:
            ax.set_title(title_geodesic)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)

def evaluate_vae_reconstruction(model, data_loader, device, n_show=8):
    """
    Evaluate VAE reconstruction as trained (with sampling).

    Uses the full VAE forward pass: x → encoder → sample z → decoder → recon.
    This represents the actual VAE behavior during inference.

    Args:
        vae: VAE model (encoder + decoder)
        data_loader: DataLoader for the dataset
        device: Device for computation
        n_show: Number of images to collect for visualization

    Returns:
        mean_mse: Average MSE across all images
        orig_imgs: Original images (n_show, C, H, W) as numpy array
        recon_imgs: Reconstructed images (n_show, C, H, W) as numpy array
    """
    model.eval()
    all_mse = []
    orig_imgs = []
    recon_imgs = []

    for image_batch, _ in tqdm(data_loader, desc="Evaluating VAE"):
        image_batch = image_batch.to(device)

        # Full VAE forward pass (as trained, with sampling)
        with torch.no_grad():
            recon, mu, logvar = model(image_batch)

        # Compute MSE
        batch_mse = torch.mean((recon - image_batch)**2).item()
        all_mse.append(batch_mse)

        # Collect images for visualization
        if len(orig_imgs) < n_show:
            to_take = min(n_show - len(orig_imgs), image_batch.size(0))
            orig_imgs.append(image_batch[:to_take].cpu())
            recon_imgs.append(recon[:to_take].cpu())

    # Concatenate collected images
    orig_imgs = torch.cat(orig_imgs, dim=0)
    recon_imgs = torch.cat(recon_imgs, dim=0)
    
    return torch.mean(torch.tensor(all_mse)), orig_imgs, recon_imgs


def evaluate_geodesic_reconstruction(model, data_loader, codebook_chunks: torch.Tensor, codes_per_image: torch.Tensor, device, n_show=8):
    """
    Evaluate reconstruction using geodesic quantization.

    Each latent vector is split into L chunks, and each chunk is
    independently quantized using the geodesic codebook. The quantized
    chunks are concatenated and decoded to reconstruct the image.

    Args:
        vae: VAE model with decoder
        data_loader: DataLoader (must be in same order as codes_per_image)
        codebook_chunks: Chunk codebook (K, chunk_size)
        codes_per_image: Assigned chunk codes (N, L) with indices 0..K-1
        device: Device for computation
        n_show: Number of images to collect for visualization

    Returns:
        mean_mse: Average MSE across all images
        orig_imgs: Original images (n_show, C, H, W) as numpy array
        recon_imgs: Reconstructed images (n_show, C, H, W) as numpy array
    """
    model.eval()
    all_mse = []
    ptr = 0
    orig_imgs = []
    recon_imgs = []

    # Ensure torch tensors
    codebook_chunks = codebook_chunks.to(device)
    codes_per_image = codes_per_image.to(device)


    for image_batch, _ in tqdm(data_loader, desc="Evaluating Geodesic"):
        bs = image_batch.size(0)
        image_batch = image_batch.to(device)

        # Get codes indices for this batch: (bs, L)
        batch_codes = codes_per_image[ptr:ptr+bs]
        ptr += bs

        # Get actual codewords (chunks): (bs, L, chunk_size)
        z_chunks = codebook_chunks[batch_codes]  # indexing

        # Reshape to full latent vectors: (bs, D) where D = L * chunk_size
        z_recon = z_chunks.reshape(bs, -1)

        # Decode quantized latents
        with torch.no_grad():
            recon = model.decoder(z_recon)

        # Compute MSE
        batch_mse = torch.mean((recon - image_batch)**2).item()
        all_mse.append(batch_mse)

        # Collect images for visualization
        if len(orig_imgs) < n_show:
            to_take = min(n_show - len(orig_imgs), bs)
            orig_imgs.append(image_batch[:to_take].cpu())
            recon_imgs.append(recon[:to_take].cpu())

    # Concatenate collected images
    if len(orig_imgs) > 0:
        orig_imgs = torch.cat(orig_imgs, dim=0)
        recon_imgs = torch.cat(recon_imgs, dim=0)
    else:
        orig_imgs = torch.empty(0)
        recon_imgs = torch.empty(0)
    
    return torch.mean(torch.tensor(all_mse)), orig_imgs, recon_imgs
