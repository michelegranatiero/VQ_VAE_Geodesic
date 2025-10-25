import torch
import matplotlib.pyplot as plt
from vq_vae_geodesic.evaluation.utils import lookup_codewords
from vq_vae_geodesic.evaluation.visualize import prepare_image_for_display

def get_vae_reconstructions(model, data_loader, device, n_show=8):
    """
    Restituisce n_show immagini originali e le relative ricostruzioni dal primo batch (VAE).
    """
    model.eval()
    data_iter = iter(data_loader)
    images, _ = next(data_iter)
    images = images.to(device)
    with torch.no_grad():
        recon, _, _ = model(images)
    orig_imgs = images[:n_show].cpu()
    recon_imgs = recon[:n_show].cpu()
    return orig_imgs, recon_imgs


def get_geodesic_reconstructions(model, data_loader, codebook_chunks: torch.Tensor, codes_per_image: torch.Tensor, device, n_show=8):
    """
    Restituisce n_show immagini originali e le relative ricostruzioni geodesic dal primo batch.
    """
    model.eval()
    data_iter = iter(data_loader)
    images, _ = next(data_iter)
    images = images.to(device)
    codebook_chunks = codebook_chunks.to(device)
    codes_per_image = codes_per_image.to(device)
    batch_codes = codes_per_image[:images.size(0)]
    with torch.no_grad():
        z_recon = lookup_codewords(codebook_chunks, batch_codes)
        recon = model.decoder(z_recon)
    orig_imgs = images[:n_show].cpu()
    recon_imgs = recon[:n_show].cpu()
    return orig_imgs, recon_imgs


def get_vqvae_reconstructions(model, data_loader, device, n_show=8):
    """
    Restituisce n_show immagini originali e le relative ricostruzioni dal primo batch (VQ-VAE).
    """
    model.eval()
    data_iter = iter(data_loader)
    images, _ = next(data_iter)
    images = images.to(device)
    with torch.no_grad():
        recon, _, _ = model(images)  # VQ-VAE returns (recon, vq_loss, codes)
    orig_imgs = images[:n_show].cpu()
    recon_imgs = recon[:n_show].cpu()
    return orig_imgs, recon_imgs



def plot_reconstructions_comparison(orig_imgs, vae_recon, geodesic_recon, vqvae_recon, save_path, n_show=8):
    """
    Create and save a comparison plot of original images and reconstructions from three models.
    Handles both grayscale (1, H, W) and RGB (3, H, W) images.
    """
    fig, axs = plt.subplots(4, n_show, figsize=(2*n_show, 8))
    
    for i in range(n_show):
        # Original
        ax = axs[0, i]
        ax.imshow(prepare_image_for_display(orig_imgs[i]), cmap='gray' if orig_imgs[i].shape[0] == 1 else None)
        ax.axis('off')
        if i == 0:
            ax.set_title('Original')
        # VAE
        ax = axs[1, i]
        ax.imshow(prepare_image_for_display(vae_recon[i]), cmap='gray' if vae_recon[i].shape[0] == 1 else None)
        ax.axis('off')
        if i == 0:
            ax.set_title('VAE')
        # Geodesic
        ax = axs[2, i]
        ax.imshow(prepare_image_for_display(geodesic_recon[i]), cmap='gray' if geodesic_recon[i].shape[0] == 1 else None)
        ax.axis('off')
        if i == 0:
            ax.set_title('VAE+Geodesic')
        # VQ-VAE
        ax = axs[3, i]
        ax.imshow(prepare_image_for_display(vqvae_recon[i]), cmap='gray' if vqvae_recon[i].shape[0] == 1 else None)
        ax.axis('off')
        if i == 0:
            ax.set_title('VQ-VAE')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Reconstruction comparison saved to {save_path}")