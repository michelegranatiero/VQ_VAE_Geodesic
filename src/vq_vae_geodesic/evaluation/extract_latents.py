"""
Extracts and saves latent representations (mu, logvar) from a trained VAE.
"""
import torch
from tqdm import tqdm


def extract_and_save_latents(model, dataloader, device, save_dir, prefix):
    """
    Extract latent representations from a VAE and save to disk.

    Runs the encoder on all batches in the dataloader and saves
    the resulting mu and logvar tensors as a .pt file (torch.save).

    Args:
        model: VAE model with an encoder
        dataloader: DataLoader for the dataset
        device: Device to run inference on
        save_dir: Directory to save the latents
        prefix: Prefix for the output filename (e.g., 'train', 'val')

    Returns:
        out_path: Path to the saved .pt file
    """
    model.eval()
    mus = []
    logvars = []

    with torch.no_grad():
        for image_batch, _ in tqdm(dataloader, desc="Extracting latents"):
            image_batch = image_batch.to(device)
            mu, logvar = model.encoder(image_batch)
            mus.append(mu.cpu())
            logvars.append(logvar.cpu())

    # Concatenate all batches
    mus = torch.cat(mus, dim=0)
    logvars = torch.cat(logvars, dim=0)

    # Save to disk
    out_path = save_dir / f"{prefix}_latents.pt"
    torch.save({'mu': mus, 'logvar': logvars}, out_path) # Keys 'mu' and 'logvar'

    print(
        f"Latent representations saved to {out_path} (mu shape: {mus.shape}, logvar shape: {logvars.shape})")
    return out_path
