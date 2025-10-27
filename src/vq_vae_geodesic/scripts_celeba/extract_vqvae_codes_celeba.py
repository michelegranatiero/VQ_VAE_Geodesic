"""
Extract discrete codes from a trained VQ-VAE on CelebA and save to latents directory.
"""
import torch
from vq_vae_geodesic.config import latents_dir, checkpoint_dir
from vq_vae_geodesic.hyperparameters import get_celeba_config
from vq_vae_geodesic.utils import set_seed
from vq_vae_geodesic.data.loaders import get_celeba_loaders
from vq_vae_geodesic.evaluation.utils import extract_vqvae_codes

def launch_extraction():
    config = get_celeba_config(img_size=32)
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load VQ-VAE
    print(f"Using device: {device}")
    print("Loading VQ-VAE model...")
    
    checkpoint_path = checkpoint_dir('celeba') / "vqvae_celeba_best.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    from vq_vae_geodesic.models.modules.vqvae import build_vqvae_from_config
    vqvae = build_vqvae_from_config(config.arch_params, config.vqvae_params, dataset="celeba")
    vqvae.load_state_dict(checkpoint['model_state_dict'])
    vqvae = vqvae.to(device)
    vqvae.eval()

    # Get CelebA loaders (train, val, test) - NOT shuffled for consistent extraction
    # Replace with get_celeba_loaders
    print("Loading data...")
    train_loader, val_loader, test_loader = get_celeba_loaders(
        img_size=32,
        batch_size=config.data_params.batch_size,
        shuffle_train_set=False
    )

    # Extract codes
    print("\n=== Extracting VQ-VAE codes ===")
    print("Extracting train codes...")
    train_codes = extract_vqvae_codes(vqvae, train_loader, device)
    print(f"Train codes: {train_codes.shape}")
    
    print("Extracting val codes...")
    val_codes = extract_vqvae_codes(vqvae, val_loader, device)
    print(f"Val codes: {val_codes.shape}")
    
    print("Extracting test codes...")
    test_codes = extract_vqvae_codes(vqvae, test_loader, device)
    print(f"Test codes: {test_codes.shape}")

    out_dir = latents_dir('celeba')
    # Save a single assigned-like file with train/val/test keys
    out_path = out_dir / 'vqvae_assigned_codes_celeba.pt'
    train_t = torch.from_numpy(train_codes).long()
    val_t = torch.from_numpy(val_codes).long()
    test_t = torch.from_numpy(test_codes).long()
    torch.save({'train_codes': train_t, 'val_codes': val_t, 'test_codes': test_t}, out_path)
    print(f'\n✅ Extraction complete. Saved VQ-VAE codes to {out_path}')
    print(f"Train: {train_t.shape}, Val: {val_t.shape}, Test: {test_t.shape}")


if __name__ == '__main__':
    launch_extraction()
