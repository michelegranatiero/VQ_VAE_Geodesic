"""
Extract discrete codes from a trained VQ-VAE on MNIST and save to latents directory.
"""
import torch
from vq_vae_geodesic.config import latents_dir
from vq_vae_geodesic.hyperparameters import get_mnist_config
from vq_vae_geodesic.utils import set_seed
from vq_vae_geodesic.data.loaders import get_MNIST_loaders
from vq_vae_geodesic.evaluation.utils import load_model_vqvae_mnist, extract_vqvae_codes

def launch_extraction():
    config = get_mnist_config()
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load VQ-VAE
    vqvae = load_model_vqvae_mnist(config.arch_params, config.vqvae_params, device)

    # Get MNIST loaders (train, val, test)
    train_loader, val_loader, test_loader = get_MNIST_loaders(batch_size=config.training_params.batch_size, shuffle_train_set=False)

    # Extract codes
    train_codes = extract_vqvae_codes(vqvae, train_loader, device)
    val_codes = extract_vqvae_codes(vqvae, val_loader, device)
    test_codes = extract_vqvae_codes(vqvae, test_loader, device)

    out_dir = latents_dir('mnist')
    # Save a single assigned-like file with train/val/test keys so CodesDataset can load it
    out_path = out_dir / 'vqvae_assigned_codes.pt'
    train_t = torch.from_numpy(train_codes).long()
    val_t = torch.from_numpy(val_codes).long()
    test_t = torch.from_numpy(test_codes).long()
    torch.save({'train_codes': train_t, 'val_codes': val_t, 'test_codes': test_t}, out_path)
    print('\nExtraction complete. Saved vqvae assigned codes to', out_path)


if __name__ == '__main__':
    launch_extraction()
