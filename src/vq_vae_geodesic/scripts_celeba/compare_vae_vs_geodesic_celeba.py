"""
Compare sampling between:
1. VAE pure (sampling from continuous latent space)
2. VAE + Geodesic quantization (sampling with PixelCNN on discrete codebook)

This shows the effect of discretization on sample quality (CelebA).
"""
import torch

from vq_vae_geodesic.config import samples_dir, checkpoint_dir, latents_dir
from vq_vae_geodesic.hyperparameters import get_celeba_config
from vq_vae_geodesic.utils import set_seed
from vq_vae_geodesic.evaluation.sample import sample_pixelcnn
from vq_vae_geodesic.evaluation.utils import codes_to_images_via_codebook
from vq_vae_geodesic.evaluation.visualize import plot_comparison_2rows

N_SAMPLES = 16
TEMPERATURE = 1.0


def sample_vae_pure(vae, n_samples, latent_dim, device):
    """
    Sample from VAE by sampling from standard normal in latent space.
    """
    vae.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, latent_dim).to(device)
        samples = vae.decoder(z)
        samples = torch.clamp(samples, 0, 1)
    
    return samples.cpu().numpy()


def compare_vae_vs_geodesic(n_samples=16, temperature=1.0):
    """
    Compare sampling quality between VAE pure and VAE+Geodesic (CelebA).
    """
    config = get_celeba_config(img_size=32)
    set_seed(config.seed)
    
    device = torch.device('cuda' if config.use_gpu and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Configuration
    arch_params = config.arch_params
    latent_dim = arch_params.latent_dim
    grid_shape = (config.quant_params.grid_h, config.quant_params.grid_w)
    
    print("Loading models...")
    # Load VAE
    vae_checkpoint_path = checkpoint_dir('celeba') / "vae_celeba_best.pt"
    vae_checkpoint = torch.load(vae_checkpoint_path, map_location=device)
    
    from vq_vae_geodesic.models.modules.vae import build_vae_from_config
    vae = build_vae_from_config(arch_params, dataset="celeba")
    vae.load_state_dict(vae_checkpoint['model_state_dict'])
    vae = vae.to(device)
    vae.eval()
    
    # Load PixelCNN
    pixelcnn_checkpoint_path = checkpoint_dir('celeba') / "pixelcnn_geodesic_celeba_best.pt"
    pixelcnn_checkpoint = torch.load(pixelcnn_checkpoint_path, map_location=device)
    
    from vq_vae_geodesic.models.modules.pixelCNN import build_pixelcnn_from_config
    pixelcnn = build_pixelcnn_from_config(config, for_vqvae=False)
    pixelcnn.load_state_dict(pixelcnn_checkpoint['model_state_dict'])
    pixelcnn = pixelcnn.to(device)
    pixelcnn.eval()
    
    # Load codebook
    codebook_path = latents_dir('celeba') / "chunk_codebook_celeba.pt"
    codebook_data = torch.load(codebook_path, map_location=device)
    codebook_chunks = codebook_data['codebook_chunks']
    
    print("All models loaded successfully\n")
    
    # Sample from VAE pure
    print(f"Sampling {n_samples} images from VAE (continuous latent space)...")
    vae_samples = sample_vae_pure(vae, n_samples, latent_dim, device)
    print(f"VAE samples shape: {vae_samples.shape}\n")
    
    # Sample from VAE+Geodesic
    print(f"Sampling {n_samples} images from VAE+Geodesic (PixelCNN, T={temperature})...")
    geodesic_codes = sample_pixelcnn(
        pixelcnn,
        device=device,
        img_size=grid_shape,
        temperature=temperature,
        batch_size=n_samples,
        progress=True
    )
    geodesic_samples = codes_to_images_via_codebook(
        geodesic_codes, 
        codebook_chunks, 
        vae.decoder, 
        device
    )
    print(f"Geodesic samples shape: {geodesic_samples.shape}\n")
    
    # Create comparison visualization
    save_path = samples_dir('celeba') / "vae_vs_geodesic_celeba.png"
    
    plot_comparison_2rows(
        top_imgs=vae_samples,
        bottom_imgs=geodesic_samples,
        top_label="VAE Pure",
        bottom_label="VAE+Geodesic",
        save_path=save_path,
        title="VAE Pure vs VAE+Geodesic Quantization (CelebA)",
        n_show=min(16, n_samples)
    )
    
    # Save data
    data_path = save_path.parent / save_path.name.replace('.png', '.pt')
    torch.save({
        'vae_samples': vae_samples,
        'geodesic_samples': geodesic_samples,
        'temperature': temperature,
        'n_samples': n_samples
    }, data_path)
    print(f"Saved comparison data to {data_path}")


if __name__ == "__main__":
    compare_vae_vs_geodesic(
        n_samples=N_SAMPLES,
        temperature=TEMPERATURE
    )
