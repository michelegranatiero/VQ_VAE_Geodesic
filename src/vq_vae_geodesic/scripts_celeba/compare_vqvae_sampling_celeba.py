"""
Compare VQ-VAE sampling methods: PixelCNN vs Random (CelebA)
"""
import torch

from vq_vae_geodesic.config import samples_dir, checkpoint_dir
from vq_vae_geodesic.hyperparameters import get_celeba_config
from vq_vae_geodesic.utils import set_seed
from vq_vae_geodesic.evaluation.sample import sample_pixelcnn
from vq_vae_geodesic.evaluation.utils import codes_to_images_via_vqvae
from vq_vae_geodesic.evaluation.visualize import plot_comparison_2rows

N_SAMPLES = 16
TEMPERATURE = 1.0

def sample_random_codes_vqvae(n_samples, grid_shape, num_embeddings, device):
    """
    Sample random discrete codes uniformly from VQ-VAE embedding space.
    """
    H, W = grid_shape
    codes = torch.randint(0, num_embeddings, (n_samples, H, W), device=device)
    return codes.cpu().numpy()


def compare_vqvae_sampling_methods(n_samples=16, temperature=1.0):
    """
    Compare VQ-VAE sampling with PixelCNN prior vs random uniform sampling (CelebA).
    """
    config = get_celeba_config(img_size=32)
    set_seed(config.seed)
    
    device = torch.device("cuda:0" if config.use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    vqvae_grid_shape = (config.vqvae_params.grid_h, config.vqvae_params.grid_w)
    
    # Load VQ-VAE
    print("\nLoading VQ-VAE...")
    vqvae_checkpoint_path = checkpoint_dir('celeba') / "vqvae_celeba_best.pt"
    vqvae_checkpoint = torch.load(vqvae_checkpoint_path, map_location=device)
    
    from vq_vae_geodesic.models.modules.vqvae import build_vqvae_from_config
    vqvae = build_vqvae_from_config(config.arch_params, config.vqvae_params, dataset="celeba")
    vqvae.load_state_dict(vqvae_checkpoint['model_state_dict'])
    vqvae = vqvae.to(device)
    vqvae.eval()
    
    num_embeddings = vqvae.vq.num_embeddings
    print(f"VQ-VAE codebook size: {num_embeddings} embeddings")
    
    # Load PixelCNN
    print("Loading VQ-VAE PixelCNN...")
    pixelcnn_checkpoint_path = checkpoint_dir('celeba') / "pixelcnn_vqvae_celeba_best.pt"
    pixelcnn_checkpoint = torch.load(pixelcnn_checkpoint_path, map_location=device)
    
    from vq_vae_geodesic.models.modules.pixelCNN import build_pixelcnn_from_config
    pixelcnn_vqvae = build_pixelcnn_from_config(config, for_vqvae=True)
    pixelcnn_vqvae.load_state_dict(pixelcnn_checkpoint['model_state_dict'])
    pixelcnn_vqvae = pixelcnn_vqvae.to(device)
    pixelcnn_vqvae.eval()
    
    print("All models loaded successfully")
    
    # Sample with PixelCNN
    print(f"\nSampling {n_samples} codes with VQ-VAE PixelCNN (T={temperature})...")
    pixelcnn_codes = sample_pixelcnn(
        pixelcnn_vqvae,
        device=device,
        img_size=vqvae_grid_shape,
        temperature=temperature,
        batch_size=n_samples,
        progress=True
    )
    print(f"PixelCNN codes shape: {pixelcnn_codes.shape}")
    
    # Sample randomly
    print(f"\nSampling {n_samples} codes randomly...")
    random_codes = sample_random_codes_vqvae(n_samples, vqvae_grid_shape, num_embeddings, device)
    print(f"Random codes shape: {random_codes.shape}")
    
    # Generate images
    print("\nGenerating images from VQ-VAE PixelCNN codes...")
    pixelcnn_images = codes_to_images_via_vqvae(pixelcnn_codes, vqvae, device)

    print("Generating images from VQ-VAE random codes...")
    random_images = codes_to_images_via_vqvae(random_codes, vqvae, device)
    
    print(f"\nVQ-VAE PixelCNN images shape: {pixelcnn_images.shape}")
    print(f"VQ-VAE Random images shape: {random_images.shape}")
    
    # Save comparison
    save_dir = samples_dir("celeba")
    save_dir.mkdir(exist_ok=True, parents=True)
    
    plot_path = save_dir / "vqvae_pixelcnn_vs_random_celeba.png"
    plot_comparison_2rows(
        top_imgs=pixelcnn_images,
        bottom_imgs=random_images,
        top_label="VQ-VAE + PixelCNN\n(E2E + Prior)",
        bottom_label="VQ-VAE + Random\n(E2E + Uniform)",
        save_path=plot_path,
        title="VQ-VAE CelebA: End-to-End Learning with PixelCNN vs Random Sampling",
        n_show=16
    )
    
    data_path = save_dir / "vqvae_pixelcnn_vs_random_celeba.pt"
    torch.save({
        'pixelcnn_images': pixelcnn_images,
        'random_images': random_images,
        'pixelcnn_codes': pixelcnn_codes,
        'random_codes': random_codes,
        'temperature': temperature
    }, data_path)
    print(f"Saved data to {data_path}")


if __name__ == "__main__":
    
    compare_vqvae_sampling_methods(
        n_samples=N_SAMPLES,
        temperature=TEMPERATURE
    )
