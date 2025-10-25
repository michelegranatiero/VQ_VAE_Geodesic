"""
Compare VQ-VAE sampling methods: PixelCNN vs Random (CIFAR-10)
"""
import torch

from vq_vae_geodesic.config import data_dir, samples_dir
from vq_vae_geodesic.hyperparameters import get_cifar10_config
from vq_vae_geodesic.utils import set_seed
from vq_vae_geodesic.evaluation.sample import sample_pixelcnn
from vq_vae_geodesic.evaluation.utils import (
    load_pixelcnn_checkpoint,
    load_model_vqvae_cifar10,
    codes_to_images_via_vqvae
)
from vq_vae_geodesic.evaluation.visualize import plot_comparison_2rows

N_SAMPLES = 16
TEMPERATURE = 1.0

def sample_random_codes_vqvae(n_samples, grid_shape, num_embeddings, device):
    """
    Sample random discrete codes uniformly from VQ-VAE embedding space.
    """
    H, W = grid_shape
    # Uniform random sampling from [0, num_embeddings-1]
    codes = torch.randint(0, num_embeddings, (n_samples, H, W), device=device)
    return codes.cpu().numpy()


def compare_vqvae_sampling_methods(n_samples=16, temperature=1.0):
    """
    Compare VQ-VAE sampling with PixelCNN prior vs random uniform sampling.
    """
    config = get_cifar10_config()
    set_seed(config.seed)
    
    device = torch.device("cuda:0" if config.use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # VQ-VAE uses 8x8 spatial latent grid for CIFAR-10
    vqvae_grid_shape = (config.vqvae_params.grid_h, config.vqvae_params.grid_w)
    
    # Load models using centralized functions
    print("\nLoading models...")
    vqvae = load_model_vqvae_cifar10(config.arch_params, config.vqvae_params, device)
    num_embeddings = vqvae.vq.num_embeddings
    print(f"VQ-VAE codebook size: {num_embeddings} embeddings")
    
    # Load PixelCNN trained on VQ-VAE codes
    pixelcnn_vqvae = load_pixelcnn_checkpoint("pixelcnn_vqvae_cifar10_best.pt", config, device)
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
    print(f"\nSampling {n_samples} codes randomly (uniform from VQ-VAE codebook)...")
    random_codes = sample_random_codes_vqvae(n_samples, vqvae_grid_shape, num_embeddings, device)
    print(f"Random codes shape: {random_codes.shape}")
    
    # Generate images using centralized function
    print("\nGenerating images from VQ-VAE PixelCNN codes...")
    pixelcnn_images = codes_to_images_via_vqvae(pixelcnn_codes, vqvae, device)

    print("Generating images from VQ-VAE random codes...")
    random_images = codes_to_images_via_vqvae(random_codes, vqvae, device)
    
    print(f"\nVQ-VAE PixelCNN images shape: {pixelcnn_images.shape}")
    print(f"VQ-VAE Random images shape: {random_images.shape}")
    
    # Save comparison
    save_dir = samples_dir("cifar10")
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # Save visualization
    plot_path = save_dir / "vqvae_pixelcnn_vs_random_cifar10.png"
    plot_comparison_2rows(
        top_imgs=pixelcnn_images,
        bottom_imgs=random_images,
        top_label="VQ-VAE + PixelCNN\n(E2E + Prior)",
        bottom_label="VQ-VAE + Random\n(E2E + Uniform)",
        save_path=plot_path,
        title="VQ-VAE CIFAR-10: End-to-End Learning with PixelCNN vs Random Sampling",
        n_show=16
    )
    
    # Save data as torch
    data_path = save_dir / "vqvae_pixelcnn_vs_random_cifar10.pt"
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
