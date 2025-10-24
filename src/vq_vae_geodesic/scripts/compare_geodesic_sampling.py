"""
Compare VAE+Geodesic sampling methods: PixelCNN vs random.
"""
import torch

from vq_vae_geodesic.config import data_dir
from vq_vae_geodesic.hyperparameters import get_mnist_config
from vq_vae_geodesic.utils import set_seed
from vq_vae_geodesic.evaluation.sample import sample_pixelcnn
from vq_vae_geodesic.evaluation.utils import (
    load_pixelcnn_checkpoint,
    load_model_vae_mnist,
    load_codebook_mnist,
    codes_to_images_via_codebook
)
from vq_vae_geodesic.evaluation.visualize import plot_comparison_2rows

N_SAMPLES = 16
TEMPERATURE = 1.0

def sample_random_codes(n_samples, grid_shape, num_tokens, device):
    """
    Sample random discrete codes uniformly from codebook.
    """
    H, W = grid_shape
    # Uniform random sampling from [0, K-1]
    codes = torch.randint(0, num_tokens, (n_samples, H, W), device=device)
    return codes.cpu().numpy()


def compare_pixelcnn_vs_random(n_samples=16, temperature=1.0):
    """
    Compare PixelCNN sampling vs random uniform sampling.
    """
    config = get_mnist_config()
    set_seed(config.seed)
    
    device = torch.device("cuda:0" if config.use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    grid_shape = (config.quant_params.grid_h, config.quant_params.grid_w)
    num_tokens = config.quant_params.n_codewords
    
    # Load models using centralized functions
    print("\nLoading models...")
    pixelcnn = load_pixelcnn_checkpoint("pixelcnn_geodesic_mnist_best.pt", config, device)
    codebook_chunks = load_codebook_mnist(device)
    vae = load_model_vae_mnist(config.arch_params, device)
    print("All models loaded successfully")
    
    # Sample with PixelCNN
    print(f"\nSampling {n_samples} codes with PixelCNN (T={temperature})...")
    pixelcnn_codes = sample_pixelcnn(
        pixelcnn,
        device=device,
        img_size=grid_shape,
        temperature=temperature,
        batch_size=n_samples,
        progress=True
    )
    print(f"PixelCNN codes shape: {pixelcnn_codes.shape}")
    
    # Sample randomly
    print(f"\nSampling {n_samples} codes randomly (uniform)...")
    random_codes = sample_random_codes(n_samples, grid_shape, num_tokens, device)
    print(f"Random codes shape: {random_codes.shape}")
    
    # Generate images using centralized function
    print("\nGenerating images from PixelCNN codes...")
    pixelcnn_images = codes_to_images_via_codebook(pixelcnn_codes, codebook_chunks, vae.decoder, device)

    print("Generating images from random codes...")
    random_images = codes_to_images_via_codebook(random_codes, codebook_chunks, vae.decoder, device)
    
    print(f"\nPixelCNN images shape: {pixelcnn_images.shape}")
    print(f"Random images shape: {random_images.shape}")
    
    # Save comparison
    save_dir = data_dir() / "samples"
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # Save visualization
    plot_path = save_dir / "geodesic_pixelcnn_vs_random.png"
    plot_comparison_2rows(
        top_imgs=pixelcnn_images,
        bottom_imgs=random_images,
        top_label="Geodesic + PixelCNN\n(Post-hoc + Prior)",
        bottom_label="Geodesic + Random\n(Post-hoc + Uniform)",
        save_path=plot_path,
        title="Geodesic Quantization: PixelCNN Prior vs Random Sampling",
        n_show=16
    )
    
    # Save data as torch
    data_path = save_dir / "geodesic_pixelcnn_vs_random.pt"
    torch.save({
        'pixelcnn_images': pixelcnn_images,
        'random_images': random_images,
        'pixelcnn_codes': pixelcnn_codes,
        'random_codes': random_codes,
        'temperature': temperature
    }, data_path)
    print(f"Saved data to {data_path}")


if __name__ == "__main__":
    
    compare_pixelcnn_vs_random(
        n_samples=N_SAMPLES,
        temperature=TEMPERATURE
    )
