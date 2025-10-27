"""
Compare VAE+Geodesic sampling methods: PixelCNN vs random (CelebA).
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

def sample_random_codes(n_samples, grid_shape, num_tokens, device):
    """
    Sample random discrete codes uniformly from codebook.
    """
    H, W = grid_shape
    codes = torch.randint(0, num_tokens, (n_samples, H, W), device=device)
    return codes.cpu().numpy()


def compare_pixelcnn_vs_random(n_samples=16, temperature=1.0):
    """
    Compare PixelCNN sampling vs random uniform sampling.
    """
    config = get_celeba_config(img_size=32)
    set_seed(config.seed)
    
    device = torch.device("cuda:0" if config.use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    grid_shape = (config.quant_params.grid_h, config.quant_params.grid_w)
    num_tokens = config.quant_params.n_codewords
    
    # Load PixelCNN
    print("\nLoading PixelCNN...")
    pixelcnn_checkpoint_path = checkpoint_dir('celeba') / "pixelcnn_geodesic_celeba_best.pt"
    pixelcnn_checkpoint = torch.load(pixelcnn_checkpoint_path, map_location=device)
    
    from vq_vae_geodesic.models.modules.pixelCNN import build_pixelcnn_from_config
    pixelcnn = build_pixelcnn_from_config(config, for_vqvae=False)
    pixelcnn.load_state_dict(pixelcnn_checkpoint['model_state_dict'])
    pixelcnn = pixelcnn.to(device)
    pixelcnn.eval()
    
    # Load codebook
    print("Loading codebook...")
    codebook_path = latents_dir('celeba') / "chunk_codebook_celeba.pt"
    codebook_data = torch.load(codebook_path, map_location=device)
    codebook_chunks = codebook_data['codebook_chunks']
    
    # Load VAE decoder
    print("Loading VAE...")
    vae_checkpoint_path = checkpoint_dir('celeba') / "vae_celeba_best.pt"
    vae_checkpoint = torch.load(vae_checkpoint_path, map_location=device)
    
    from vq_vae_geodesic.models.modules.vae import build_vae_from_config
    vae = build_vae_from_config(config.arch_params, dataset="celeba")
    vae.load_state_dict(vae_checkpoint['model_state_dict'])
    vae = vae.to(device)
    vae.eval()
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
    
    # Generate images
    print("\nGenerating images from PixelCNN codes...")
    pixelcnn_images = codes_to_images_via_codebook(pixelcnn_codes, codebook_chunks, vae.decoder, device)

    print("Generating images from random codes...")
    random_images = codes_to_images_via_codebook(random_codes, codebook_chunks, vae.decoder, device)
    
    print(f"\nPixelCNN images shape: {pixelcnn_images.shape}")
    print(f"\nRandom images shape: {random_images.shape}")
    
    # Save comparison
    save_dir = samples_dir('celeba')
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # Save visualization
    plot_path = save_dir / "geodesic_pixelcnn_vs_random_celeba.png"
    plot_comparison_2rows(
        top_imgs=pixelcnn_images,
        bottom_imgs=random_images,
        top_label="Geodesic + PixelCNN\n(Post-hoc + Prior)",
        bottom_label="Geodesic + Random\n(Post-hoc + Uniform)",
        save_path=plot_path,
        title="Geodesic Quantization CelebA: PixelCNN Prior vs Random Sampling",
        n_show=16
    )
    
    # Save data as torch
    data_path = save_dir / "geodesic_pixelcnn_vs_random_celeba.pt"
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
