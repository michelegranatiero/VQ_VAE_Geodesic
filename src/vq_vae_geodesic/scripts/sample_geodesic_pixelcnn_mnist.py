"""
VAE+Geodesic: sample new discrete codes from trained PixelCNN and generate images.

Uses ancestral sampling to generate new latent code grids from the learned
autoregressive prior, then decodes them to images using the VAE decoder.

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
    lookup_codewords
)
from vq_vae_geodesic.evaluation.visualize import save_generated_samples

N_SAMPLES = 64
TEMPERATURE = 1.0

def launch_sample_pixelcnn(n_samples=16, temperature=1.0):
    config = get_mnist_config()
    set_seed(config.seed)
    
    device = torch.device("cuda:0" if config.use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load PixelCNN
    pixelcnn = load_pixelcnn_checkpoint("pixelcnn_geodesic_mnist_best.pt", config, device)
    
    # Sample codes from PixelCNN... producing indices (n_samples, H, W)
    grid_shape = (config.quant_params.grid_h, config.quant_params.grid_w)
    codes = sample_pixelcnn(
        pixelcnn,
        device=device,
        img_size=grid_shape,
        temperature=temperature,
        batch_size=n_samples,  # number of samples to generate
    ) 
    
    # Load codebook
    codebook_chunks = load_codebook_mnist(device)

    # Convert codes indices to continuous latents using lookup_codewords
    codes_flat = torch.from_numpy(codes.reshape(n_samples, -1)).long().to(device)  # (n_samples, H*W)
    latents = lookup_codewords(codebook_chunks, codes_flat)  # (n_samples, D, <latent_dim>) (tensor)
    print(f"Latents shape: {latents.shape}")
    
    # Load VAE for decoder
    vae = load_model_vae_mnist(config.arch_params, device)
    
    # Decode latents to images
    print("Decoding latents to images...")
    with torch.no_grad():
        images = vae.decoder(latents)
    
    images = images.cpu().numpy()
    print(f"Generated images shape: {images.shape}")
    
    # Save images 
    save_dir = data_dir() / "samples"
    save_dir.mkdir(exist_ok=True, parents=True)
    
    data_path = save_dir / f"geodesic_pixelcnn_samples_t{temperature:.1f}.pt"
    grid_path = save_dir / f"geodesic_pixelcnn_samples_grid_t{temperature:.1f}.png"
    
    save_generated_samples(
        images=images,
        codes=codes,
        latents=latents,
        data_path=data_path,
        grid_path=grid_path,
        grid_title=f"Geodesic PixelCNN Samples (T={temperature})",
        temperature=temperature
    )
    
    print("\nSampling complete!")


if __name__ == "__main__":
    
    launch_sample_pixelcnn(
        n_samples=N_SAMPLES,
        temperature=TEMPERATURE,
    )
