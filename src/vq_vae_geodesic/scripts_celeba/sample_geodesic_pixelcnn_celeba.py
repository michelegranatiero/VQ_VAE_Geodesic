"""
VAE+Geodesic: sample new discrete codes from trained PixelCNN and generate images (CelebA).

Uses ancestral sampling to generate new latent code grids from the learned
autoregressive prior, then decodes them to images using the VAE decoder.

"""
import torch

from vq_vae_geodesic.config import samples_dir, checkpoint_dir, latents_dir
from vq_vae_geodesic.hyperparameters import get_celeba_config
from vq_vae_geodesic.utils import set_seed
from vq_vae_geodesic.evaluation.sample import sample_pixelcnn
from vq_vae_geodesic.evaluation.visualize import save_generated_samples

N_SAMPLES = 64
TEMPERATURE = 1.0

def launch_sample_pixelcnn(n_samples=16, temperature=1.0):
    config = get_celeba_config(img_size=32)
    set_seed(config.seed)
    
    device = torch.device("cuda:0" if config.use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load PixelCNN
    checkpoint_path = checkpoint_dir('celeba') / "pixelcnn_geodesic_celeba_best.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    from vq_vae_geodesic.models.modules.pixelCNN import build_pixelcnn_from_config
    pixelcnn = build_pixelcnn_from_config(config, for_vqvae=False)
    pixelcnn.load_state_dict(checkpoint['model_state_dict'])
    pixelcnn = pixelcnn.to(device)
    pixelcnn.eval()
    
    # Sample codes from PixelCNN
    grid_shape = (config.quant_params.grid_h, config.quant_params.grid_w)
    codes = sample_pixelcnn(
        pixelcnn,
        device=device,
        img_size=grid_shape,
        temperature=temperature,
        batch_size=n_samples,
    ) 
    
    # Load codebook
    codebook_path = latents_dir('celeba') / "chunk_codebook_celeba.pt"
    codebook_data = torch.load(codebook_path, map_location=device)
    codebook_chunks = codebook_data['codebook_chunks']
    
    # Convert codes indices to continuous latents
    from vq_vae_geodesic.evaluation.utils import lookup_codewords
    codes_flat = torch.from_numpy(codes.reshape(n_samples, -1)).long().to(device)
    latents = lookup_codewords(codebook_chunks, codes_flat)
    print(f"Latents shape: {latents.shape}")
    
    # Load VAE decoder
    vae_checkpoint_path = checkpoint_dir('celeba') / "vae_celeba_best.pt"
    vae_checkpoint = torch.load(vae_checkpoint_path, map_location=device)
    
    from vq_vae_geodesic.models.modules.vae import build_vae_from_config
    vae = build_vae_from_config(config.arch_params, dataset="celeba")
    vae.load_state_dict(vae_checkpoint['model_state_dict'])
    vae = vae.to(device)
    vae.eval()
    
    # Decode latents to images
    print("Decoding latents to images...")
    with torch.no_grad():
        images = vae.decoder(latents)
    
    images = images.cpu().numpy()
    print(f"Generated images shape: {images.shape}")
    
    # Save images 
    save_dir = samples_dir("celeba")
    save_dir.mkdir(exist_ok=True, parents=True)
    
    data_path = save_dir / f"geodesic_pixelcnn_samples_celeba_t{temperature:.1f}.pt"
    grid_path = save_dir / f"geodesic_pixelcnn_samples_celeba_grid_t{temperature:.1f}.png"
    
    save_generated_samples(
        images=images,
        codes=codes,
        latents=latents,
        data_path=data_path,
        grid_path=grid_path,
        grid_title=f"Geodesic PixelCNN Samples CelebA (T={temperature})",
        temperature=temperature
    )
    
    print("\nSampling complete!")


if __name__ == "__main__":
    
    launch_sample_pixelcnn(
        n_samples=N_SAMPLES,
        temperature=TEMPERATURE,
    )
