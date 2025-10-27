"""
Compare VAE+Geodesic PixelCNN sampling at different temperatures (CelebA).
"""
import torch
from vq_vae_geodesic.config import samples_dir, checkpoint_dir, latents_dir
from vq_vae_geodesic.hyperparameters import get_celeba_config
from vq_vae_geodesic.utils import set_seed
from vq_vae_geodesic.evaluation.sample import sample_pixelcnn
from vq_vae_geodesic.evaluation.utils import codes_to_images_via_codebook
from vq_vae_geodesic.evaluation.visualize import plot_temperature_comparison

TEMPERATURES = [0.5, 1.0, 1.5, 2.0, 5.0]
N_SAMPLES_PER_TEMP = 16


def compare_temperatures(temperatures=[0.5, 0.8, 1.0, 1.2, 1.5], n_samples_per_temp=8):
    """
    Generate samples at different temperatures and create comparison plot.
    
    Args:
        temperatures: List of temperature values to compare
        n_samples_per_temp: Number of samples to generate per temperature
    """
    config = get_celeba_config(img_size=32)
    set_seed(config.seed)
    
    device = torch.device("cuda:0" if config.use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
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
    
    # Sample at each temperature
    grid_shape = (config.quant_params.grid_h, config.quant_params.grid_w)
    all_samples = []
    
    for temp in temperatures:
        print(f"\nSampling at temperature {temp}...")

        # Sample codes
        codes = sample_pixelcnn(
            pixelcnn,
            device=device,
            img_size=grid_shape,
            temperature=temp,
            batch_size=n_samples_per_temp,
            progress=False
        )

        # Convert to images
        images = codes_to_images_via_codebook(codes, codebook_chunks, vae.decoder, device)
        all_samples.append(images)
    
    # Save comparison
    save_dir = samples_dir("celeba")
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # Save visualization
    plot_path = save_dir / "geodesic_temperature_comparison_celeba.png"
    plot_temperature_comparison(
        samples_by_temp=all_samples,
        temperatures=temperatures,
        save_path=plot_path,
        n_show=n_samples_per_temp
    )
    
    # Save data as torch
    data_path = save_dir / "geodesic_temperature_comparison_celeba.pt"
    torch.save({
        'temperatures': temperatures,
        'samples': all_samples,
        'n_samples_per_temp': n_samples_per_temp
    }, data_path)
    print(f"Saved comparison data to {data_path}")
    
    print("\nTemperature effects:")
    print("T < 1.0: Sharper, less diverse (model chooses the most probable codes)")
    print("T = 1.0: Balanced sampling from learned distribution")
    print("T > 1.0: More diverse, potentially noisier (exploratory)")


if __name__ == "__main__":
    
    compare_temperatures(
        temperatures=TEMPERATURES,
        n_samples_per_temp=N_SAMPLES_PER_TEMP
    )
