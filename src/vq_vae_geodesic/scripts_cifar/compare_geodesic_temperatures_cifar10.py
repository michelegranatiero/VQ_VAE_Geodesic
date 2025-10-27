"""
Compare VAE+Geodesic PixelCNN sampling at different temperatures (CIFAR-10).
"""
import torch
from vq_vae_geodesic.config import data_dir, samples_dir
from vq_vae_geodesic.hyperparameters import get_cifar10_config
from vq_vae_geodesic.utils import set_seed
from vq_vae_geodesic.evaluation.sample import sample_pixelcnn
from vq_vae_geodesic.evaluation.utils import (
    load_pixelcnn_checkpoint,
    load_model_vae_cifar10,
    load_codebook_cifar10,
    codes_to_images_via_codebook
)
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
    config = get_cifar10_config()
    set_seed(config.seed)
    
    device = torch.device("cuda:0" if config.use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load models using centralized functions
    print("\nLoading models...")
    pixelcnn = load_pixelcnn_checkpoint("pixelcnn_geodesic_cifar10_best.pt", config, device)
    codebook_chunks = load_codebook_cifar10(device)
    vae = load_model_vae_cifar10(config.arch_params, device)
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

        # Convert to images using centralized function
        images = codes_to_images_via_codebook(codes, codebook_chunks, vae.decoder, device)
        all_samples.append(images)
    
    # Save comparison
    save_dir = samples_dir("cifar10")
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # Save visualization
    plot_path = save_dir / "geodesic_temperature_comparison_cifar10.png"
    plot_temperature_comparison(
        samples_by_temp=all_samples,
        temperatures=temperatures,
        save_path=plot_path,
        n_show=n_samples_per_temp
    )
    
    # Save data as torch
    data_path = save_dir / "geodesic_temperature_comparison_cifar10.pt"
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
