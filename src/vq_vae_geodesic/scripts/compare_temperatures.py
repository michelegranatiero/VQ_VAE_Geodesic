"""
Compare PixelCNN sampling at different temperatures.

Generates samples at multiple temperature values to visualize
the effect of temperature on sample diversity and sharpness.

Usage:
    python -m vq_vae_geodesic.scripts.compare_temperatures
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from vq_vae_geodesic.config import checkpoint_dir, latents_dir, data_dir
from vq_vae_geodesic.hyperparameters import get_mnist_config
from vq_vae_geodesic.utils import set_seed
from vq_vae_geodesic.models.modules.pixelCNN import PixelCNN
from vq_vae_geodesic.models.modules.vae import build_vae_from_config
from vq_vae_geodesic.evaluation.sample import sample_pixelcnn


def compare_temperatures(temperatures=[0.5, 0.8, 1.0, 1.2, 1.5], n_samples_per_temp=8):
    """
    Generate samples at different temperatures and create comparison plot.
    
    Args:
        temperatures: List of temperature values to compare
        n_samples_per_temp: Number of samples to generate per temperature
    """
    config = get_mnist_config()
    set_seed(config.seed)
    
    device = torch.device("cuda:0" if config.use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load models
    pixelcnn_path = checkpoint_dir() / "pixelcnn_geodesic_mnist_best.pt"
    
    print(f"Loading PixelCNN from {pixelcnn_path}")
    pixelcnn = PixelCNN(
        num_tokens=config.quant_params.n_codewords,
        embed_dim=config.pixelcnn_params.embed_dim,
        hidden_channels=config.pixelcnn_params.hidden_channels,
        n_layers=config.pixelcnn_params.n_layers,
        kernel_size=config.pixelcnn_params.kernel_size
    )
    
    checkpoint = torch.load(pixelcnn_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        pixelcnn.load_state_dict(checkpoint['model_state_dict'])
    else:
        pixelcnn.load_state_dict(checkpoint)
    
    pixelcnn = pixelcnn.to(device)
    pixelcnn.eval()
    
    # Load codebook (torch tensor)
    codebook_path = latents_dir() / "chunk_codebook.pt"
    codebook_data = torch.load(codebook_path, map_location=device)
    codebook_chunks = codebook_data['codebook_chunks'] if isinstance(codebook_data, dict) and 'codebook_chunks' in codebook_data else codebook_data
    
    # Load VAE
    vae_path = checkpoint_dir() / "main_checkpoint_mnist.pt"
    vae = build_vae_from_config(config.arch_params)
    checkpoint = torch.load(vae_path, map_location=device)
    vae.load_state_dict(checkpoint['model_state_dict'])
    vae = vae.to(device)
    vae.eval()
    
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

        # Convert to latents
        latents_list = []
        for b in range(n_samples_per_temp):
            code_indices = codes[b].flatten()
            if isinstance(code_indices, np.ndarray):
                code_indices = torch.from_numpy(code_indices).long()
            chunks = codebook_chunks[code_indices]
            latent = chunks.flatten()
            latents_list.append(latent.cpu().numpy())

        latents = np.stack(latents_list)
        latents_t = torch.from_numpy(latents).float().to(device)

        # Decode
        with torch.no_grad():
            images = vae.decoder(latents_t)

        all_samples.append(images.cpu().numpy())
    
    # Create comparison plot
    n_temps = len(temperatures)
    fig, axes = plt.subplots(n_temps, n_samples_per_temp, 
                            figsize=(2*n_samples_per_temp, 2*n_temps))
    
    for i, (temp, samples) in enumerate(zip(temperatures, all_samples)):
        for j in range(n_samples_per_temp):
            ax = axes[i, j] if n_temps > 1 else axes[j]
            img = samples[j, 0]  # (H, W)
            ax.imshow(img, cmap='gray', vmin=0, vmax=1)
            ax.axis('off')
            
            if j == 0:
                ax.set_ylabel(f"T={temp}", fontsize=12, rotation=0, 
                            labelpad=30, va='center')
            
            if i == 0:
                ax.set_title(f"Sample {j+1}", fontsize=10)
    
    fig.suptitle("PixelCNN Samples at Different Temperatures", fontsize=14, y=0.98)
    plt.tight_layout()
    
    # Save
    save_dir = data_dir() / "samples"
    save_dir.mkdir(exist_ok=True, parents=True)
    save_path = save_dir / "temperature_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved temperature comparison to {save_path}")
    
    # Save data as torch
    data_path = save_dir / "temperature_comparison.pt"
    torch.save({
        'temperatures': temperatures,
        'samples': np.stack(all_samples),
        'n_samples_per_temp': n_samples_per_temp
    }, data_path)
    print(f"Saved comparison data to {data_path}")
    
    print("\nTemperature effects:")
    print("  T < 1.0: Sharper, less diverse (mode-seeking)")
    print("  T = 1.0: Balanced sampling from learned distribution")
    print("  T > 1.0: More diverse, potentially noisier (exploratory)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare PixelCNN sampling temperatures")
    parser.add_argument("--temperatures", nargs='+', type=float, 
                       default=[0.5, 0.8, 1.0, 1.2, 1.5],
                       help="Temperature values to compare")
    parser.add_argument("--n_samples", type=int, default=8,
                       help="Number of samples per temperature")
    
    args = parser.parse_args()
    
    compare_temperatures(
        temperatures=args.temperatures,
        n_samples_per_temp=args.n_samples
    )
