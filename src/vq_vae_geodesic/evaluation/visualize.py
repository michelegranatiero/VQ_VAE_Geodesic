"""
Visualization and saving utilities for generated samples
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_samples_grid(samples, out_path, n_rows=4, n_cols=4, title="PixelCNN Samples"):
    """
    Plot a grid of generated samples.
    
    Args:
        samples: Generated images (N, C, H, W) or (N, H, W)
        out_path: Path to save the plot
        n_rows: Number of rows in grid
        n_cols: Number of columns in grid
        title: Title for the figure
    """
    n_samples = min(len(samples), n_rows * n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2*n_cols, 2*n_rows))
    
    for idx, ax in enumerate(axes.flat):
        if idx < n_samples:
            img = samples[idx].squeeze()  # Remove channel dim if present
            ax.imshow(img, cmap='gray', vmin=0, vmax=1)
            ax.axis('off')
            ax.set_title(f"Sample {idx+1}", fontsize=8)
        else:
            ax.axis('off')
    
    fig.suptitle(title, fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved sample grid to {out_path}")


def save_generated_samples(images, codes, latents, data_path, 
                          grid_path=None, grid_title=None, temperature=1.0):
    """
    Save generated samples (images, codes, latents) to disk.
    
    Args:
        images: Generated images as numpy array (N, C, H, W) or (N, H, W)
        codes: Discrete code indices (N, H, W) as numpy array
        latents: Continuous latent vectors (N, D) as numpy array or torch tensor
        data_path: Full path where to save .pt file with all data (Path or str)
        grid_path: Optional path for visualization grid image (Path or str). If None, no grid is saved.
        grid_title: Title for the grid visualization (default: "Generated Samples")
        temperature: Sampling temperature (saved in .pt metadata)
        
    Returns:
        dict with paths to saved files: {'data': Path, 'grid': Path or None}
    """
    data_path = Path(data_path)
    data_path.parent.mkdir(exist_ok=True, parents=True)
    
    n_samples = len(images)
    
    # Convert latents to numpy if tensor
    if isinstance(latents, torch.Tensor):
        latents_np = latents.cpu().numpy()
    else:
        latents_np = latents
    
    saved_paths = {}
    
    # Save codes, latents, and images as torch
    torch.save({
        'codes': codes,
        'latents': latents_np,
        'images': images,
        'temperature': temperature,
        'n_samples': n_samples
    }, data_path)
    print(f"Saved codes, latents, and images to {data_path}")
    saved_paths['data'] = data_path
    
    # Create grid visualization (optional)
    if grid_path is not None:
        grid_path = Path(grid_path)
        grid_path.parent.mkdir(exist_ok=True, parents=True)
        
        n_rows = int(np.ceil(np.sqrt(n_samples)))
        n_cols = int(np.ceil(n_samples / n_rows))
        
        if grid_title is None:
            grid_title = "Generated Samples"
        
        plot_samples_grid(
            images, 
            grid_path, 
            n_rows=n_rows, 
            n_cols=n_cols,
            title=grid_title
        )
        saved_paths['grid'] = grid_path
    
    return saved_paths


def plot_comparison_2rows(top_imgs, bottom_imgs, top_label, bottom_label, 
                         save_path, title, n_show=16):
    """
    Plot 2-row comparison (e.g., PixelCNN vs Random).
    
    Args:
        top_imgs: Images for top row (N, C, H, W)
        bottom_imgs: Images for bottom row (N, C, H, W)
        top_label: Label for top row
        bottom_label: Label for bottom row
        save_path: Path to save figure
        title: Main title for the figure
        n_show: Number of samples to show
    """
    n = min(n_show, len(top_imgs), len(bottom_imgs))
    
    fig, axes = plt.subplots(2, n, figsize=(2*n, 5))
    
    # Top row
    for i in range(n):
        ax = axes[0, i]
        ax.imshow(top_imgs[i].squeeze(), cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
        if i == 0:
            ax.text(-0.1, 0.5, top_label, fontsize=11, fontweight='bold', 
                   rotation=0, ha='right', va='center', transform=ax.transAxes)
    
    # Bottom row
    for i in range(n):
        ax = axes[1, i]
        ax.imshow(bottom_imgs[i].squeeze(), cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
        if i == 0:
            ax.text(-0.1, 0.5, bottom_label, fontsize=11, fontweight='bold', 
                   rotation=0, ha='right', va='center', transform=ax.transAxes)
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved comparison to {save_path}")


def plot_temperature_comparison(samples_by_temp, temperatures, save_path, n_show=8):
    """
    Plot samples at different temperatures (one row per temperature).
    
    Args:
        samples_by_temp: List of image arrays, one per temperature (N, C, H, W)
        temperatures: List of temperature values
        save_path: Path to save figure
        n_show: Number of samples to show per temperature
    """
    n_temps = len(temperatures)
    n = min(n_show, min(len(imgs) for imgs in samples_by_temp))
    
    fig, axes = plt.subplots(n_temps, n, figsize=(2*n, 2*n_temps))
    
    if n_temps == 1:
        axes = axes.reshape(1, -1)
    
    for i, (temp, samples) in enumerate(zip(temperatures, samples_by_temp)):
        for j in range(n):
            ax = axes[i, j]
            ax.imshow(samples[j].squeeze(), cmap='gray', vmin=0, vmax=1)
            ax.axis('off')
            
            # Label temperature on the left
            if j == 0:
                ax.text(-0.15, 0.5, f"T={temp}", fontsize=14, fontweight='bold',
                       rotation=0, ha='right', va='center', transform=ax.transAxes)
            
            # Column headers
            if i == 0:
                ax.set_title(f"Sample {j+1}", fontsize=10)
    
    fig.suptitle("PixelCNN Samples at Different Temperatures", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved temperature comparison to {save_path}")
