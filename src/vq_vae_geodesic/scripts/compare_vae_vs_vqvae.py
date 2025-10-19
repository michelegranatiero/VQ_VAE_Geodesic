"""
Compare VAE + Geodesic Quantization vs VQ-VAE.

Evaluates and compares two approaches to learning discrete representations:
1. VAE + Post-hoc Geodesic Quantization (your approach)
2. VQ-VAE with end-to-end learned codebook (baseline)

Metrics:
- Reconstruction quality (MSE)
- Codebook utilization
- Sample quality from PixelCNN prior

Usage:
    python -m vq_vae_geodesic.scripts.compare_vae_vs_vqvae
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from vq_vae_geodesic.config import checkpoint_dir, latents_dir, data_dir
from vq_vae_geodesic.hyperparameters import get_mnist_config
from vq_vae_geodesic.utils import set_seed
from vq_vae_geodesic.data.loaders import get_MNIST_loaders
from vq_vae_geodesic.models.modules.vae import build_vae_from_config
from vq_vae_geodesic.models.modules.vqvae import build_vqvae_from_config


def evaluate_vae_geodesic(model, quantizer, data_loader, device, n_samples=16):
    """Evaluate VAE + Geodesic Quantization."""
    model.eval()
    
    all_mse = []
    orig_imgs = []
    recon_imgs = []
    codes_list = []
    
    # Carica codici quantizzati da file assegnato (es. assigned_codes.pt)
    assigned_codes_path = latents_dir() / "assigned_codes.pt"
    assigned_codes = torch.load(assigned_codes_path)
    # Usa la chiave corretta 'val_codes'
    codes_tensor = assigned_codes['val_codes']
    codes_tensor = codes_tensor.cpu().numpy()

    with torch.no_grad():
        idx = 0
        for x, _ in data_loader:
            x = x.to(device)
            batch_size = x.size(0)
            codes = codes_tensor[idx:idx+batch_size]
            codes_list.extend(codes)
            # Reconstruct from quantized codes
            z_quantized_list = []
            for b in range(batch_size):
                chunks = quantizer.codebook_chunks[codes[b]]
                latent = chunks.flatten()
                z_quantized_list.append(latent.cpu().numpy())
            z_quantized = np.stack(z_quantized_list)
            z_quantized_t = torch.from_numpy(z_quantized).float().to(device)
            recon = model.decoder(z_quantized_t)
            # Compute MSE
            mse = torch.mean((recon - x)**2).item()
            all_mse.append(mse)
            # Collect samples
            if len(orig_imgs) < n_samples:
                to_take = min(n_samples - len(orig_imgs), x.size(0))
                orig_imgs.append(x[:to_take].cpu())
                recon_imgs.append(recon[:to_take].cpu())
            idx += batch_size
    
    # Compute metrics
    avg_mse = np.mean(all_mse)
    
    # Codebook utilization
    unique_codes = len(np.unique(codes_list))
    utilization = unique_codes / quantizer.n_codewords * 100
    
    # Concatenate samples
    orig_imgs = torch.cat(orig_imgs, dim=0).numpy()
    recon_imgs = torch.cat(recon_imgs, dim=0).numpy()
    
    return {
        'mse': avg_mse,
        'codebook_utilization': utilization,
        'unique_codes': unique_codes,
        'orig_imgs': orig_imgs,
        'recon_imgs': recon_imgs
    }


def evaluate_vqvae(model, data_loader, device, n_samples=8):
    """Evaluate VQ-VAE."""
    model.eval()
    
    all_mse = []
    orig_imgs = []
    recon_imgs = []
    codes_list = []
    
    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)
            
            # Forward pass
            recon, vq_loss, codes = model(x)
            
            # Compute MSE
            mse = torch.mean((recon - x)**2).item()
            all_mse.append(mse)
            
            # Collect codes
            codes_list.extend(codes.cpu().flatten().numpy())
            
            # Collect samples
            if len(orig_imgs) < n_samples:
                to_take = min(n_samples - len(orig_imgs), x.size(0))
                orig_imgs.append(x[:to_take].cpu())
                recon_imgs.append(recon[:to_take].cpu())
    
    # Compute metrics
    avg_mse = np.mean(all_mse)
    
    # Codebook utilization
    unique_codes = len(np.unique(codes_list))
    utilization = unique_codes / model.vq.num_embeddings * 100
    
    # Concatenate samples
    orig_imgs = torch.cat(orig_imgs, dim=0).numpy()
    recon_imgs = torch.cat(recon_imgs, dim=0).numpy()
    
    return {
        'mse': avg_mse,
        'codebook_utilization': utilization,
        'unique_codes': unique_codes,
        'orig_imgs': orig_imgs,
        'recon_imgs': recon_imgs
    }


def plot_comparison(vae_results, vqvae_results, save_path):
    """Plot side-by-side comparison."""
    n = 8
    
    fig, axes = plt.subplots(3, n, figsize=(2*n, 8))
    
    # Row 1: Original images (shared)
    for i in range(n):
        ax = axes[0, i]
        ax.imshow(vae_results['orig_imgs'][i, 0], cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
        if i == 0:
            ax.text(-0.1, 0.5, "Original\nImages", fontsize=11, fontweight='bold',
                   rotation=0, ha='right', va='center', transform=ax.transAxes)
    
    # Row 2: VAE + Geodesic reconstructions
    for i in range(n):
        ax = axes[1, i]
        ax.imshow(vae_results['recon_imgs'][i, 0], cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
        if i == 0:
            ax.text(-0.1, 0.5, "VAE +\nGeodesic\nQuantization\n(a posteriori)", 
                   fontsize=11, fontweight='bold', rotation=0, ha='right', 
                   va='center', transform=ax.transAxes, color='blue')
    
    # Row 3: VQ-VAE reconstructions
    for i in range(n):
        ax = axes[2, i]
        ax.imshow(vqvae_results['recon_imgs'][i, 0], cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
        if i == 0:
            ax.text(-0.1, 0.5, "VQ-VAE\n(end-to-end\nlearned)", 
                   fontsize=11, fontweight='bold', rotation=0, ha='right', 
                   va='center', transform=ax.transAxes, color='green')
    
    fig.suptitle("VAE + Geodesic Quantization vs VQ-VAE: Reconstruction Comparison", 
                fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved comparison to {save_path}")


def compare_models():
    """Main comparison function."""
    config = get_mnist_config()
    set_seed(config.seed)
    
    device = torch.device("cuda:0" if config.use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data (use validation set for fair comparison)
    _, val_loader, _ = get_MNIST_loaders(batch_size=128, shuffle_train_set=False)
    
    # ===== Evaluate VAE + Geodesic =====
    print("\n" + "="*60)
    print("Evaluating VAE + Geodesic Quantization")
    print("="*60)

    # Load VAE
    vae_path = checkpoint_dir() / "main_checkpoint_mnist.pt"
    if not vae_path.exists():
        raise FileNotFoundError(f"VAE checkpoint not found at {vae_path}")

    vae = build_vae_from_config(config.arch_params)
    checkpoint = torch.load(vae_path, map_location=device)
    vae.load_state_dict(checkpoint['model_state_dict'])
    vae = vae.to(device)
    vae.eval()
    print(f"Loaded VAE from {vae_path}")

    # Load geodesic codebook (torch tensor)
    codebook_path = latents_dir() / "chunk_codebook.pt"
    if not codebook_path.exists():
        raise FileNotFoundError(f"Geodesic codebook not found at {codebook_path}")

    codebook_data = torch.load(codebook_path, map_location=device)
    codebook_chunks = codebook_data['codebook_chunks'] if isinstance(codebook_data, dict) and 'codebook_chunks' in codebook_data else codebook_data
    print(f"Loaded geodesic codebook: {tuple(codebook_chunks.shape)}")

    # Usa un oggetto semplice per passare solo il codebook
    class CodebookWrapper:
        def __init__(self, codebook_chunks):
            self.codebook_chunks = codebook_chunks
            self.n_codewords = codebook_chunks.shape[0]

    quantizer = CodebookWrapper(codebook_chunks)
    vae_results = evaluate_vae_geodesic(vae, quantizer, val_loader, device)
    
    print(f"\nVAE + Geodesic Results:")
    print(f"MSE: {vae_results['mse']:.6f}")
    print(f"Codebook utilization: {vae_results['codebook_utilization']:.2f}%")
    print(f"Unique codes: {vae_results['unique_codes']}/{quantizer.n_codewords}")
    
    # ===== Evaluate VQ-VAE =====
    print("\n" + "="*60)
    print("Evaluating VQ-VAE")
    print("="*60)
    
    # Load VQ-VAE
    vqvae_path = checkpoint_dir() / "vqvae_mnist_best.pt"
    
    if not vqvae_path.exists():
        raise FileNotFoundError(
            f"VQ-VAE checkpoint not found. Train VQ-VAE first:\n"
            "python -m vq_vae_geodesic.scripts.train_vqvae_mnist"
        )
    
    vqvae = build_vqvae_from_config(config.arch_params, config.vqvae_params)
    checkpoint = torch.load(vqvae_path, map_location=device)
    vqvae.load_state_dict(checkpoint['model_state_dict'])
    vqvae = vqvae.to(device)
    vqvae.eval()
    print(f"Loaded VQ-VAE from {vqvae_path}")
    
    vqvae_results = evaluate_vqvae(vqvae, val_loader, device)
    
    print(f"\nVQ-VAE Results:")
    print(f"  MSE: {vqvae_results['mse']:.6f}")
    print(f"  Codebook utilization: {vqvae_results['codebook_utilization']:.2f}%")
    print(f"  Unique codes: {vqvae_results['unique_codes']}/{config.vqvae_params.num_embeddings}")
    
    # ===== Comparison =====
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    print(f"\nReconstruction Quality (lower is better):")
    print(f"  VAE + Geodesic MSE: {vae_results['mse']:.6f}")
    print(f"  VQ-VAE MSE:         {vqvae_results['mse']:.6f}")
    
    if vae_results['mse'] < vqvae_results['mse']:
        print(f"  → VAE + Geodesic is {(vqvae_results['mse']/vae_results['mse'] - 1)*100:.2f}% better")
    else:
        print(f"  → VQ-VAE is {(vae_results['mse']/vqvae_results['mse'] - 1)*100:.2f}% better")
    
    print(f"\nCodebook Utilization (higher is better):")
    print(f"  VAE + Geodesic: {vae_results['codebook_utilization']:.2f}%")
    print(f"  VQ-VAE:         {vqvae_results['codebook_utilization']:.2f}%")
    
    # Save comparison plot
    save_dir = data_dir() / "comparison"
    save_dir.mkdir(exist_ok=True, parents=True)
    save_path = save_dir / "vae_vs_vqvae_reconstructions.png"
    plot_comparison(vae_results, vqvae_results, save_path)
    
    # Save metrics as torch
    metrics_path = save_dir / "comparison_metrics.pt"
    torch.save({
        'vae_mse': vae_results['mse'],
        'vae_utilization': vae_results['codebook_utilization'],
        'vae_unique_codes': vae_results['unique_codes'],
        'vqvae_mse': vqvae_results['mse'],
        'vqvae_utilization': vqvae_results['codebook_utilization'],
        'vqvae_unique_codes': vqvae_results['unique_codes']
    }, metrics_path)
    print(f"\nSaved metrics to {metrics_path}")
    
    print("\nKey Insights:")
    print("  - Geodesic quantization: Post-hoc, preserves VAE training")
    print("  - VQ-VAE: End-to-end, joint optimization")
    print("  - Both approaches can use PixelCNN for sampling")


if __name__ == "__main__":
    compare_models()
