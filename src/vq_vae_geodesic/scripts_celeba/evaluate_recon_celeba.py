import torch
from vq_vae_geodesic.utils import set_seed
from vq_vae_geodesic.hyperparameters import get_celeba_config
from vq_vae_geodesic.data.loaders import get_celeba_loaders
from vq_vae_geodesic.config import recons_dir, checkpoint_dir, latents_dir
from vq_vae_geodesic.evaluation.evaluate import (
    evaluate_vae_celeba,
    evaluate_vae_geodesic_celeba,
    evaluate_vqvae_celeba
)
from vq_vae_geodesic.evaluation.reconstructions import (
    get_vae_reconstructions,
    get_geodesic_reconstructions,
    get_vqvae_reconstructions,
    plot_reconstructions_comparison
)


def launch_evaluation():
    config = get_celeba_config(img_size=32)
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load VAE
    print("Loading VAE...")
    vae_checkpoint = torch.load(checkpoint_dir('celeba') / "vae_celeba_best.pt", map_location=device)
    from vq_vae_geodesic.models.modules.vae import build_vae_from_config
    vae = build_vae_from_config(config.arch_params, dataset="celeba")
    vae.load_state_dict(vae_checkpoint['model_state_dict'])
    vae = vae.to(device)
    vae.eval()
    
    # Load codebook
    print("Loading codebook...")
    codebook_path = latents_dir('celeba') / "chunk_codebook_celeba.pt"
    codebook_data = torch.load(codebook_path, map_location=device)
    codebook_chunks = codebook_data['codebook_chunks']
    
    # Load test codes
    print("Loading test codes...")
    codes_data = torch.load(latents_dir('celeba') / "assigned_codes_celeba.pt", map_location=device)
    test_codes = codes_data['test_codes']
    
    # Load VQ-VAE
    print("Loading VQ-VAE...")
    vqvae_checkpoint = torch.load(checkpoint_dir('celeba') / "vqvae_celeba_best.pt", map_location=device)
    from vq_vae_geodesic.models.modules.vqvae import build_vqvae_from_config
    vqvae = build_vqvae_from_config(config.arch_params, config.vqvae_params, dataset="celeba")
    vqvae.load_state_dict(vqvae_checkpoint['model_state_dict'])
    vqvae = vqvae.to(device)
    vqvae.eval()

    # Get test loader with larger batch size for faster evaluation
    _, _, test_loader = get_celeba_loaders(
        img_size=32,
        batch_size=512,  # Use large batch size for fast evaluation
        num_workers=0,
        shuffle_train_set=False
    )

    # Evaluate all models
    print("\nEvaluating VAE...")
    metrics_vae = evaluate_vae_celeba(vae, test_loader, device)
    
    print("Evaluating VAE + Geodesic...")
    metrics_geodesic = evaluate_vae_geodesic_celeba(vae, test_loader, codebook_chunks, test_codes, device)
    
    print("Evaluating VQ-VAE...")
    metrics_vqvae = evaluate_vqvae_celeba(vqvae, test_loader, device)

    # Print comparison
    print("\n=== Test Set Metric Comparison (CelebA) ===")
    print(f"MSE:          VAE={metrics_vae['mse']:.6f}  Geodesic={metrics_geodesic['mse']:.6f}  VQ-VAE={metrics_vqvae['mse']:.6f}")
    print(f"PSNR:         VAE={metrics_vae['psnr']:.2f}   Geodesic={metrics_geodesic['psnr']:.2f}   VQ-VAE={metrics_vqvae['psnr']:.2f}")
    print(f"SSIM:         VAE={metrics_vae['ssim']:.4f}  Geodesic={metrics_geodesic['ssim']:.4f}  VQ-VAE={metrics_vqvae['ssim']:.4f}")
    print(f"L1:           VAE={metrics_vae['l1']:.6f}  Geodesic={metrics_geodesic['l1']:.6f}  VQ-VAE={metrics_vqvae['l1']:.6f}")
    print(f"Recon Loss:   VAE={metrics_vae['recon_loss']:.6f}  Geodesic={metrics_geodesic['recon_loss']:.6f}  VQ-VAE={metrics_vqvae['recon_loss']:.6f}")

    # Generate reconstruction plots
    print("\n=== Generating reconstruction plots ===")
    n_show = 16
    orig_imgs, vae_recon = get_vae_reconstructions(vae, test_loader, device, n_show=n_show)
    _, geodesic_recon = get_geodesic_reconstructions(
        vae, test_loader, codebook_chunks, test_codes, device, n_show=n_show)
    _, vqvae_recon = get_vqvae_reconstructions(vqvae, test_loader, device, n_show=n_show)

    # Save comparison plot
    save_dir = recons_dir('celeba')
    save_dir.mkdir(exist_ok=True, parents=True)
    plot_path = save_dir / "reconstructions_comparison_celeba.png"
    plot_reconstructions_comparison(orig_imgs, vae_recon, geodesic_recon, vqvae_recon, 
                                   save_path=plot_path, n_show=n_show)
    print(f"\nReconstruction comparison saved to {plot_path}")


if __name__ == "__main__":
    launch_evaluation()
