import torch
import wandb
from vq_vae_geodesic.utils import set_seed
from vq_vae_geodesic.hyperparameters import get_cifar10_config
from vq_vae_geodesic.data.loaders import get_cifar_loaders
from vq_vae_geodesic.evaluation.utils import (
    load_model_vae_cifar10,
    load_codebook_cifar10,
    load_codes_indices_cifar10,
    load_model_vqvae_cifar10
)
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
from vq_vae_geodesic.config import recons_dir


def launch_evaluation():
    config = get_cifar10_config()
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models (and codebook for geodesic)
    # VAE (and VAE + Geodesic)
    vae = load_model_vae_cifar10(config.arch_params, device)
    codebook_chunks = load_codebook_cifar10(device)
    _, _, test_codes = load_codes_indices_cifar10()
    # VQ-VAE
    vqvae = load_model_vqvae_cifar10(config.arch_params, config.vqvae_params, device)

    # Loader test (NOT shuffled for consistent evaluation)
    _, _, test_loader = get_cifar_loaders(
        batch_size=config.data_params.batch_size,
        shuffle_train_set=False
    )

    # VAE run
    # wandb.init(project="vq_vae_geodesic", name="vae_evaluation_cifar10", config=config, reinit=True)
    metrics_vae = evaluate_vae_celeba(vae, test_loader, device)
    # wandb.log(add_prefix_to_dict(metrics_vae, "test/"))
    # wandb.finish()

    # VAE + Geodesic run
    # wandb.init(project="vq_vae_geodesic", name="geodesic_evaluation_cifar10", config=config, reinit=True)
    metrics_geodesic = evaluate_vae_geodesic_celeba(vae, test_loader, codebook_chunks, test_codes, device)
    # wandb.log(add_prefix_to_dict(metrics_geodesic, "test/"))
    # wandb.finish()

    # VQ-VAE run
    # wandb.init(project="vq_vae_geodesic", name="vqvae_evaluation_cifar10", config=config, reinit=True)
    metrics_vqvae = evaluate_vqvae_celeba(vqvae, test_loader, device)
    # wandb.log(add_prefix_to_dict(metrics_vqvae, "test/"))
    # wandb.finish()

    # Print metric comparisons
    print("\n=== Test Set Metric Comparison (CIFAR-10) ===")
    print(
        f"MSE:          VAE={metrics_vae['mse']:.6f}  Geodesic={metrics_geodesic['mse']:.6f}  VQ-VAE={metrics_vqvae['mse']:.6f}")
    print(
        f"PSNR:         VAE={metrics_vae['psnr']:.2f}   Geodesic={metrics_geodesic['psnr']:.2f}   VQ-VAE={metrics_vqvae['psnr']:.2f}")
    print(
        f"SSIM:         VAE={metrics_vae['ssim']:.4f}  Geodesic={metrics_geodesic['ssim']:.4f}  VQ-VAE={metrics_vqvae['ssim']:.4f}")
    print(
        f"L1:           VAE={metrics_vae['l1']:.6f}  Geodesic={metrics_geodesic['l1']:.6f}  VQ-VAE={metrics_vqvae['l1']:.6f}")
    print(
        f"Recon Loss:   VAE={metrics_vae['recon_loss']:.6f}  Geodesic={metrics_geodesic['recon_loss']:.6f}  VQ-VAE={metrics_vqvae['recon_loss']:.6f}")

    # Collect images for plotting
    print("=== Generating reconstruction plots ===")
    n_show = 16
    orig_imgs, vae_recon = get_vae_reconstructions(vae, test_loader, device, n_show=n_show)
    _, geodesic_recon = get_geodesic_reconstructions(
        vae, test_loader, codebook_chunks, test_codes, device, n_show=n_show)
    _, vqvae_recon = get_vqvae_reconstructions(vqvae, test_loader, device, n_show=n_show)

    # Save comparison plot
    save_dir = recons_dir('cifar10')
    save_dir.mkdir(exist_ok=True, parents=True)
    plot_path = save_dir / "reconstructions_comparison_cifar10.png"
    plot_reconstructions_comparison(orig_imgs, vae_recon, geodesic_recon, vqvae_recon, 
                                   save_path=plot_path, n_show=n_show)
    print(f"\n Reconstruction comparison saved to {plot_path}")


if __name__ == "__main__":
    launch_evaluation()
