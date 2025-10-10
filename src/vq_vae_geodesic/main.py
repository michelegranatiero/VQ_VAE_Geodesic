"""
DEPRECATED: This file is kept for backward compatibility and quick testing.
For production use, please use the scripts in the scripts/ folder:
  - scripts/train.py: Train VAE
  - scripts/extract_latents.py: Extract latent representations
  - scripts/quantize.py: Perform geodesic quantization
  - scripts/reconstruct.py: Reconstruct images from codebook

Example usage:
    python scripts/train.py --dataset mnist --epochs 100
    python scripts/extract_latents.py --checkpoint data/checkpoints/checkpoint_mnist.pth --dataset mnist
    python scripts/quantize.py --latents data/latents/train_latents.npz
    python scripts/reconstruct.py --checkpoint data/checkpoints/checkpoint_mnist.pth --codebook data/latents/chunk_codebook.npz --dataset mnist
"""
import numpy as np
from vq_vae_geodesic.reconstruct_codebook import recon_from_mu, reconstruct_from_chunk_codebook
from vq_vae_geodesic.extract_latents import extract_and_save_latents
from vq_vae_geodesic.train import fit_vae
from vq_vae_geodesic.losses import vae_loss_bce, vae_loss_mse
from vq_vae_geodesic.data import get_cifar_loaders, get_MNIST_loaders
from vq_vae_geodesic.utils import set_seed
from vq_vae_geodesic.models import (
    Decoder_CIFAR,
    Decoder_MNIST,
    Encoder_CIFAR,
    Encoder_MNIST,
    VariationalAutoencoder,
    GeodesicQuantizer,
)
from vq_vae_geodesic.hyperparameters import get_mnist_config
import os
import torch
import warnings
warnings.warn(
    "main.py is deprecated. Please use scripts in the scripts/ folder instead.",
    DeprecationWarning,
    stacklevel=2
)


# Load default configuration
config = get_mnist_config()

# CONFIGURATION (for backward compatibility)
SEED = config.data_params.seed
RESUME_TRAINING = False
EXSTRACT_LATENTS = False

CAPACITY = config.arch_params.hidden_channels
LATENT_DIM = config.arch_params.latent_dim
NUM_EPOCHS = config.training_params.num_epochs
BATCH_SIZE = config.data_params.batch_size
LR_RATE = config.training_params.lr
WEIGHT_DECAY = config.training_params.weight_decay
VARIATIONAL_BETA = config.training_params.variational_beta

SAVE_CHECKPOINT_EVERY = config.training_params.save_checkpoint_every
USE_GPU = config.use_gpu
NUM_WORKERS = config.data_params.num_workers


def train_cifar(resume=False):
    """
    DEPRECATED: Use scripts/train.py instead
    Example: python scripts/train.py --dataset cifar --epochs 100
    """
    warnings.warn("Use scripts/train.py instead of main.train_cifar()", DeprecationWarning)
    set_seed(SEED)
    device = torch.device("cuda:0" if USE_GPU and torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = get_cifar_loaders(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    encoder = Encoder_CIFAR(in_channels=3, hidden_channels=CAPACITY, latent_dim=LATENT_DIM)
    decoder = Decoder_CIFAR(out_channels=3, hidden_channels=CAPACITY, latent_dim=LATENT_DIM)
    vae = VariationalAutoencoder(encoder, decoder)
    vae.to(device)

    optimizer = torch.optim.Adam(params=vae.parameters(), lr=LR_RATE, weight_decay=WEIGHT_DECAY)
    loss = vae_loss_mse

    start_epoch = 1
    train_loss_history = []
    val_loss_history = []
    checkpoint_path = "data/checkpoints/checkpoint_cifar.pth"

    if resume and os.path.exists(checkpoint_path):
        print(f"Resuming training from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        vae.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        train_loss_history = checkpoint.get('train_loss_history', [])
        val_loss_history = checkpoint.get('val_loss_history', [])
        print(f"Resumed from epoch {start_epoch}")
    else:
        print("Starting training from scratch")

    train_loss_avg, val_loss_avg = fit_vae(
        model=vae,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss,
        variational_beta=VARIATIONAL_BETA,
        num_epochs=NUM_EPOCHS,
        device=device,
        start_epoch=start_epoch,
        checkpoint_path=checkpoint_path,
        train_loss_history=train_loss_history,
        val_loss_history=val_loss_history,
        save_checkpoint_every=SAVE_CHECKPOINT_EVERY
    )


def train_mnist(resume=False, extract_latents=False):
    """
    DEPRECATED: Use modular scripts instead

    For training:
        python scripts/train.py --dataset mnist --epochs 100

    For full pipeline:
        python scripts/train.py --dataset mnist --epochs 100
        python scripts/extract_latents.py --checkpoint data/checkpoints/checkpoint_mnist.pth --dataset mnist
        python scripts/quantize.py --latents data/latents/train_latents.npz
        python scripts/reconstruct.py --checkpoint data/checkpoints/checkpoint_mnist.pth --codebook data/latents/chunk_codebook.npz --dataset mnist
    """
    warnings.warn("Use modular scripts in scripts/ folder instead of main.train_mnist()", DeprecationWarning)
    set_seed(SEED)
    device = torch.device("cuda:0" if USE_GPU and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = get_MNIST_loaders(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    encoder = Encoder_MNIST(in_channels=1, hidden_channels=CAPACITY, latent_dim=LATENT_DIM)
    decoder = Decoder_MNIST(out_channels=1, hidden_channels=CAPACITY, latent_dim=LATENT_DIM)
    vae = VariationalAutoencoder(encoder, decoder)
    vae.to(device)

    optimizer = torch.optim.Adam(params=vae.parameters(), lr=LR_RATE)
    loss = vae_loss_bce

    start_epoch = 1
    train_loss_history = []
    val_loss_history = []
    checkpoint_path = "data/checkpoints/checkpoint_mnist.pth"

    if (resume or extract_latents) and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        vae.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        train_loss_history = checkpoint.get('train_loss_history', [])
        val_loss_history = checkpoint.get('val_loss_history', [])

    if not extract_latents:
        train_loss_avg, val_loss_avg = fit_vae(
            model=vae,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            loss_fn=loss,
            variational_beta=VARIATIONAL_BETA,
            num_epochs=NUM_EPOCHS,
            device=device,
            start_epoch=start_epoch,
            checkpoint_path=checkpoint_path,
            train_loss_history=train_loss_history,
            val_loss_history=val_loss_history,
            save_checkpoint_every=SAVE_CHECKPOINT_EVERY
        )

    if extract_latents:
        latents_dir = "data/latents"
        train_loader_no_shuffle = torch.utils.data.DataLoader(
            train_loader.dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

        extract_and_save_latents(model=vae, dataloader=train_loader_no_shuffle,
                                 device=device, save_dir=latents_dir, prefix="train")
        extract_and_save_latents(model=vae, dataloader=val_loader, device=device,
                                 save_dir=latents_dir, prefix="val")

        npz = np.load("data/latents/train_latents.npz")
        Z_mu = npz['mu']
        Z_logvar = npz['logvar']
        N, D = Z_mu.shape
        L = config.quant_params.n_chunks
        chunk_size = config.quant_params.chunk_size
        H, W = config.quant_params.grid_h, config.quant_params.grid_w

        quantizer = GeodesicQuantizer(
            n_chunks=L,
            chunk_size=chunk_size,
            n_codewords=config.quant_params.n_codewords,
            k=config.quant_params.knn_k
        )
        quantizer.fit(Z_mu, Z_logvar)
        codes_per_image = quantizer.assign(Z_mu)
        codes_grid = codes_per_image.reshape(N, H, W)
        quantizer.save("data/latents/chunk_codebook.npz",
                       codes_per_image=codes_per_image, codes_grid=codes_grid)
        print("[INFO] Saved chunk codebook and codes to data/latents/chunk_codebook.npz")

        print("[INFO] Recon on TRAIN using chunk-codebook...")
        mse_codebook_train = reconstruct_from_chunk_codebook(
            vae, train_loader_no_shuffle, quantizer.codebook_chunks, codes_per_image, device, out_dir="data/recons_train")
        print("MSE chunk-codebook recon (train):", mse_codebook_train)

        print("[INFO] Recon on TRAIN using mu (baseline)...")
        mse_mu_train = recon_from_mu(vae, train_loader_no_shuffle, device, out_dir="data/recons_train")
        print("MSE mu recon (train):", mse_mu_train)

        npz_val = np.load("data/latents/val_latents.npz")
        Z_mu_val = npz_val['mu']
        N_val = Z_mu_val.shape[0]
        val_codes_per_image = quantizer.assign(Z_mu_val)
        print("[INFO] Recon on VAL using chunk-codebook...")
        mse_codebook_val = reconstruct_from_chunk_codebook(
            vae, val_loader, quantizer.codebook_chunks, val_codes_per_image, device, out_dir="data/recons_val")
        print("MSE chunk-codebook recon (val):", mse_codebook_val)

        print("[INFO] Ricostruzione (val) da mu (baseline)...")
        mse_mu = recon_from_mu(vae, val_loader, device, out_dir="data/recons_val")
        print("MSE mu recon (val):", mse_mu)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("WARNING: This main.py is DEPRECATED!")
    print("="*70)
    print("\nPlease use the modular scripts instead:\n")
    print("1. Train VAE:")
    print("   python scripts/train.py --dataset mnist --epochs 100\n")
    print("2. Extract latents:")
    print("   python scripts/extract_latents.py --checkpoint data/checkpoints/checkpoint_mnist.pth --dataset mnist\n")
    print("3. Build codebook:")
    print("   python scripts/quantize.py --latents data/latents/train_latents.npz\n")
    print("4. Reconstruct:")
    print("   python scripts/reconstruct.py --checkpoint data/checkpoints/checkpoint_mnist.pth --codebook data/latents/chunk_codebook.npz --dataset mnist\n")
    print("="*70)
    print("\nRunning legacy demo (for backward compatibility)...\n")

    # Legacy demo
    train_mnist(resume=RESUME_TRAINING, extract_latents=EXSTRACT_LATENTS)
