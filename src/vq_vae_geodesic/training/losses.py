import torch
import torch.nn.functional as F

# VAE loss with Binary Cross Entropy for reconstruction (useful for binary images like MNIST)
def vae_loss_bce(recon_x, x, mu, logvar, beta=1.0):
    # MNIST: 1 channel (grayscale) x 28 x 28 = 784 pixels
    flat_dim = recon_x.size(1) * recon_x.size(2) * recon_x.size(3)  # Compute flattened dimension
    recon_loss = F.binary_cross_entropy(recon_x.view(-1, flat_dim), x.view(-1, flat_dim), reduction='sum')
    recon_loss = recon_loss / x.size(0)  # Normalize by batch size

    kldivergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / \
        x.size(0)  # Normalize by batch size

    return recon_loss + beta * kldivergence


# VAE loss with MSE for reconstruction (useful for natural images like CIFAR-10)
def vae_loss_mse(recon_x, x, mu, logvar, beta=1.0):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum') / x.size(0)  # Normalize by batch size
    kldivergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / \
        x.size(0)  # Normalize by batch size
    return recon_loss + beta * kldivergence
