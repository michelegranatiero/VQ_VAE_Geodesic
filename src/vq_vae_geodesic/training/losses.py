import torch
import torch.nn.functional as F

# VAE loss with Binary Cross Entropy for reconstruction (useful for binary images like MNIST)
def vae_loss_bce(x_recon, target, mu, logvar, beta=1.0):
    batch_size = target.size(0)
    # MNIST: 1 channel (grayscale) x 28 x 28 = 784 pixels
    # use view(batch_size, -1) to flatten the images to (batch_size, 784)
    recon_loss = F.binary_cross_entropy(x_recon.view(batch_size, -1), target.view(batch_size, -1), reduction='sum')
    recon_loss = recon_loss / batch_size  # Normalize to get average per image

    kldivergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kldivergence = kldivergence / batch_size  # Normalize to get average per image

    loss = recon_loss + beta * kldivergence

    return loss, recon_loss

def vqvae_loss_bce(x_recon, target):
    batch_size = target.size(0)
    recon_loss = F.binary_cross_entropy(x_recon.view(batch_size, -1), target.view(batch_size, -1), reduction='sum')
    recon_loss = recon_loss / batch_size  # Normalize to get average per image
    return recon_loss



# VAE loss with MSE for reconstruction (useful for natural images like CIFAR-10)
def vae_loss_mse(x_recon, target, mu, logvar, beta=1.0):
    recon_loss = F.mse_loss(x_recon, target, reduction='sum') / target.size(0)  # Normalize by batch size
    kldivergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / target.size(0)  # Normalize by batch size
    loss = recon_loss + beta * kldivergence
    return loss
