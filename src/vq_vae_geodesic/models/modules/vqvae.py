"""
VQ-VAE model with end-to-end learned codebook.

Standard VQ-VAE as in "Neural Discrete Representation Learning" (van den Oord et al., 2017).
This serves as a baseline comparison against the geodesic quantization approach.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from vq_vae_geodesic.models.modules.encoder import (
    Encoder_MNIST_VQVAE,
    Encoder_CIFAR_VQVAE
)
from vq_vae_geodesic.models.modules.decoder import (
    Decoder_MNIST_VQVAE,
    Decoder_CIFAR_VQVAE
)


class VectorQuantizer(nn.Module):
    """
    Vector Quantization layer with learned codebook.
    
    Maps continuous encoder outputs to discrete codes via nearest-neighbor
    lookup in a learned embedding space. Uses straight-through estimator
    for gradient backpropagation.
    
    Args:
        num_embeddings: K, size of the codebook (vocabulary)
        embedding_dim: D, dimensionality of each embedding vector
        commitment_cost: Beta, weight for encoder commitment loss
    """
    
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
        # Learnable codebook embeddings
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
    
    def forward(self, z):
        """
        Quantize encoder outputs to nearest codebook embeddings.
        
        Args:
            z: Encoder outputs (B, D, H, W)
            
        Returns:
            quantized: Quantized latents (B, D, H, W)
            loss: VQ loss (codebook + commitment)
            encoding_indices: Discrete codes (B, H, W)
        """
        # Reshape: (B, D, H, W) → (B, H, W, D) → (B*H*W, D)
        z_flattened = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z_flattened.view(-1, self.embedding_dim)
        
        # Compute L2 distances to all embeddings
        # ||z - e||^2 = ||z||^2 + ||e||^2 - 2 * z·e
        distances = (
            torch.sum(z_flattened**2, dim=1, keepdim=True) +
            torch.sum(self.embeddings.weight**2, dim=1) -
            2 * torch.matmul(z_flattened, self.embeddings.weight.t())
        )
        
        # Find nearest embedding for each position
        encoding_indices = torch.argmin(distances, dim=1)  # (B*H*W,)
        
        # Convert indices to one-hot encodings
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
        
        # Lookup quantized vectors
        quantized = torch.matmul(encodings, self.embeddings.weight)  # (B*H*W, D)
        
        # Reshape back: (B*H*W, D) → (B, H, W, D) → (B, D, H, W)
        quantized = quantized.view(z.shape[0], z.shape[2], z.shape[3], self.embedding_dim)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        # Reshape encoding_indices to (B, H, W)
        encoding_indices = encoding_indices.view(z.shape[0], z.shape[2], z.shape[3])

        # VQ Loss
        # e_latent_loss: push encoder outputs toward chosen embeddings
        e_latent_loss = F.mse_loss(quantized.detach(), z, reduction='sum') / z.size(0)
        # q_latent_loss: move embeddings toward encoder outputs
        q_latent_loss = F.mse_loss(quantized, z.detach(), reduction='sum') / z.size(0)
        
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight-through estimator: forward uses quantized, backward uses z
        quantized = z + (quantized - z).detach()

        return quantized, loss, encoding_indices


class VQVAE(nn.Module):
    """
    VQ-VAE model with end-to-end learned codebook.
        
    Loss:
        Total = Reconstruction + VQ Loss (codebook + commitment)
        
    Args:
        encoder: Encoder module
        decoder: Decoder module
        num_embeddings: Codebook size
        embedding_dim: Embedding dimensionality
        commitment_cost: Beta for commitment loss
    """
    
    def __init__(self, encoder, decoder, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        
    def forward(self, x):
        """
        Forward pass through VQ-VAE.
        
        Args:
            x: Input images (B, C, H, W)
            
        Returns:
            x_recon: Reconstructed images (B, C, H, W)
            vq_loss: Vector quantization loss
            encoding_indices: Discrete codes (B, H, W)
        """
        z = self.encoder(x)
        quantized, vq_loss, encoding_indices = self.vq(z)
        x_recon = self.decoder(quantized)
        return x_recon, vq_loss, encoding_indices 

def build_vqvae_from_config(arch_params, vqvae_params, dataset="mnist"):
    """
    Build VQ-VAE model from configuration.
    
    Args:
        arch_params: VAEArchParams with encoder/decoder config
        vqvae_params: VQVAEParams with VQ-specific config
        dataset: "mnist", "cifar", or "celeba"
        
    Returns:
        VQVAE model
    """
    
    if dataset == "mnist":
        encoder = Encoder_MNIST_VQVAE(
            arch_params.in_channels,
            arch_params.hidden_channels,
            vqvae_params.embedding_dim
        )
        decoder = Decoder_MNIST_VQVAE(
            arch_params.out_channels,
            arch_params.hidden_channels,
            vqvae_params.embedding_dim
        )
    elif dataset in ["cifar", "celeba"]:
        encoder = Encoder_CIFAR_VQVAE(
            arch_params.in_channels,
            arch_params.hidden_channels,
            vqvae_params.embedding_dim,
            num_residual_layers=2
        )
        decoder = Decoder_CIFAR_VQVAE(
            arch_params.out_channels,
            arch_params.hidden_channels,
            vqvae_params.embedding_dim,
            num_residual_layers=2
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Must be 'mnist', 'cifar', or 'celeba'")
    
    return VQVAE(
        encoder=encoder,
        decoder=decoder,
        num_embeddings=vqvae_params.num_embeddings,
        embedding_dim=vqvae_params.embedding_dim,
        commitment_cost=vqvae_params.commitment_cost
    )
