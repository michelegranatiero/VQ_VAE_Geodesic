"""
PixelCNN model for learning autoregressive prior over discrete latent codes.

Implements a PixelCNN with masked convolutions. After training on quantized
latent codes from a VAE/VQ-VAE, it can generate new valid latents codes (indices)
through ancestral sampling.

"""
from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedConv2d(nn.Conv2d):
    """
    Masked 2D convolution for autoregressive modeling.
    
    Ensures that each position can only depend on previous positions
    in raster scan order (top-to-bottom, left-to-right).
    
    Args:
        mask_type: 'A' for first layer (can't see current pixel)
                   'B' for subsequent layers (can see current pixel)
    """
    def __init__(self, in_channels, out_channels, kernel_size, mask_type: Literal["A", "B"] = "B", stride=1, padding=0, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        if mask_type not in ("A", "B"):
            raise ValueError("mask_type must be 'A' or 'B'")
        
        kh, kw = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        yc, xc = kh // 2, kw // 2
        
        # Create mask of ones and zero-out "future" pixels
        mask = torch.ones_like(self.weight.data)
        # Zero rows below center
        mask[:, :, yc+1:, :] = 0
        # Zero center and right for mask type A, but only right-of-center for B
        if mask_type == 'A':
            mask[:, :, yc, xc:] = 0   # Center excluded -> can't see itself
        else:
            mask[:, :, yc, xc+1:] = 0  # Center allowed -> can see itself
        
        # Register as buffer so it's moved with .to(device)
        self.register_buffer('mask', mask)

    def forward(self, x):
        # Do not overwrite self.weight.data in place
        masked_weight = self.weight * self.mask
        return F.conv2d(x, masked_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class PixelCNN(nn.Module):
    """
    PixelCNN for autoregressive prior over discrete latent codes.
    """
    
    def __init__(self, num_tokens, embed_dim=64, hidden_channels=128, n_layers=7, kernel_size=7):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd because of the masking"
        pad = kernel_size // 2

        # num_tokens is the number of possible values (discrete codes) that each pixel can take
        self.num_tokens = num_tokens
        self.embed = nn.Embedding(num_tokens, embed_dim)

        # first layer: masked type 'A' (can't see current pixel)
        self.first = MaskedConv2d(in_channels=embed_dim, out_channels=hidden_channels,
                                  kernel_size=kernel_size, padding=pad, mask_type='A')

        # stack of masked type 'B' layers (can see current pixel)
        self.hidden = nn.ModuleList([
            MaskedConv2d(in_channels=hidden_channels, out_channels=hidden_channels,
                         kernel_size=kernel_size, padding=pad, mask_type='B')
            for _ in range(n_layers)
        ])

        # output head: reduce channels -> logits over tokens
        self.out_net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels // 2, num_tokens, kernel_size=1)
        )

        # init
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, MaskedConv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0.0, 0.05)

    def forward(self, x):
        """
        Forward pass: compute logits for each position.
        
        Args:
            x: Tensor of discrete codes (B, H, W) with values in [0, num_tokens)
            
        Returns:
            logits: Tensor (B, num_tokens, H, W) with unnormalized log probabilities
        """
        # B, H, W = x.shape
        emb = self.embed(x).permute(0, 3, 1, 2)  # (B, embed_dim, H, W)
        h = self.first(emb)
        h = F.relu(h)
        for layer in self.hidden:
            h = layer(h)
            h = F.relu(h)
        logits = self.out_net(h)  # (B, K, H, W)
        return logits


def build_pixelcnn_from_config(config, for_vqvae=False):
    """
    Build PixelCNN model from experiment configuration.
    
    Args:
        config: ExperimentConfig with pixelcnn_params
        for_vqvae: If True, build for VQ-VAE (uses vqvae_params and pixelcnn_vqvae_params).
                   If False, build for Geodesic (uses quant_params and pixelcnn_params).
        
    Returns:
        PixelCNN model
    """
    if for_vqvae:
        params = config.pixelcnn_vqvae_params
        num_tokens = config.vqvae_params.num_embeddings
    else:
        params = config.pixelcnn_params
        num_tokens = config.quant_params.n_codewords
    
    return PixelCNN(
        num_tokens=num_tokens,
        embed_dim=params.embed_dim,
        hidden_channels=params.hidden_channels,
        n_layers=params.n_layers,
        kernel_size=params.kernel_size
    )


if __name__ == "__main__":
    # Quick test
    model = PixelCNN(num_tokens=256, embed_dim=64, hidden_channels=128, n_layers=7)
    x = torch.randint(0, 256, (4, 8, 8))  # (B, H, W)
    logits = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")