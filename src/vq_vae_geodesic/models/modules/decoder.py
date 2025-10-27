import torch
import torch.nn as nn
import torch

class Decoder_MNIST_VAE(nn.Module):
    def __init__(self, out_channels: int, hidden_channels: int, latent_dim: int) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels

        self.fc = nn.Linear(in_features=latent_dim,
                            out_features=hidden_channels*2*7*7)

        self.conv1 = nn.ConvTranspose2d(in_channels=hidden_channels*2, # in 
                                        out_channels=hidden_channels,
                                        kernel_size=4,
                                        stride=2,
                                        padding=1) # out: hidden_channels x 14 x 14
        self.conv2 = nn.ConvTranspose2d(in_channels=hidden_channels,
                                        out_channels=out_channels,
                                        kernel_size=4,
                                        stride=2,
                                        padding=1) # out: out_channels x 28 x 28

        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: a sample from the distribution governed by the mean and log(var)
        :returns: a reconstructed image with size [batch, 1, w, h]
        """
        x = self.fc(x)
        x = x.view(x.size(0), self.hidden_channels*2, 7, 7) # reshape as feature map (B, C, H, W)
        x = self.activation(self.conv1(x))
        # Last layer before output is sigmoid, since we are using BCE as reconstruction loss
        x = torch.sigmoid(self.conv2(x))
        return x


class Decoder_MNIST_VQVAE(nn.Module):
    """
    Decoder for VQ-VAE that takes spatial feature maps as input.
    
    Input: (B, embedding_dim, H, W) where H=7, W=7
    Output: (B, out_channels, 28, 28) reconstructed image
    """
    def __init__(self, out_channels: int, hidden_channels: int, embedding_dim: int) -> None:
        super().__init__()
        
        # Project from embedding_dim to hidden_channels*2 (instead of fc + unflatten like VAE)
        self.proj = nn.Conv2d(
            in_channels=embedding_dim,
            out_channels=hidden_channels*2,
            kernel_size=1,
            stride=1,
            padding=0
        )  # out: hidden_channels*2 x 7 x 7
        
        # Same as VAE decoder
        self.conv1 = nn.ConvTranspose2d(
            in_channels=hidden_channels*2,
            out_channels=hidden_channels,
            kernel_size=4,
            stride=2,
            padding=1
        )  # out: hidden_channels x 14 x 14
        
        self.conv2 = nn.ConvTranspose2d(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=4,
            stride=2,
            padding=1
        )  # out: out_channels x 28 x 28
        
        self.activation = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Quantized latents (B, embedding_dim, H, W)
            
        Returns:
            Reconstructed image (B, out_channels, 28, 28)
        """
        x = self.proj(x)  # Project to hidden space
        x = self.activation(self.conv1(x))
        x = torch.sigmoid(self.conv2(x))  # Normalize to [0, 1]
        return x


class Decoder_CIFAR_VAE(nn.Module):
    def __init__(self, out_channels: int, hidden_channels: int, latent_dim: int,
                 num_residual_layers: int = 2) -> None:
        super().__init__()
        
        self.hidden_channels = hidden_channels
        
        # Project from latent to spatial
        self.fc = nn.Linear(latent_dim, hidden_channels * 8 * 8)
        
        # Initial conv to process unflattened features
        self.conv_init = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        
        # Residual stack
        self.residual_stack = nn.ModuleList([
            ResidualBlock(hidden_channels, hidden_channels // 4)
            for _ in range(num_residual_layers)
        ])
        
        # Decoder convolutions
        self.conv1 = nn.ConvTranspose2d(
            in_channels=hidden_channels,
            out_channels=hidden_channels // 2,
            kernel_size=4,
            stride=2,
            padding=1
        )  # out: (hidden_channels//2) x 16 x 16
        
        self.conv2 = nn.ConvTranspose2d(
            in_channels=hidden_channels // 2,
            out_channels=out_channels,
            kernel_size=4,
            stride=2,
            padding=1
        )  # out: out_channels x 32 x 32
        
        self.activation = nn.ReLU()
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Latent vectors (B, latent_dim)
            
        Returns:
            Reconstructed images (B, 3, 32, 32) in [0, 1]
        """
        x = self.fc(z)
        x = x.view(x.size(0), self.hidden_channels, 8, 8)
        x = self.activation(self.conv_init(x))
        
        # Residual stack
        for res_block in self.residual_stack:
            x = res_block(x)
        x = self.activation(x)
        
        # Upsample
        x = self.activation(self.conv1(x))
        x = torch.sigmoid(self.conv2(x))  # Output in [0, 1]
        
        return x


class Decoder_CIFAR_VQVAE(nn.Module):
    def __init__(self, out_channels: int, hidden_channels: int, embedding_dim: int,
                 num_residual_layers: int = 2) -> None:
        super().__init__()
        
        # Project from embedding to hidden space
        self.proj = nn.Conv2d(
            in_channels=embedding_dim,
            out_channels=hidden_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        
        # Initial conv
        self.conv_init = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        
        # Residual stack
        self.residual_stack = nn.ModuleList([
            ResidualBlock(hidden_channels, hidden_channels // 4)
            for _ in range(num_residual_layers)
        ])
        
        # Decoder convolutions (same as VAE)
        self.conv1 = nn.ConvTranspose2d(
            in_channels=hidden_channels,
            out_channels=hidden_channels // 2,
            kernel_size=4,
            stride=2,
            padding=1
        )  # out: (hidden_channels//2) x 16 x 16
        
        self.conv2 = nn.ConvTranspose2d(
            in_channels=hidden_channels // 2,
            out_channels=out_channels,
            kernel_size=4,
            stride=2,
            padding=1
        )  # out: out_channels x 32 x 32
        
        self.activation = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Quantized latents (B, embedding_dim, 8, 8)
            
        Returns:
            Reconstructed images (B, 3, 32, 32) in [0, 1]
        """
        x = self.proj(x)
        x = self.activation(self.conv_init(x))
        
        # Residual stack
        for res_block in self.residual_stack:
            x = res_block(x)
        x = self.activation(x)
        
        # Upsample
        x = self.activation(self.conv1(x))
        x = torch.sigmoid(self.conv2(x))  # Output in [0, 1]
        
        return x


# Import ResidualBlock from encoder (defined there to avoid duplication)
from .encoder import ResidualBlock
