import torch.nn as nn
import torch
from typing import Tuple


class Encoder_MNIST_VAE(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, latent_dim: int) -> None:
        """
        Encoder module that predicts the mean and log(variance) parameters.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=hidden_channels,
                               kernel_size=4,
                               stride=2,
                               padding=1)  # out: hidden_channels x 14 x 14

        self.conv2 = nn.Conv2d(in_channels=hidden_channels,
                               out_channels=hidden_channels*2,
                               kernel_size=4,
                               stride=2,
                               padding=1)  # out: (hidden_channels x 2) x 7 x 7

        self.fc_mu = nn.Linear(in_features=hidden_channels*2*7*7,
                               out_features=latent_dim)
        self.fc_logvar = nn.Linear(in_features=hidden_channels*2*7*7,
                                   out_features=latent_dim)

        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param x: batch of images with shape [batch, channels, w, h]
        :returns: the predicted mean and log(variance)
        """
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))

        x = x.view(x.shape[0], -1)

        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)

        return x_mu, x_logvar


class Encoder_MNIST_VQVAE(nn.Module):
    """
    Encoder for VQ-VAE that outputs spatial feature maps instead of flat vectors.
    
    Output: (B, embedding_dim, H, W) where H=7, W=7 for MNIST 28x28 input
    """
    def __init__(self, in_channels: int, hidden_channels: int, embedding_dim: int) -> None:
        super().__init__()
        
        # Same as VAE encoder
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=4,
            stride=2,
            padding=1
        )  # out: hidden_channels x 14 x 14
        
        self.conv2 = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=hidden_channels*2,
            kernel_size=4,
            stride=2,
            padding=1
        )  # out: hidden_channels*2 x 7 x 7
        
        # Project to embedding dimension (instead of flatten + fc like VAE)
        self.proj = nn.Conv2d(
            in_channels=hidden_channels*2,
            out_channels=embedding_dim,
            kernel_size=1,
            stride=1,
            padding=0
        )  # out: embedding_dim x 7 x 7
        
        self.activation = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images (B, C, H, W)
            
        Returns:
            Spatial feature map (B, embedding_dim, H', W')
        """
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.proj(x)  # No activation on projection layer
        return x


class Encoder_CIFAR_VAE(nn.Module):
    """
    Encoder for VAE on CIFAR-10 (32x32x3 RGB images).
    """
    def __init__(self, in_channels: int, hidden_channels: int, latent_dim: int, 
                 num_residual_layers: int = 2) -> None:
        super().__init__()
        
        # Encoder convolutions
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_channels // 2,
            kernel_size=4,
            stride=2,
            padding=1
        )  # out: (hidden_channels//2) x 16 x 16
        
        self.conv2 = nn.Conv2d(
            in_channels=hidden_channels // 2,
            out_channels=hidden_channels,
            kernel_size=4,
            stride=2,
            padding=1
        )  # out: hidden_channels x 8 x 8
        
        self.conv3 = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )  # out: hidden_channels x 8 x 8
        
        # Residual stack
        self.residual_stack = nn.ModuleList([
            ResidualBlock(hidden_channels, hidden_channels // 4)
            for _ in range(num_residual_layers)
        ])
        
        # Projection to latent space
        self.fc_mu = nn.Linear(hidden_channels * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(hidden_channels * 8 * 8, latent_dim)
        
        self.activation = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input images (B, 3, 32, 32)
            
        Returns:
            mu: Mean vectors (B, latent_dim)
            logvar: Log-variance vectors (B, latent_dim)
        """
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        
        # Residual stack
        for res_block in self.residual_stack:
            x = res_block(x)
        
        # Flatten and project
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar


class Encoder_CIFAR_VQVAE(nn.Module):
    """
    Encoder for VQ-VAE on CIFAR-10 (32x32x3 RGB images).
    """
    def __init__(self, in_channels: int, hidden_channels: int, embedding_dim: int,
                 num_residual_layers: int = 2) -> None:
        super().__init__()
        
        # Encoder convolutions (same as VAE)
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_channels // 2,
            kernel_size=4,
            stride=2,
            padding=1
        )  # out: (hidden_channels//2) x 16 x 16
        
        self.conv2 = nn.Conv2d(
            in_channels=hidden_channels // 2,
            out_channels=hidden_channels,
            kernel_size=4,
            stride=2,
            padding=1
        )  # out: hidden_channels x 8 x 8
        
        self.conv3 = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )  # out: hidden_channels x 8 x 8
        
        # Residual stack
        self.residual_stack = nn.ModuleList([
            ResidualBlock(hidden_channels, hidden_channels // 4)
            for _ in range(num_residual_layers)
        ])
        
        # Project to embedding dimension
        self.proj = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=embedding_dim,
            kernel_size=1,
            stride=1,
            padding=0
        )  # out: embedding_dim x 8 x 8
        
        self.activation = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images (B, 3, 32, 32)
            
        Returns:
            Spatial feature map (B, embedding_dim, 8, 8)
        """
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        
        # Residual stack
        for res_block in self.residual_stack:
            x = res_block(x)
        
        # Project to embedding space
        x = self.proj(x)
        return x


class ResidualBlock(nn.Module):
    """
    Residual block used in CIFAR-10 encoder/decoder.
    
    Architecture:
    - Conv 3x3 to num_residual_hiddens
    - ReLU
    - Conv 1x1 back to num_hiddens
    - Add residual connection
    """
    def __init__(self, num_hiddens: int, num_residual_hiddens: int) -> None:
        super().__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels=num_hiddens,
            out_channels=num_residual_hiddens,
            kernel_size=3,
            stride=1,
            padding=1
        )
        
        self.conv2 = nn.Conv2d(
            in_channels=num_residual_hiddens,
            out_channels=num_hiddens,
            kernel_size=1,
            stride=1,
            padding=0
        )
        
        self.activation = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, num_hiddens, H, W)
            
        Returns:
            Output tensor with residual connection (B, num_hiddens, H, W)
        """
        residual = x
        x = self.activation(self.conv1(x))
        x = self.conv2(x)
        return x + residual
