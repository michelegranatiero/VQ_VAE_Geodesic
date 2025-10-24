import torch.nn as nn
import torch
from typing import Tuple


# class Encoder_CIFAR(nn.Module):
#     def __init__(self, in_channels=3, hidden_channels=128, latent_dim=32):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv2d(in_channels=in_channels,
#                       out_channels=hidden_channels,
#                       kernel_size=4,
#                       stride=2,
#                       padding=1),
#             nn.BatchNorm2d(num_features=hidden_channels),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=hidden_channels,
#                       out_channels=hidden_channels*2,
#                       kernel_size=4,
#                       stride=2,
#                       padding=1),
#             nn.BatchNorm2d(num_features=hidden_channels*2),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=hidden_channels*2,
#                       out_channels=hidden_channels*4,
#                       kernel_size=4,
#                       stride=2,
#                       padding=1),
#             nn.BatchNorm2d(num_features=hidden_channels*4),
#             nn.ReLU(),
#             nn.Flatten()
#         )
#         self.fc_mu = nn.Linear(in_features=hidden_channels*4*4*4,
#                                out_features=latent_dim)
#         self.fc_logvar = nn.Linear(in_features=hidden_channels*4*4*4,
#                                    out_features=latent_dim)

#     def forward(self, x):
#         x = self.net(x)
#         mu = self.fc_mu(x)
#         logvar = self.fc_logvar(x)
#         return mu, logvar


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
