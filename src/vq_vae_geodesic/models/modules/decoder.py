import torch
import torch.nn as nn
import torch


# class Decoder_CIFAR(nn.Module):
#     def __init__(self, out_channels=3, hidden_channels=128, latent_dim=32):
#         super().__init__()
#         self.fc = nn.Linear(in_features=latent_dim, out_features=hidden_channels*4*4*4)
#         self.net = nn.Sequential(
#             nn.Unflatten(1, (hidden_channels*4, 4, 4)),
#             nn.ConvTranspose2d(in_channels=hidden_channels*4,
#                                out_channels=hidden_channels * 2,
#                                kernel_size=4,
#                                stride=2,
#                                padding=1),
#             nn.BatchNorm2d(num_features=hidden_channels*2),
#             nn.ReLU(),
#             nn.ConvTranspose2d(in_channels=hidden_channels*2,
#                                out_channels=hidden_channels,
#                                kernel_size=4,
#                                stride=2,
#                                padding=1),
#             nn.BatchNorm2d(num_features=hidden_channels),
#             nn.ReLU(),
#             nn.ConvTranspose2d(in_channels=hidden_channels,
#                                out_channels=out_channels,
#                                kernel_size=4,
#                                stride=2,
#                                padding=1),
#             nn.Sigmoid()  # Normalize output to [0, 1]
#         )

#     def forward(self, z):
#         x = self.fc(z)
#         x = self.net(x)
#         return x


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
