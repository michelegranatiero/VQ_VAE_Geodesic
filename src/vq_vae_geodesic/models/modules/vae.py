import torch
import torch.nn as nn
from vq_vae_geodesic.models.modules.encoder import Encoder_MNIST
from vq_vae_geodesic.models.modules.decoder import Decoder_MNIST

class VariationalAutoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        latent_mu, latent_logvar = self.encoder(x)
        latent = self.latent_sample(latent_mu, latent_logvar)
        x_recon = self.decoder(latent)
        return x_recon, latent_mu, latent_logvar

    def latent_sample(self, mu, logvar):

        if self.training:
            # Convert the logvar to std
            std = (logvar * 0.5).exp()
            std = torch.clamp(std, min=1e-6)  # Avoid too small std values

            # Reparameterization trick
            return torch.distributions.Normal(loc=mu, scale=std).rsample()
        else:
            return mu


def build_vae_from_config(arch_params):

    encoder = Encoder_MNIST(
        arch_params.in_channels,
        arch_params.hidden_channels,
        arch_params.latent_dim
    )
    decoder = Decoder_MNIST(
        arch_params.in_channels,
        arch_params.hidden_channels,
        arch_params.latent_dim
    )

    return VariationalAutoencoder(encoder, decoder)


if __name__ == "__main__":
    # Quick test
    x = torch.randn(1, 1, 28, 28)
    encoder = Encoder_MNIST(in_channels=1, hidden_channels=64, latent_dim=16)
    decoder = Decoder_MNIST(out_channels=1, hidden_channels=64, latent_dim=16)
    vae = VariationalAutoencoder(encoder, decoder)
    x_recon, mu, logvar = vae(x)
    print(x_recon.shape)
    print(mu.shape)
    print(logvar.shape)
