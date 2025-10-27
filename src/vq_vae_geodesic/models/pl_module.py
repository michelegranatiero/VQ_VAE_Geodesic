import torch
import lightning as L
import wandb
from lightning.pytorch.loggers import WandbLogger

from vq_vae_geodesic.models.modules.decoder import Decoder_CIFAR
from vq_vae_geodesic.models.modules.encoder import Encoder_CIFAR
from vq_vae_geodesic.training.losses import vae_loss_bce
from vq_vae_geodesic.data.loaders import get_cifar_loaders


class LitVAE(L.LightningModule):
    def __init__(self, encoder, decoder, loss_fn, variational_beta=1.0, lr=1e-3):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.loss_fn = loss_fn
        self.variational_beta = variational_beta
        self.lr = lr

        # self.save_hyperparameters()

    def forward(self, x):
        latent_mu, latent_logvar = self.encoder(x)
        latent = self.latent_sample(latent_mu, latent_logvar)
        x_recon = self.decoder(latent)
        return x_recon, latent_mu, latent_logvar

    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(params=self.parameters(), lr=self.lr)

    def _common_step(self, batch, batch_idx):
        x, y = batch

        y = x  # For autoencoders, target is the input itself

        # Forward pass
        image_batch_recon, latent_mu, latent_logvar = self.forward(x)

        # Compute loss
        loss = self.loss_fn(image_batch_recon, y, latent_mu, latent_logvar,
                            beta=self.variational_beta)
        return loss

    def latent_sample(self, mu, logvar):

        if self.training:
            # Convert the logvar to std
            std = (logvar * 0.5).exp()

            # the reparameterization trick
            return torch.distributions.Normal(loc=mu, scale=std).rsample()
        else:
            return mu

    # Optional: Implement any additional methods or hooks as needed

    # def on_train_epoch_end(self):
    #   # Optional: Log epoch-level metrics or perform actions at the end of each training epoch
    #   pass

    # def backward(self, loss, optimizer, optimizer_idx):
    #   # Optional: Customize the backward pass if needed
    #   loss.backward()


if __name__ == "__main__":
    # Quick test
    L.seed_everything(42)
    run_name = "test_run1"

    encoder = Encoder_CIFAR(in_channels=3, hidden_channels=64, latent_dim=2)
    decoder = Decoder_CIFAR(out_channels=3, hidden_channels=64, latent_dim=2)
    train_loader, val_loader, test_loader = get_cifar_loaders(batch_size=64, num_workers=2)
    litVae = LitVAE(encoder, decoder, loss_fn=vae_loss_bce)
    wandb_logger = WandbLogger(project='wandb-lightning', name=run_name)

    trainer = L.Trainer(max_epochs=1, accelerator="gpu", logger=wandb_logger)
    trainer.fit(model=litVae, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(model=litVae, dataloaders=test_loader)
    wandb.finish()
