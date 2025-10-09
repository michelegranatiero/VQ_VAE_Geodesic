import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from vq_vae_geodesic.utils import make_averager, refresh_bar


def test(model, test_loader, loss_fn, device, variational_beta):
    model.eval()
    test_loss_averager = make_averager()
    with torch.no_grad():
        test_bar = tqdm(test_loader, total=len(test_loader), desc='Testing')
        for image_batch, target_batch in test_bar:
            image_batch = image_batch.to(device)
            target_batch = image_batch  # target is the original image in VAE
            recon_batch, mu, logvar = model(image_batch)
            loss = loss_fn(recon_batch, target_batch, mu, logvar, beta=variational_beta)
            test_loss_averager(loss.item())
            refresh_bar(test_bar, f"test batch [loss: {test_loss_averager():.3f}]")
    print(f"Test set: Average loss: {test_loss_averager():.4f}")
    return test_loss_averager()


def plot_reconstructions(model, test_loader, device, n=8):
    model.eval()
    with torch.no_grad():
        corrupted, target = next(iter(test_loader))
        corrupted = corrupted.to(device)
        recon = model(corrupted)[0].cpu()
        plt.figure(figsize=(15, 4))
        for i in range(n):
            plt.subplot(3, n, i+1)
            plt.imshow(np.transpose(target[i].numpy(), (1, 2, 0)))
            plt.axis('off')
            if i == 0:
                plt.ylabel("Original")
            plt.subplot(3, n, i+1+n)
            plt.imshow(np.transpose(corrupted[i].cpu().numpy(), (1, 2, 0)))
            plt.axis('off')
            if i == 0:
                plt.ylabel("Corrupted")
            plt.subplot(3, n, i+1+2*n)
            plt.imshow(np.transpose(recon[i].numpy(), (1, 2, 0)))
            plt.axis('off')
            if i == 0:
                plt.ylabel("Reconstr.")
        plt.show()


def plot_reconstructions_mnist(model, test_loader, device, n=8):
    model.eval()
    with torch.no_grad():
        images, _ = next(iter(test_loader))
        images = images.to(device)
        recon = model(images)[0].cpu()
        plt.figure(figsize=(2 * n, 4))
        for i in range(n):
            # Originale
            plt.subplot(2, n, i + 1)
            plt.imshow(images[i].cpu().squeeze(), cmap="gray")
            plt.axis('off')
            if i == 0:
                plt.ylabel("Original")
            # Ricostruzione
            plt.subplot(2, n, i + 1 + n)
            plt.imshow(recon[i].squeeze(), cmap="gray")
            plt.axis('off')
            if i == 0:
                plt.ylabel("Reconstr.")
        plt.show()


if __name__ == "__main__":
    import torch
    from vq_vae_geodesic.models.vae import VariationalAutoencoder
    from vq_vae_geodesic.loaders.loaders import get_corrupted_cifar_loaders

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vae = VariationalAutoencoder(hidden_channels=64, latent_dim=2)  # usa i parametri giusti!
    vae.load_state_dict(torch.load("data/vae_cifar.pth", map_location=device))
    vae.to(device)
    _, test_loader = get_corrupted_cifar_loaders(batch_size=128, num_workers=2)
    plot_reconstructions(vae, test_loader, device)
