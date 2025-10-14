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
