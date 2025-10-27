import torch
from tqdm import tqdm
from vq_vae_geodesic.training.losses import vae_loss_bce, vqvae_loss_bce, vae_loss_mse, vqvae_loss_mse
from vq_vae_geodesic.metrics import compute_image_metrics
from vq_vae_geodesic.evaluation.utils import lookup_codewords
from vq_vae_geodesic.utils import WeightedAverager
import torch.nn.functional as F


def evaluate_vae_mnist(model, data_loader, device, beta=1.0):
    """ VAE only. No Geodesic Quantization"""
    model.eval()
    loss_avg = WeightedAverager()
    recon_loss_avg = WeightedAverager()
    l1_avg = WeightedAverager()
    mse_avg = WeightedAverager()
    psnr_avg = WeightedAverager()
    ssim_avg = WeightedAverager()
    with torch.no_grad():
        for image_batch, _ in tqdm(data_loader, desc="Evaluating VAE"):
            image_batch = image_batch.to(device)
            recon, mu, logvar = model(image_batch)
            loss, recon_loss = vae_loss_bce(recon, image_batch, mu, logvar, beta=beta)
            batch_size = image_batch.size(0)
            # Averagers
            loss_avg.add(loss.item(), batch_size)
            recon_loss_avg.add(recon_loss.item(), batch_size)
            # Metrics
            batch_metrics = compute_image_metrics(recon.detach().cpu(), image_batch.cpu())
            l1_avg.add(batch_metrics['l1'], batch_size)
            mse_avg.add(batch_metrics['mse'], batch_size)
            psnr_avg.add(batch_metrics['psnr'], batch_size)
            ssim_avg.add(batch_metrics['ssim'], batch_size)
    metrics_mean = {
        'loss': loss_avg.mean(),
        'recon_loss': recon_loss_avg.mean(),
        'l1': l1_avg.mean(),
        'mse': mse_avg.mean(),
        'psnr': psnr_avg.mean(),
        'ssim': ssim_avg.mean()
    }
    return metrics_mean

def evaluate_vae_geodesic_mnist(model, data_loader, codebook_chunks: torch.Tensor, codes_per_image: torch.Tensor, device, beta=1.0):
    model.eval()
    recon_loss_avg = WeightedAverager()
    l1_avg = WeightedAverager()
    mse_avg = WeightedAverager()
    psnr_avg = WeightedAverager()
    ssim_avg = WeightedAverager()
    ptr = 0
    codebook_chunks = codebook_chunks.to(device)
    codes_per_image = codes_per_image.to(device)
    with torch.no_grad():
        for image_batch, _ in tqdm(data_loader, desc="Evaluating Geodesic"):
            bs = image_batch.size(0)
            image_batch = image_batch.to(device)
            batch_codes = codes_per_image[ptr:ptr+bs]
            ptr += bs
            # Get actual codewords from codebook
            z_recon = lookup_codewords(codebook_chunks, batch_codes)
            recon = model.decoder(z_recon)
            # Use vqvae loss since no KL term
            recon_loss = vqvae_loss_bce(recon, image_batch)
            # Averagers
            recon_loss_avg.add(recon_loss.item(), bs)
            # Metrics
            batch_metrics = compute_image_metrics(recon.detach().cpu(), image_batch.cpu())
            l1_avg.add(batch_metrics['l1'], bs)
            mse_avg.add(batch_metrics['mse'], bs)
            psnr_avg.add(batch_metrics['psnr'], bs)
            ssim_avg.add(batch_metrics['ssim'], bs)
    metrics_mean = {
        'recon_loss': recon_loss_avg.mean(),
        'l1': l1_avg.mean(),
        'mse': mse_avg.mean(),
        'psnr': psnr_avg.mean(),
        'ssim': ssim_avg.mean()
    }
    return metrics_mean

def evaluate_vqvae_mnist(model, data_loader, device):
    model.eval()
    loss_avg = WeightedAverager()
    recon_loss_avg = WeightedAverager()
    l1_avg = WeightedAverager()
    mse_avg = WeightedAverager()
    psnr_avg = WeightedAverager()
    ssim_avg = WeightedAverager()
    with torch.no_grad():
        for image_batch, _ in tqdm(data_loader, desc="Evaluating VQ-VAE"):
            image_batch = image_batch.to(device)
            x_recon, vq_loss, _ = model(image_batch)
            recon_loss = vqvae_loss_bce(x_recon, image_batch)
            loss = recon_loss + vq_loss
            batch_size = image_batch.size(0)
            # Averagers
            loss_avg.add(loss.item(), batch_size)
            recon_loss_avg.add(recon_loss.item(), batch_size)
            # Metrics
            batch_metrics = compute_image_metrics(x_recon.detach().cpu(), image_batch.cpu())
            l1_avg.add(batch_metrics['l1'], batch_size)
            mse_avg.add(batch_metrics['mse'], batch_size)
            psnr_avg.add(batch_metrics['psnr'], batch_size)
            ssim_avg.add(batch_metrics['ssim'], batch_size)
    metrics_mean = {
        'loss': loss_avg.mean(),
        'recon_loss': recon_loss_avg.mean(),
        'l1': l1_avg.mean(),
        'mse': mse_avg.mean(),
        'psnr': psnr_avg.mean(),
        'ssim': ssim_avg.mean()
    }
    return metrics_mean



def evaluate_vae_celeba(model, data_loader, device, beta=1.0):
    """ VAE evaluation for CelebA/CIFAR-10 using MSE loss """
    model.eval()
    loss_avg = WeightedAverager()
    recon_loss_avg = WeightedAverager()
    l1_avg = WeightedAverager()
    mse_avg = WeightedAverager()
    psnr_avg = WeightedAverager()
    ssim_avg = WeightedAverager()
    with torch.no_grad():
        for image_batch, _ in tqdm(data_loader, desc="Evaluating VAE"):
            image_batch = image_batch.to(device)
            recon, mu, logvar = model(image_batch)
            loss, recon_loss = vae_loss_mse(recon, image_batch, mu, logvar, beta=beta)
            batch_size = image_batch.size(0)
            # Averagers
            loss_avg.add(loss.item(), batch_size)
            recon_loss_avg.add(recon_loss.item(), batch_size)
            # Metrics
            batch_metrics = compute_image_metrics(recon.detach().cpu(), image_batch.cpu())
            l1_avg.add(batch_metrics['l1'], batch_size)
            mse_avg.add(batch_metrics['mse'], batch_size)
            psnr_avg.add(batch_metrics['psnr'], batch_size)
            ssim_avg.add(batch_metrics['ssim'], batch_size)
    metrics_mean = {
        'loss': loss_avg.mean(),
        'recon_loss': recon_loss_avg.mean(),
        'l1': l1_avg.mean(),
        'mse': mse_avg.mean(),
        'psnr': psnr_avg.mean(),
        'ssim': ssim_avg.mean()
    }
    return metrics_mean


def evaluate_vae_geodesic_celeba(model, data_loader, codebook_chunks: torch.Tensor, codes_per_image: torch.Tensor, device, beta=1.0):
    """ VAE + Geodesic evaluation for CelebA/CIFAR-10 using MSE loss """
    model.eval()
    recon_loss_avg = WeightedAverager()
    l1_avg = WeightedAverager()
    mse_avg = WeightedAverager()
    psnr_avg = WeightedAverager()
    ssim_avg = WeightedAverager()
    ptr = 0
    codebook_chunks = codebook_chunks.to(device)
    codes_per_image = codes_per_image.to(device)
    with torch.no_grad():
        for image_batch, _ in tqdm(data_loader, desc="Evaluating Geodesic"):
            bs = image_batch.size(0)
            image_batch = image_batch.to(device)
            batch_codes = codes_per_image[ptr:ptr+bs]
            ptr += bs
            # Get actual codewords from codebook
            z_recon = lookup_codewords(codebook_chunks, batch_codes)
            recon = model.decoder(z_recon)
            # Use vqvae MSE loss since no KL term
            recon_loss = vqvae_loss_mse(recon, image_batch)
            # Averagers
            recon_loss_avg.add(recon_loss.item(), bs)
            # Metrics
            batch_metrics = compute_image_metrics(recon.detach().cpu(), image_batch.cpu())
            l1_avg.add(batch_metrics['l1'], bs)
            mse_avg.add(batch_metrics['mse'], bs)
            psnr_avg.add(batch_metrics['psnr'], bs)
            ssim_avg.add(batch_metrics['ssim'], bs)
    metrics_mean = {
        'recon_loss': recon_loss_avg.mean(),
        'l1': l1_avg.mean(),
        'mse': mse_avg.mean(),
        'psnr': psnr_avg.mean(),
        'ssim': ssim_avg.mean()
    }
    return metrics_mean


def evaluate_vqvae_celeba(model, data_loader, device):
    """ VQ-VAE evaluation for CelebA/CIFAR-10 using MSE loss """
    model.eval()
    loss_avg = WeightedAverager()
    recon_loss_avg = WeightedAverager()
    l1_avg = WeightedAverager()
    mse_avg = WeightedAverager()
    psnr_avg = WeightedAverager()
    ssim_avg = WeightedAverager()
    with torch.no_grad():
        for image_batch, _ in tqdm(data_loader, desc="Evaluating VQ-VAE"):
            image_batch = image_batch.to(device)
            x_recon, vq_loss, _ = model(image_batch)
            recon_loss = vqvae_loss_mse(x_recon, image_batch)
            loss = recon_loss + vq_loss
            batch_size = image_batch.size(0)
            # Averagers
            loss_avg.add(loss.item(), batch_size)
            recon_loss_avg.add(recon_loss.item(), batch_size)
            # Metrics
            batch_metrics = compute_image_metrics(x_recon.detach().cpu(), image_batch.cpu())
            l1_avg.add(batch_metrics['l1'], batch_size)
            mse_avg.add(batch_metrics['mse'], batch_size)
            psnr_avg.add(batch_metrics['psnr'], batch_size)
            ssim_avg.add(batch_metrics['ssim'], batch_size)
    metrics_mean = {
        'loss': loss_avg.mean(),
        'recon_loss': recon_loss_avg.mean(),
        'l1': l1_avg.mean(),
        'mse': mse_avg.mean(),
        'psnr': psnr_avg.mean(),
        'ssim': ssim_avg.mean()
    }
    return metrics_mean



def evaluate_pixelcnn_geodesic_mnist(model, data_loader, device):
    """Evaluate PixelCNN (Geodesic) on discrete codes."""
    model.eval()
    loss_avg = WeightedAverager()
    with torch.no_grad():
        bar = tqdm(data_loader, desc="Evaluating PixelCNN (geodesic)")
        for codes_batch in bar:
            # DataLoader for codes may return just the tensor or (tensor, _)
            if isinstance(codes_batch, (list, tuple)):
                codes = codes_batch[0].to(device)
            else:
                codes = codes_batch.to(device)

            logits = model(codes)  # (B, K, H, W)
            # F.cross_entropy expects (B, C, H, W) logits and (B, H, W) targets
            loss = F.cross_entropy(logits, codes).item()
            loss_avg.add(loss, codes.size(0))

    mean_loss = loss_avg.mean()
    perplexity = float(torch.exp(torch.tensor(mean_loss)))
    num_embeddings = model.num_tokens
    perplexity_pct = (perplexity / num_embeddings) * 100
    return {'loss': mean_loss, 'perplexity': perplexity, 'perplexity_pct': perplexity_pct}


def evaluate_pixelcnn_vqvae_mnist(model, data_loader, device):
    """Evaluate PixelCNN (VQ-VAE) on discrete codes."""
    model.eval()
    loss_avg = WeightedAverager()
    with torch.no_grad():
        bar = tqdm(data_loader, desc="Evaluating PixelCNN (vqvae)")
        for codes_batch in bar:
            if isinstance(codes_batch, (list, tuple)):
                codes = codes_batch[0].to(device)
            else:
                codes = codes_batch.to(device)

            logits = model(codes)
            loss = F.cross_entropy(logits, codes).item()
            loss_avg.add(loss, codes.size(0))

    mean_loss = loss_avg.mean()
    perplexity = float(torch.exp(torch.tensor(mean_loss)))
    num_embeddings = model.num_tokens
    perplexity_pct = (perplexity / num_embeddings) * 100
    return {'loss': mean_loss, 'perplexity': perplexity, 'perplexity_pct': perplexity_pct}


# Aliases for CIFAR-10 (same implementation, just different names for clarity)
evaluate_pixelcnn_geodesic_cifar10 = evaluate_pixelcnn_geodesic_mnist
evaluate_pixelcnn_vqvae_cifar10 = evaluate_pixelcnn_vqvae_mnist
