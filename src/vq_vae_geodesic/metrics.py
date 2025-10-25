# metrics.py
import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim_sk
import math

def batch_l1(recon, target):
    return torch.mean(torch.abs(recon - target)).item()

def batch_mse(recon, target):
    return torch.mean((recon - target)**2).item()

def batch_psnr(mse, data_range=1.0):
    # mse scalar
    if mse == 0:
        return float('inf')
    return 10.0 * math.log10((data_range**2) / mse)

def batch_ssim(recon, target):
    # compute average SSIM over batch (works for both grayscale and RGB)
    recon_np = (recon.detach().cpu().numpy())
    target_np = (target.detach().cpu().numpy())
    B = recon_np.shape[0]
    ssim_vals = []
    for i in range(B):
        im1 = recon_np[i]  # Shape: (C, H, W) for both MNIST and CIFAR
        im2 = target_np[i]
        
        # Check if image is RGB (3 channels) or grayscale (1 channel)
        if im1.shape[0] == 3:  # RGB image (CIFAR-10)
            # For RGB, specify channel_axis=0 (channels first format)
            s = ssim_sk(im1, im2, data_range=1.0, channel_axis=0)
        else:  # Grayscale image (MNIST)
            # Squeeze to HxW for grayscale
            im1 = np.squeeze(im1)
            im2 = np.squeeze(im2)
            s = ssim_sk(im1, im2, data_range=1.0)
        
        ssim_vals.append(s)
    return float(np.mean(ssim_vals))

# Optional: wrapper that computes all metrics for a batch
def compute_image_metrics(recon, target):
    l1 = batch_l1(recon, target)
    mse = batch_mse(recon, target)
    psnr = batch_psnr(mse)
    ssim = batch_ssim(recon, target)
    return {"l1": l1, "mse": mse, "psnr": psnr, "ssim": ssim}
