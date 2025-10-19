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
    # compute average SSIM over batch (grayscale MNIST)
    recon_np = (recon.detach().cpu().numpy())
    target_np = (target.detach().cpu().numpy())
    B = recon_np.shape[0]
    ssim_vals = []
    for i in range(B):
        im1 = np.squeeze(recon_np[i])
        im2 = np.squeeze(target_np[i])
        # skimage expects HxW, floats 0..1
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
