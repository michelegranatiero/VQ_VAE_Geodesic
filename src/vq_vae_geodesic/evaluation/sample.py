"""
Sampling functions for PixelCNN autoregressive prior.

Generates new discrete latent code grids by ancestral sampling from
a trained PixelCNN model.
"""
import torch
import torch.nn.functional as F
from tqdm import tqdm


def sample_pixelcnn(model, device, img_size, temperature=1.0, 
                    batch_size=1, progress=True):
    """
    Sample discrete latent codes from trained PixelCNN model.
    
    Uses ancestral sampling in raster scan order (left-to-right, top-to-bottom).
    At each position, the model predicts a distribution over discrete codes
    conditioned on all previous positions.
    
    Args:
        model: Trained PixelCNN model
        device: Device to run on
        img_size: (H, W) grid size to generate
        temperature: Sampling temperature
            - 1.0: sample from model distribution
            - >1.0: more diverse/random samples
            - <1.0: sharper/more confident samples
        batch_size: Number of samples to generate in parallel
        progress: Show progress bar
        
    Returns:
        samples: numpy array (batch_size, H, W) with sampled code indices
    """
    model.eval()
    H, W = img_size
    
    with torch.no_grad():
        # Initialize with zeros (or could use random/uniform prior)
        x = torch.zeros(batch_size, H, W, dtype=torch.long, device=device)
        
        # Iterate over each position in raster order
        positions = [(i, j) for i in range(H) for j in range(W)]
        iterator = tqdm(positions, desc="Sampling codes") if progress else positions
        
        for i, j in iterator:
            # Get logits for all positions
            logits = model(x)  # (B, K, H, W)
            
            # Extract logits for current position
            logits_ij = logits[:, :, i, j]  # (B, K)
            
            # Apply temperature
            logits_ij = logits_ij / max(temperature, 1e-8)
            
            # Sample from categorical distribution
            probs = F.softmax(logits_ij, dim=1)  # (B, K)
            samples = torch.multinomial(probs, num_samples=1).squeeze(1)  # (B,)
            
            # Update grid
            x[:, i, j] = samples
    
    return x.cpu().numpy()


if __name__ == "__main__":
    # Quick test
    from vq_vae_geodesic.models.modules import PixelCNN
    
    model = PixelCNN(num_tokens=256, embed_dim=64, hidden_channels=128, n_layers=7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    samples = sample_pixelcnn(model, device, img_size=(2, 4), temperature=1.0, batch_size=4)
    print(f"Sampled codes shape: {samples.shape}")
    print(f"Sample 0:\n{samples[0]}")
