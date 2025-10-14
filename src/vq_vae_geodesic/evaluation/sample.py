"""
Sampling functions for PixelCNN autoregressive prior.

Generates new discrete latent code grids by ancestral sampling from
a trained PixelCNN model.
"""
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def sample_pixelcnn(model, device, img_size=(2, 4), temperature=1.0, 
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


def sample_pixelcnn_with_decoder(pixelcnn_model, decoder_model, device,
                                  img_size=(2, 4), temperature=1.0,
                                  batch_size=1, n_chunks=8, chunk_size=4):
    """
    Sample complete images by combining PixelCNN prior and VAE decoder.
    
    Pipeline:
    1. Sample discrete code grid from PixelCNN
    2. Lookup codes in codebook to get continuous latents
    3. Decode latents to images
    
    Args:
        pixelcnn_model: Trained PixelCNN
        decoder_model: VAE decoder
        device: Device to run on
        img_size: (H, W) code grid size
        temperature: Sampling temperature
        batch_size: Number of images to generate
        n_chunks: Number of chunks in latent
        chunk_size: Size of each chunk
        
    Returns:
        images: Generated images tensor (batch_size, C, H_img, W_img)
        codes: Sampled discrete codes (batch_size, H, W)
    """
    from vq_vae_geodesic.config import latents_dir

    # Sample codes from PixelCNN
    codes = sample_pixelcnn(
        pixelcnn_model,
        device=device,
        img_size=img_size,
        temperature=temperature,
        batch_size=batch_size,
        progress=True
    )

    # Load codebook (torch tensor)
    codebook_path = latents_dir() / "chunk_codebook.pt"
    codebook_data = torch.load(codebook_path, map_location=device)
    codebook_chunks = codebook_data['codebook_chunks']

    # Convert codes to continuous latents
    codes_flat = codes.reshape(batch_size, -1)  # (B, H*W)
    latents_list = []

    for b in range(batch_size):
        code_indices = codes_flat[b]  # (H*W,)
        chunks = codebook_chunks[torch.from_numpy(code_indices).long()]  # (H*W, chunk_size)
        latent = chunks.flatten()  # (D,)
        latents_list.append(latent.cpu().numpy())

    latents = np.stack(latents_list)  # (B, D)
    latents_t = torch.from_numpy(latents).float().to(device)

    # Decode to images
    decoder_model.eval()
    with torch.no_grad():
        images = decoder_model(latents_t)

    return images, codes


def interpolate_codes(code1, code2, n_steps=10):
    """
    Interpolate between two discrete code grids.
    
    Since codes are discrete, this performs a "hard" interpolation by
    gradually replacing code1 with code2 position by position.
    
    Args:
        code1: First code grid (H, W)
        code2: Second code grid (H, W)
        n_steps: Number of interpolation steps
        
    Returns:
        interpolated: List of code grids interpolating from code1 to code2
    """
    H, W = code1.shape
    total_positions = H * W
    
    # Flatten codes
    flat1 = code1.flatten()
    flat2 = code2.flatten()
    
    interpolated = []
    for step in range(n_steps):
        # Fraction of positions to replace
        frac = step / (n_steps - 1)
        n_replace = int(frac * total_positions)
        
        # Start with code1, replace first n_replace positions with code2
        result = flat1.copy()
        result[:n_replace] = flat2[:n_replace]
        
        interpolated.append(result.reshape(H, W))
    
    return interpolated


if __name__ == "__main__":
    # Quick test
    from vq_vae_geodesic.models.modules import PixelCNN
    
    model = PixelCNN(num_tokens=256, embed_dim=64, hidden_channels=128, n_layers=7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    samples = sample_pixelcnn(model, device, img_size=(2, 4), temperature=1.0, batch_size=4)
    print(f"Sampled codes shape: {samples.shape}")
    print(f"Sample 0:\n{samples[0]}")
