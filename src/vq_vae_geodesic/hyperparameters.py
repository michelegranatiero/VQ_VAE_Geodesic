"""
Hyperparameters and model configuration.
"""
from dataclasses import dataclass, field
from typing import Dict

from vq_vae_geodesic import config as cfg


@dataclass(unsafe_hash=True)
class VAEArchParams:
    """VAE architecture parameters."""
    in_channels: int = 1  # 1 for MNIST, 3 for CIFAR and CelebA
    out_channels: int = 1  # 1 for MNIST, 3 for CIFAR and CelebA
    hidden_channels: int = 128  # Capacity 
    latent_dim: int = 32  # Size of the latent bottleneck


@dataclass(unsafe_hash=True)
class TrainingParams:
    """Training hyperparameters."""
    num_epochs: int = 100
    batch_size: int = 128
    lr: float = 0.001
    variational_beta: float = 1.0  # Beta parameter for VAE loss
    save_checkpoint_every: int = 1


@dataclass(unsafe_hash=True)
class GeodesicQuantizationParams:
    """Parameters for geodesic quantization."""
    n_chunks: int = 8  # Number of chunks to split latent
    n_codewords: int = 256  # Size of the codebook
    knn_k: int = 20  # Number of neighbors for k-NN graph
    # subsample_max_pts: int = 5000  # Max points for subsampling
    random_state: int = cfg.SEED

    # Grid shape for codes (must satisfy H*W = n_chunks)
    grid_h: int = 4
    grid_w: int = 2

    def __post_init__(self):
        """Validate parameters."""
        if self.grid_h * self.grid_w != self.n_chunks:
            raise ValueError(
                f"grid_h * grid_w must equal n_chunks "
                f"({self.grid_h} * {self.grid_w} != {self.n_chunks})"
            )

    def chunk_size(self, latent_dim: int) -> int:
        """Compute chunk size from latent_dim and n_chunks."""
        if latent_dim % self.n_chunks != 0:
            raise ValueError(
                f"latent_dim must be divisible by n_chunks "
                f"({latent_dim} % {self.n_chunks} != 0)"
            )
        return latent_dim // self.n_chunks


@dataclass(unsafe_hash=True)
class PixelCNNParams:
    """Parameters for PixelCNN autoregressive prior."""
    embed_dim: int = 64  # Dimension of token embeddings
    hidden_channels: int = 128  # Channels in hidden layers
    n_layers: int = 7  # Number of masked conv layers (after first)
    kernel_size: int = 7  # Kernel size for masked convolutions (must be odd)
    
    # Training
    num_epochs: int = 50
    batch_size: int = 256
    lr: float = 2e-4
    val_split: float = 0.1  # Validation split fraction
    
    # Sampling
    temperature: float = 1.0  # Sampling temperature (>1 more diverse, <1 sharper)


@dataclass(unsafe_hash=True)
class VQVAEParams:
    """Parameters for VQ-VAE (end-to-end learned codebook)."""
    num_embeddings: int = 256  # K, codebook size (same as geodesic n_codewords)
    embedding_dim: int = 4  # D, embedding dimension (same as chunk_size)
    commitment_cost: float = 0.25  # weight for commitment loss
    
    # Training
    num_epochs: int = 50
    batch_size: int = 128
    lr: float = 1e-3
    
    # Grid shape for latent codes (must match encoder output)
    grid_h: int = 7
    grid_w: int = 7


@dataclass(unsafe_hash=True)
class DataParams:
    """Data loading parameters."""
    dataset: str = "mnist"  # 'mnist' or 'cifar' or 'celeba'
    batch_size: int = 128
    num_workers: int = 0
    seed: int = cfg.SEED


@dataclass
class ExperimentConfig:
    """Full experiment configuration combining all parameter groups."""
    arch_params: VAEArchParams = field(default_factory=VAEArchParams)
    training_params: TrainingParams = field(default_factory=TrainingParams)
    quant_params: GeodesicQuantizationParams = field(default_factory=GeodesicQuantizationParams)
    pixelcnn_params: PixelCNNParams = field(default_factory=PixelCNNParams)
    pixelcnn_vqvae_params: PixelCNNParams = field(default_factory=PixelCNNParams)  # Same class, different instance (just for testing)
    vqvae_params: VQVAEParams = field(default_factory=VQVAEParams)
    data_params: DataParams = field(default_factory=DataParams)

    use_gpu: bool = True
    seed: int = cfg.SEED

    def to_dict(self) -> Dict:
        """Convert to dictionary for logging."""
        return {
            "architecture": self.arch_params.__dict__,
            "training": self.training_params.__dict__,
            "quantization": self.quant_params.__dict__,
            "pixelcnn": self.pixelcnn_params.__dict__,
            "pixelcnn_vqvae": self.pixelcnn_vqvae_params.__dict__,
            "vqvae": self.vqvae_params.__dict__,
            "data": self.data_params.__dict__,
            "use_gpu": self.use_gpu,
            "seed": self.seed,
        }


# Predefined configurations
def get_mnist_config() -> ExperimentConfig:
    """Get configuration for MNIST experiments."""
    return ExperimentConfig(
        arch_params=VAEArchParams(
            in_channels=1,
            out_channels=1,
            hidden_channels=128,
            latent_dim=256
        ), 
        training_params=TrainingParams(
            num_epochs=50,
            batch_size=256,
            lr=1e-3,
            variational_beta=1.0
        ),
        quant_params=GeodesicQuantizationParams(
            n_chunks=16,  # latent_dim / n_chunks = chunk_size -> 256 / 16 = 16 -> grid 4x4=16 
            n_codewords=512, # same as num_embeddings in VQ-VAE
            knn_k=30,
            grid_h=4,
            grid_w=4
        ),   
        vqvae_params=VQVAEParams(
            num_embeddings=512, 
            embedding_dim=64, 
            commitment_cost=0.25,
            batch_size=256,
            grid_h=7,  # FIXED: Encoder produces 7×7 spatial map
            grid_w=7   # FIXED
        ),
        pixelcnn_params=PixelCNNParams(
            num_epochs=10,
            embed_dim=128,
            hidden_channels=256,
            n_layers=10,
            batch_size=256,
            lr=1e-4
        ),
        pixelcnn_vqvae_params=PixelCNNParams(
            num_epochs=10,
            embed_dim=128,
            hidden_channels=256,
            n_layers=10,
            batch_size=256,
            lr=1e-4
        ),
        data_params=DataParams(dataset="mnist", batch_size=256, num_workers=0)
    )


def get_cifar10_config() -> ExperimentConfig:
    return ExperimentConfig(
        arch_params=VAEArchParams(
            in_channels=3, 
            out_channels=3, 
            hidden_channels=128, 
            latent_dim=512
        ),
        training_params=TrainingParams(
            num_epochs=50,
            batch_size=256,
            lr=1e-3, 
            variational_beta=0.1 # beta for CIFAR-10 
        ),
        quant_params=GeodesicQuantizationParams(
            n_chunks=64,  # Latent_dim / n_chunks = chunk_size -> 512 / 64 = 8 -> grid 8x8=64
            n_codewords=512,  # Same as num_embeddings in VQ-VAE
            knn_k=30,
            grid_h=8,
            grid_w=8
        ),
        vqvae_params=VQVAEParams(
            num_embeddings=512,  # Codebook size 
            embedding_dim=64, 
            commitment_cost=0.25,
            num_epochs=50,
            batch_size=256,
            lr=1e-3,
            grid_h=8, # FIXED: Encoder produces 8×8 spatial map
            grid_w=8  # FIXED
        ),
        pixelcnn_params=PixelCNNParams(
            num_epochs=20,
            embed_dim=64,  
            hidden_channels=128,  
            n_layers=12, 
            batch_size=256,
            lr=1e-3,  
            temperature=1.0
        ),
        pixelcnn_vqvae_params=PixelCNNParams(
            num_epochs=20,
            embed_dim=64,  
            hidden_channels=128,  
            n_layers=12, 
            batch_size=256,
            lr=1e-3,  
            temperature=1.0
        ),
        data_params=DataParams(
            dataset="cifar",
            batch_size=256,
            num_workers=0
        )
    )


def get_celeba_config(img_size=32) -> ExperimentConfig:
    # hardcoded for 32x32 images for now
    # 32×32 CelebA: 8×8 latent grid
    grid_h, grid_w = 8, 8
    n_chunks = 64
    beta = 1.0
    embedding_dim = 64
    batch_size = 512
    
    return ExperimentConfig(
        arch_params=VAEArchParams(
            in_channels=3,
            out_channels=3,
            hidden_channels=128,
            latent_dim=512
        ),
        training_params=TrainingParams(
            num_epochs=50,
            batch_size=batch_size,
            lr=1e-3,
            variational_beta=beta
        ),
        quant_params=GeodesicQuantizationParams(
            n_chunks=n_chunks,
            n_codewords=512,
            knn_k=30,
            grid_h=grid_h,
            grid_w=grid_w
        ),
        vqvae_params=VQVAEParams(
            num_embeddings=512,
            embedding_dim=embedding_dim,
            commitment_cost=0.25,
            num_epochs=50,
            batch_size=batch_size,
            lr=1e-3,
            grid_h=grid_h,
            grid_w=grid_w
        ),
        pixelcnn_params=PixelCNNParams(
            num_epochs=20,
            embed_dim=64,
            hidden_channels=128,
            n_layers=12,
            batch_size=batch_size,
            lr=1e-3,
            temperature=1.0
        ),
        pixelcnn_vqvae_params=PixelCNNParams(
            num_epochs=20,
            embed_dim=64,
            hidden_channels=128,
            n_layers=12,
            batch_size=batch_size,
            lr=1e-3,
            temperature=1.0
        ),
        data_params=DataParams(
            dataset="celeba",
            batch_size=batch_size, 
            num_workers=1 
        )
    )
