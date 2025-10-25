"""
Hyperparameters and model configuration.
"""
from dataclasses import dataclass, field
from typing import Dict

from vq_vae_geodesic import config as cfg


@dataclass(unsafe_hash=True)
class VAEArchParams:
    """VAE architecture parameters."""
    in_channels: int = 1  # 1 for MNIST, 3 for CIFAR
    out_channels: int = 1  # 1 for MNIST, 3 for CIFAR
    hidden_channels: int = 128  # Capacity - base width for the model
    latent_dim: int = 32  # Size of the latent bottleneck


@dataclass(unsafe_hash=True)
class TrainingParams:
    """Training hyperparameters."""
    num_epochs: int = 100
    batch_size: int = 128
    lr: float = 0.001
    weight_decay: float = 1e-5
    variational_beta: float = 1.0  # Beta parameter for VAE loss
    save_checkpoint_every: int = 1


@dataclass(unsafe_hash=True)
class GeodesicQuantizationParams:
    """Parameters for geodesic quantization."""
    n_chunks: int = 8  # Number of chunks to split latent
    n_codewords: int = 256  # Size of the codebook
    knn_k: int = 20  # Number of neighbors for k-NN graph
    subsample_max_pts: int = 5000  # Max points for subsampling
    random_state: int = cfg.SEED

    # Grid shape for codes (must satisfy H*W = n_chunks)
    grid_h: int = 2
    grid_w: int = 4

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
    num_epochs: int = 20
    batch_size: int = 128
    lr: float = 2e-4
    val_split: float = 0.1  # Validation split fraction
    
    # Sampling
    temperature: float = 1.0  # Sampling temperature (>1 more diverse, <1 sharper)


@dataclass(unsafe_hash=True)
class VQVAEParams:
    """Parameters for VQ-VAE (end-to-end learned codebook)."""
    num_embeddings: int = 256  # K, codebook size (same as geodesic n_codewords)
    embedding_dim: int = 4  # D, embedding dimension (same as chunk_size)
    commitment_cost: float = 0.25  # Beta, weight for commitment loss
    
    # Training
    num_epochs: int = 50
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 0.0  
    
    # Grid shape for latent codes (must match encoder output)
    grid_h: int = 2
    grid_w: int = 4


@dataclass(unsafe_hash=True)
class DataParams:
    """Data loading parameters."""
    dataset: str = "mnist"  # 'mnist' or 'cifar'
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
            "vqvae": self.vqvae_params.__dict__,
            "data": self.data_params.__dict__,
            "use_gpu": self.use_gpu,
            "seed": self.seed,
        }


# Predefined configurations
def get_mnist_config() -> ExperimentConfig:
    """Get configuration for MNIST experiments."""
    return ExperimentConfig(
        arch_params=VAEArchParams(in_channels=1, out_channels=1, hidden_channels=128, latent_dim=128), 
        training_params=TrainingParams(num_epochs=50, batch_size=128, variational_beta=0.1),
        quant_params=GeodesicQuantizationParams(n_chunks=16, n_codewords=512, grid_h=4, grid_w=4),   # latent_dim / n_chunks = chunk_size -> 128 / 16 = 8 -> grid 4x4=16  
        vqvae_params=VQVAEParams(num_embeddings=512, embedding_dim=64, commitment_cost=0.25),
        pixelcnn_params=PixelCNNParams(num_epochs=50, embed_dim=64, hidden_channels=128, n_layers=7),
        data_params=DataParams(dataset="mnist", batch_size=128)
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
            batch_size=100,
            lr=3e-4, 
            variational_beta=0.1
        ),
        quant_params=GeodesicQuantizationParams(
            n_chunks=64,  # 8x8 grid
            n_codewords=512,  # Codebook size (K)
            knn_k=20,  # k-NN neighbors
            grid_h=8,
            grid_w=8
        ),
        vqvae_params=VQVAEParams(
            num_embeddings=512,  # codebook size 
            embedding_dim=64, 
            commitment_cost=0.25,
            num_epochs=50,
            batch_size=256,  
            lr=1e-3,
            grid_h=8,
            grid_w=8
        ),
        pixelcnn_params=PixelCNNParams(
            embed_dim=64,  
            hidden_channels=128,  
            n_layers=12, 
            num_epochs=50,  
            batch_size=100,
            lr=1e-3,  
            temperature=1.0
        ),
        data_params=DataParams(
            dataset="cifar",
            batch_size=100,
            num_workers=0
        )
    )
