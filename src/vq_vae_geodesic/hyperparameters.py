"""
Hyperparameters and model configuration using dataclasses.
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
    variational_beta: float = 0.1  # Beta parameter for VAE loss
    save_checkpoint_every: int = 1


@dataclass(unsafe_hash=True)
class GeodesicQuantizationParams:
    """Parameters for geodesic quantization."""
    n_chunks: int = 8  # Number of chunks to split latent
    n_codewords: int = 256  # Size of the codebook
    use_var_in_features: bool = False  # Whether to use variance in clustering
    knn_k: int = 20  # Number of neighbors for k-NN graph
    mds_dim: int = 64  # Dimensionality for MDS embedding
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
    data_params: DataParams = field(default_factory=DataParams)

    use_gpu: bool = True
    seed: int = cfg.SEED

    def to_dict(self) -> Dict:
        """Convert to dictionary for logging."""
        return {
            "architecture": self.arch_params.__dict__,
            "training": self.training_params.__dict__,
            "quantization": self.quant_params.__dict__,
            "data": self.data_params.__dict__,
            "use_gpu": self.use_gpu,
            "seed": self.seed,
        }


# Predefined configurations
def get_mnist_config() -> ExperimentConfig:
    """Get configuration for MNIST experiments."""
    return ExperimentConfig(
        arch_params=VAEArchParams(in_channels=1, out_channels=1, hidden_channels=128, latent_dim=32),
        training_params=TrainingParams(num_epochs=100, batch_size=128, variational_beta=0.1),
        quant_params=GeodesicQuantizationParams(n_chunks=8, n_codewords=256),
        data_params=DataParams(dataset="mnist", batch_size=128)
    )


def get_cifar_config() -> ExperimentConfig:
    """Get configuration for CIFAR-10 experiments."""
    return ExperimentConfig(
        arch_params=VAEArchParams(in_channels=3, out_channels=3, hidden_channels=128, latent_dim=32),
        training_params=TrainingParams(num_epochs=100, batch_size=128, variational_beta=1.0),
        quant_params=GeodesicQuantizationParams(n_chunks=8, n_codewords=256),
        data_params=DataParams(dataset="cifar", batch_size=128)
    )
