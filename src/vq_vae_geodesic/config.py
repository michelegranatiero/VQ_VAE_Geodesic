from pathlib import Path

# Project root
ROOT = Path(__file__).parent.parent.parent
DATA_DIR = ROOT / "data"
WANDB_DIR = ROOT / "wandb"

# Seed for reproducibility
SEED = 42

def data_dir() -> Path:
    """Get the data directory."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


def checkpoint_dir(dataset: str = None) -> Path:
    """
    Get the checkpoint directory for a specific dataset
    """
    if dataset is None:
        checkpoint_path = DATA_DIR / "checkpoints"
    elif dataset.lower() in ['mnist', 'cifar10']:
        checkpoint_path = DATA_DIR / f"checkpoints_{dataset.lower()}"
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Use 'mnist' or 'cifar10'")
    
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    return checkpoint_path


def latents_dir(dataset: str = None) -> Path:
    """
    Get the latents directory for a specific dataset
    """
    if dataset is None:
        latents_path = DATA_DIR / "latents"
    elif dataset.lower() in ['mnist', 'cifar10']:
        latents_path = DATA_DIR / f"latents_{dataset.lower()}"
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Use 'mnist' or 'cifar10'")
    
    latents_path.mkdir(parents=True, exist_ok=True)
    return latents_path


def recons_dir(dataset: str = None) -> Path:
    """
    Get the reconstructions directory for a specific dataset
    """
    if dataset is None:
        recons_path = DATA_DIR / "recons"
    elif dataset.lower() in ['mnist', 'cifar10']:
        recons_path = DATA_DIR / f"recons_{dataset.lower()}"
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Use 'mnist' or 'cifar10'")
    
    recons_path.mkdir(parents=True, exist_ok=True)
    return recons_path


def samples_dir(dataset: str = None) -> Path:
    """
    Get the samples directory for a specific dataset
    """
    if dataset is None:
        samples_path = DATA_DIR / "samples"
    elif dataset.lower() in ['mnist', 'cifar10']:
        samples_path = DATA_DIR / f"samples_{dataset.lower()}"
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Use 'mnist' or 'cifar10'")
    
    samples_path.mkdir(parents=True, exist_ok=True)
    return samples_path


def wandb_dir() -> Path:
    """Get the wandb directory."""
    WANDB_DIR.mkdir(parents=True, exist_ok=True)
    return WANDB_DIR
