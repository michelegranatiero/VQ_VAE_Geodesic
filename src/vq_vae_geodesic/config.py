from pathlib import Path

# Project root
ROOT = Path(__file__).parent.parent.parent
DATA_DIR = ROOT / "data"
CHECKPOINT_DIR = DATA_DIR / "checkpoints"
LATENTS_DIR = DATA_DIR / "latents"
RECONS_DIR = DATA_DIR / "recons"
WANDB_DIR = ROOT / "wandb"

# Seed for reproducibility
SEED = 42

def data_dir() -> Path:
    """Get the data directory."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


def checkpoint_dir() -> Path:
    """Get the checkpoint directory."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    return CHECKPOINT_DIR


def latents_dir() -> Path:
    """Get the latents directory."""
    LATENTS_DIR.mkdir(parents=True, exist_ok=True)
    return LATENTS_DIR


def recons_dir() -> Path:
    """Get the reconstructions directory."""
    RECONS_DIR.mkdir(parents=True, exist_ok=True)
    return RECONS_DIR


def wandb_dir() -> Path:
    """Get the wandb directory."""
    WANDB_DIR.mkdir(parents=True, exist_ok=True)
    return WANDB_DIR
