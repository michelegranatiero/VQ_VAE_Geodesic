import copy
import random
from typing import Callable, Optional

import numpy as np
import torch


def make_averager() -> Callable[[Optional[float]], float]:
    """ Returns a function that maintains a running average

    :returns: running average function
    """
    count = 0
    total = 0

    def averager(new_value: Optional[float]) -> float:
        """ Running averager

        :param new_value: number to add to the running average,
                          if None returns the current average
        :returns: the current average
        """
        nonlocal count, total
        if new_value is None:
            return total / count if count else float("nan")
        count += 1
        total += new_value
        return total / count

    return averager


def refresh_bar(bar, desc):
    bar.set_description(desc)
    bar.refresh()


def set_seed(seed: int):
    """Set the seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True  # Note that this Deterministic mode can have a performance impact
    torch.backends.cudnn.benchmark = False


class EarlyStopping:
    def __init__(self, path, patience=5, min_delta=0.0, restore_best_weights=True, mode='min'):
        """
        mode: 'min' or 'max' depending on metric (use 'min' for loss, 'max' for accuracy/psnr/ssim)
        restore_best_weights: if True, restores best model weights after early stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.best_model = None
        self.best_loss = None
        self.counter = 0
        self.status = ""
        self.path = path

    def _is_better(self, current, best):
        if best is None:
            return True
        if self.mode == 'min':
            return current < best - self.min_delta
        else:
            return current > best + self.min_delta

    def step(self, current, model, checkpoint: dict):
        """
        current: current monitored metric (float)
        checkpoint: dict with all info to save if new best (epoch, histories, model/optimizer state, ...)
        Returns: True if should_stop
        """
        if self._is_better(current, self.best_loss):
            self.best_loss = current
            self.best_model = copy.deepcopy(model.state_dict())
            self.counter = 0
            self.status = f"Improvement found, counter reset to {self.counter}"
            torch.save(checkpoint, self.path)
            return False
        else:
            self.counter += 1
            self.status = f"No improvement in the last {self.counter} epochs"
            if self.counter >= self.patience:
                self.status = f"Early stopping triggered after {self.counter} epochs."
                if self.restore_best_weights and model is not None and self.best_model is not None:
                    model.load_state_dict(self.best_model)
                return True
            return False