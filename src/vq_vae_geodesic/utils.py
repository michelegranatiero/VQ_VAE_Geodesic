import torch
import numpy as np
import random
from typing import Callable, Optional

import lightning.pytorch as L
import wandb
from pathlib import Path
from lightning.pytorch.callbacks import Callback
from typing import Any, Optional, Dict, Callable
from torch import Tensor
import os
import matplotlib.pyplot as plt


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


# =======================================================
# ======== Utilities for logging images to wandb ========
# =======================================================
# def tensor_to_image(t: Tensor):
#   return t.detach().cpu().permute(1, 2, 0).clip(0, 1).numpy()


# class SaveImages(Callback):

#   def __init__(self,
#                save_path: Path,
#                every_n_epochs: int = 1,
#                log_every: Optional[int] = None):
#     super().__init__()
#     self.save_path = save_path
#     self.every_n_epochs: int = every_n_epochs
#     self.log_every: Optional[int] = log_every

#   def on_validation_batch_end(self,
#                               trainer: L.Trainer,
#                               pl_module: L.LightningModule,
#                               outputs: Tensor,
#                               batch: Any,
#                               batch_idx: int,
#                               dataloader_idx: int = 0) -> None:

#     if batch_idx != 0:
#       return

#     if trainer.global_step == 0 or (trainer.current_epoch +
#                                     1) % self.every_n_epochs == 0:
#       os.makedirs(self.save_path, exist_ok=True)
#       save_path = (
#           self.save_path /
#           f"epoch={trainer.current_epoch}-step={trainer.global_step}")
#       predictions = pl_module(batch[0][:4].to(pl_module.device))
#       f, ax = plt.subplots(3, 4, figsize=(20, 20))

#       columns = ["input", "output", "ground truth"]
#       data = []
#       for i, (output, sample,
#               target) in enumerate(zip(predictions, batch[0], batch[1])):
#         if i > 3:
#           break
#         row = []
#         for x, image in enumerate([sample, output, target]):
#           if image.shape[-3] == 4:
#             image = image[..., :3, :, :]
#           image = tensor_to_image(image)
#           ax[x, i].imshow(image)
#           row.append(wandb.Image(image))
#         data.append(row)

#       plt.savefig(save_path)
#       plt.close()

#       if (self.log_every is not None and
#           ((trainer.current_epoch + 1) % self.log_every == 0) and
#               pl_module.logger is not None):
#         print("Logging demos to wandb")
#         pl_module.logger.log_table(  # type:ignore
#             key=f"demos_step{trainer.global_step}",
#             columns=columns,
#             data=data)
