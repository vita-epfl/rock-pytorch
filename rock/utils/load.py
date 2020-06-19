import os
from typing import Optional, Any, Tuple

import torch
import torch.nn


def load_from_checkpoint(checkpoint_path: str,
                         model: torch.nn.Module,
                         optimizer: Optional[Any] = None,  # from torch.optim
                         scheduler: Optional[Any] = None,  # from torch.optim
                         verbose: bool = True) -> Tuple[int, int]:
    """Loads model from checkpoint, loads optimizer and scheduler too if not None, and returns epoch and iteration
    of the checkpoints
    """
    if not os.path.exists(checkpoint_path):
        raise ("File doesn't exist {}".format(checkpoint_path))

    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['model'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])

    if scheduler:
        scheduler.load_state_dict(checkpoint['scheduler'])

    epoch = checkpoint['epoch']
    iteration = checkpoint['iteration']

    if verbose:
        print("Loaded model from checkpoint")

    return epoch, iteration
