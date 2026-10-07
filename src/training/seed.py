import random

import numpy as np
import torch

def set_seed(seed, deterministic=True):
    """
    Seed Python, NumPy and PyTorch so that a run can be repeated exactly.

    deterministic=True also asks cuDNN for deterministics algorithms. Training on GPU may get slower,
    and results can still differ slightly across machine or libray version
    """

    random.seed(seed) # Python's random module
    np.random.seed(seed) # NumPy (e.g. data augmentation)
    torch.manual_seed(seed) # PyTorch (weight init, dropout, covers CPU and GPU)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False