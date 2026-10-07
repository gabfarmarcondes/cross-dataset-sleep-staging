# scripts/check_seed.py
import random

import numpy as np
import torch

from src.training.seed import set_seed


def draw():
    """One number from each generator."""
    return random.random(), np.random.rand(), torch.rand(1).item()


set_seed(0)
a = draw()
set_seed(0)
b = draw()
set_seed(1)
c = draw()

assert a == b, "same seed gave different numbers"
assert a != c, "different seeds gave the same numbers"
print("set_seed ok:", a)