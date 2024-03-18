import torch
import torch.nn.functional as F
from torch import nn

from enum import Enum

class LossType(Enum):
    MSE = 1
    SPARSE_L1 = 2
    SPARSE_KL = 3
    DENOISING = 4
    VARIATIONAL = 5

    def __str__(self):
        return str(self.name)
