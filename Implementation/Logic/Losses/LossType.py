import torch
import torch.nn.functional as F
from torch import nn

from enum import Enum

class LossType(Enum):
    MSE = 1
    SPARSE = 2
