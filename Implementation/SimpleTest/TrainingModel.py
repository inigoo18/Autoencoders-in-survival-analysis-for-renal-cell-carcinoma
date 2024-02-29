import string
from typing import Type

import torch
import torch.optim


class TrainingModel():
    def __init__(self, name, X, y, model : torch.nn.Module, optim: Type[torch.optim.Optimizer], epochs: int, loss_function):
        self.name = name
        self.X = X
        self.y = y
        self.model = model
        self.optim = optim
        self.epochs = epochs
        self.loss_function = loss_function
