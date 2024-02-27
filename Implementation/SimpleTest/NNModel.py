from abc import ABC, abstractmethod
from typing import Type
import torch

class NNModel(ABC):
    def __init__(self, lr: float, optim: Type[torch.optim.Optimizer], epochs: int, loss_function):
        self.lr = lr
        self.optim = optim
        self.epochs = epochs
        self.loss_function = loss_function

    @abstractmethod
    def forward(self):
        pass
