import string
from typing import Type

import torch
import torch.optim
import numpy as np


class TrainingModel():
    def __init__(self, name, X_train, y_train, X_test, y_test, X_val, y_val, model : torch.nn.Module, optim: Type[torch.optim.Optimizer], epochs: int, loss_function, columns, L):
        self.name = name
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.X_val = X_val
        self.y_val = y_val
        self.model = model
        self.optim = optim
        self.epochs = epochs
        self.loss_function = loss_function
        self.columns = columns
        self.L = L

    def unroll_Xtrain(self):
        return [i for j in self.X_train for i in j]

    def unroll_Xtest(self):
        return [i for j in self.X_test for i in j]

    def unroll_Xval(self):
        return [i for j in self.X_val for i in j]

    def unroll_Ytrain(self):
        flattened_data = [tuple(item) for sublist in self.y_train for item in sublist]
        return np.array(flattened_data, dtype=[('event', bool), ('time', float)])

    def unroll_Ytest(self):
        flattened_data = [tuple(item) for sublist in self.y_test for item in sublist]
        return np.array(flattened_data, dtype=[('event', bool), ('time', float)])

    def unroll_Yval(self):
        flattened_data = [tuple(item) for sublist in self.y_val for item in sublist]
        return np.array(flattened_data, dtype=[('event', bool), ('time', float)])
