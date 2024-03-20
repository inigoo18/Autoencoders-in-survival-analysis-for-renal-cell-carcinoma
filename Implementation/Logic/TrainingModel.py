import string
from typing import Type

import torch
import torch.optim
import numpy as np
import os

from Logic.Losses.LossHandler import LossHandler
from Logic.Losses.LossType import LossType


class TrainingModel():
    def __init__(self, name, X_train, y_train, X_test, y_test, X_val, y_val, demographic_test, model : torch.nn.Module,
                 loss_fn : LossHandler, optim: Type[torch.optim.Optimizer], epochs: int, BATCH_SIZE, columns, L, load_state = None):
        self.name = name
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.X_val = X_val
        self.y_val = y_val
        # Here we hold a dataframe with all the data regarding test patients
        self.demographic_test = demographic_test
        self.model = model
        self.loss_fn = loss_fn
        self.optim = optim
        self.epochs = epochs
        self.columns = columns
        self.L = L
        self.BATCH_SIZE = BATCH_SIZE
        self.trained = False
        self.variational = False
        if LossType.VARIATIONAL in loss_fn.loss_types:
            self.variational = True
        if not os.path.exists("Results/"+name):
            os.makedirs("Results/"+name)
        if load_state is not None:
            self.model.load_state_dict(torch.load(load_state))
            self.trained = True

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

    def compute_model_loss(self, X, predX, mu = None, log_var = None):
        mode = 'Val'
        if self.model.training:
            mode = 'Train'
        return self.loss_fn.compute_loss(mode, X, predX, self.model.parameters(), mu, log_var)

    def fetch_model_loss(self):
        loss_dict = self.loss_fn.loss_dict

    def fetch_train_val_total_length(self):
        length_tr = 0
        length_val = 0
        for i in self.X_train:
            length_tr += len(i)
        for i in self.X_val:
            length_val += len(i)
        return length_tr, length_val


