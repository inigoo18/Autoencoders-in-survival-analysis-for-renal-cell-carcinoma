import string
from typing import Type

import torch
import torch.optim
import numpy as np
import os

from Logic.Losses.LossHandler import LossHandler
from Logic.Losses.LossType import LossType


class TrainingModel():
    def __init__(self, name, train_loader, test_loader, val_loader, cli_vars, model : torch.nn.Module,
                 loss_fn : LossHandler, optim: Type[torch.optim.Optimizer], epochs: int, BATCH_SIZE, L, isGNN, load_state = None):
        self.name = name
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader

        self.cli_vars = cli_vars

        # Here we hold a dataframe with all the data regarding test patients
        self.model = model
        self.loss_fn = loss_fn
        self.optim = optim
        self.epochs = epochs
        self.L = L
        self.BATCH_SIZE = BATCH_SIZE
        self.GNN = isGNN
        self.trained = False
        self.variational = False
        if LossType.VARIATIONAL in loss_fn.loss_types:
            self.variational = True
        if not os.path.exists("Results/"+name):
            os.makedirs("Results/"+name)
        if not os.path.exists("Checkpoints/"):
            os.makedirs("Checkpoints/")
        if load_state is not None:
            self.model.load_state_dict(torch.load(load_state))
            self.trained = True

    def compute_model_loss(self, X, predX, mu = None, log_var = None, graph_loss = None):
        mode = 'Val'
        if self.model.training:
            mode = 'Train'
        return self.loss_fn.compute_loss(mode, X, predX, self.model.parameters(), mu, log_var, graph_loss)

    def fetch_model_loss(self):
        loss_dict = self.loss_fn.loss_dict

    def unroll_loader(self, loader, dim):
        # dim = 0 corresponds to genData
        # 1 corresponds to cliData
        # we can't have multi-indices with a list, therefore we remove the list with the for loop and then perform tensor operations
        loader_lists = [tensor for tensor in loader]

        # Extract slices along the specified dimension
        slices = [tensor[dim] for tensor in loader_lists]

        # Flatten the slices
        flattened_slices = [item for sublist in slices for item in sublist]

        len0 = len(flattened_slices)
        len1 = len(flattened_slices[0])

        stacked_tensor = torch.stack(flattened_slices)
        reshaped_tensor = stacked_tensor.reshape(len0, len1)

        return reshaped_tensor


