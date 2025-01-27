import string
from typing import Type

import torch
import torch.optim
import numpy as np
import os

from Logic.IterationObject import IterationObject
from Logic.Losses.LossHandler import LossHandler
from Logic.Losses.LossType import LossType


class TrainingModel():

    '''
    This class takes care of training the model and evaluating it. Therefore we need all information regarding a
    specific fold: train, test, val loaders, clinical features, input dimension, the model we're using, the loss function,
    batch size, whether its a GNN, etc.
    '''

    def __init__(self, name, data_loader, iteration : IterationObject, cli_vars, model : torch.nn.Module,
                 loss_fn : LossHandler, optim: Type[torch.optim.Optimizer], epochs: int, BATCH_SIZE, L, isGNN, load_state = None):
        self.name = name
        self.data_loader = data_loader
        self.input_dim = data_loader.input_dim
        self.train_loader = iteration.train_data
        self.test_loader = iteration.test_data
        self.val_loader = iteration.val_data

        self.test_genes = iteration.test_genes
        self.cli_vars = cli_vars

        # Here we hold a dataframe with all the data regarding test patients
        self.model = model
        self.loss_fn = loss_fn
        self.optim = optim
        self.epochs = epochs
        self.L = L
        self.batch_size = BATCH_SIZE
        self.GNN = isGNN
        self.trained = False
        self.variational = False
        if loss_fn is not None:
            if LossType.VARIATIONAL in loss_fn.loss_types:
                self.variational = True
        if not os.path.exists("Results/"+name):
            os.makedirs("Results/"+name)
        if not os.path.exists("Checkpoints/"):
            os.makedirs("Checkpoints/")
        if load_state is not None:
            self.model.load_state_dict(torch.load("Checkpoints/"+load_state))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(device)
            self.trained = True

    def compute_model_loss(self, X, predX, mu = None, log_var = None):
        '''
        We call LossHandler to compute the loss. mu and log_var are optional parameters in case we're using
        a variational autoencoder.
        '''
        mode = 'Val'
        if self.model.training:
            mode = 'Train'
        return self.loss_fn.compute_loss(mode, X, predX, self.model.parameters(), mu, log_var)


    def transform_to_tabular(self, data):
        '''
        This method takes the graph attribute batch and returns it in tabular format, so we can compare with the obtained representation.
        :return: Tabular format of the data
        '''
        if self.GNN:
            return torch.reshape(data.x, (len(data), self.input_dim))
        else:
            return data


