import torch
import torch.nn.functional as F
from torch import nn

import numpy as np

from Logic.Losses.LossType import LossType


class LossHandler():

    def __init__(self, loss_types, args, adj_matrix = None):
        self.loss_types = loss_types
        self.args = args
        self.loss_dict_tr = {}  # here we keep count of the different losses we have
        self.loss_dict_val = {}
        self.loss_dict_tr['MSE'] = []
        self.loss_dict_val['MSE'] = []
        for loss_type in loss_types:
            if loss_type != LossType.MSE and loss_type != LossType.DENOISING:
                self.loss_dict_tr[str(loss_type)] = []
                self.loss_dict_val[str(loss_type)] = []
        self.check_arguments()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.adj_matrix = adj_matrix

    def check_arguments(self):
        keys = []
        if LossType.SPARSE_L1 in self.loss_types:
            keys += ['reg_param']
        if LossType.SPARSE_KL in self.loss_types:
            keys += ['reg_param', 'rho']
        if LossType.DENOISING in self.loss_types:
            keys += ['noise_factor']

        if not (all(key in self.args for key in keys)):
            print("ERROR :: a loss type was specified but the required arguments aren't")


    def clear(self):
        self.loss_dict = {}

    def initialize_loss(self, data, isGNN):
        '''
        This function takes the data and does something to it depending on the loss type we selected.
        Mainly so that if we selected a denoising autoencoder, we can add some noise to the data
        :param data: input data
        :return: a variant of the input data
        '''
        if LossType.DENOISING in self.loss_types:
            if isGNN:
                # we only want to access the x field (features) of the graph
                data.x = data.x + self.args['noise_factor'] * torch.randn_like(data.x)
                data.x = torch.clamp(data.x, min=0, max=1)
                return data
            else:
                noisy_data = data + self.args['noise_factor'] * torch.randn_like(data) # torch.randn follows gaussian distribution
                noisy_data = torch.clamp(noisy_data, min= 0, max = 1)
                return noisy_data
        return data

    def _sparse_loss(self, params):
        '''
        Loss function used for L1 sparsity, it just sums all the parameters together after taking the absolute value
        :param params: argument list defined in LossHandler class
        :return: total loss
        '''
        return sum([p.abs().sum() for p in params])

    def _sparse_kl_loss(self, RHO, params):
        '''
        The idea of this loss function is that we have two probabilities, RHO and RHO_HAT.
        RHO is the parameter we choose to regularize the autoencoder with, for example if we set it to 0.1, we
        expect most parameters to be close to 0.1.
        RHO_HAT is the actual activation of the neurons by looking at their parameters.
        By computing the KL divergence between RHO and RHO_HAT, we can find how distinct both probabilities are.
        :param RHO: hyperparameter to regularize the autoencoder with
        :param params: argument list defined in LossHandler class
        :return: each layer adds to the total sparsity loss
        '''
        TOTAL = 0
        for p in params:
            p = p.to(self.device)
            RHO_HAT = torch.mean(F.tanh(p.abs())).to(self.device)
            if RHO_HAT == 0: # if the param is 0, we add a little bit so that the calculation doesn't fail
                RHO_HAT += 0.001

            RES = torch.tanh((RHO * torch.log(RHO / RHO_HAT) + (1 - RHO) * torch.log((1 - RHO) / (1 - RHO_HAT))).abs()).to(self.device)
            TOTAL += RES * len(p) # we multiply by p so we can give some weight depending on how many neurons
        return TOTAL

    def _add_loss(self, mode, name, val):
        if (mode == 'Train'):
            self.loss_dict_tr[name] += [val.item()]
        else:
            self.loss_dict_val[name] += [val.item()]


    def compute_loss(self, mode, X, predX, params = None, mean = None, log_var = None):
        total_loss = 0
        if self.adj_matrix is None:
            criterion = nn.MSELoss(reduction='sum')
            total_loss = criterion(X, predX)
        else:
            criterion = nn.MSELoss(reduction = 'sum')

            x_features = torch.reshape(X.x, (predX.shape[0], predX.shape[1]))

            total_loss = criterion(x_features, predX)


            # Calculate BCE loss
            adjacency_loss = 0
            for x_i in predX:
                adjacency_pred = torch.ger(x_i, x_i.t())
                adjacency_loss += nn.BCEWithLogitsLoss()(adjacency_pred.flatten(), self.adj_matrix.flatten())

            total_loss += adjacency_loss


        self._add_loss(mode, 'MSE', total_loss)

        # We're only interested in additional losses during training phase, when we learn the weights.
        if mode == 'Train':
            if LossType.SPARSE_L1 in self.loss_types:
                sparse_loss = self._sparse_loss(params)
                self._add_loss(mode, 'SPARSE_L1', sparse_loss * self.args['reg_param'])
                total_loss += sparse_loss * self.args['reg_param']

            if LossType.SPARSE_KL in self.loss_types:
                sparse_kl_loss = self._sparse_kl_loss(self.args['rho'], params)
                self._add_loss(mode, 'SPARSE_KL', sparse_kl_loss * self.args['reg_param'])
                total_loss += sparse_kl_loss * self.args['reg_param']

            if LossType.VARIATIONAL in self.loss_types:
                variational_kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
                self._add_loss(mode, 'VARIATIONAL', variational_kl_loss)
                total_loss += variational_kl_loss

        return total_loss


    def process_batch(self, tr_batch_size, val_batch_size):
        keys = list(self.loss_dict_tr.keys())

        tr_dict = {}
        val_dict = {}
        # In loss_dict_tr we have losses per batch. What we do is grab the mean of the whole group and have that as the
        # loss obtained in the epoch.
        for key in keys:
            tr_dict[key] = np.mean(np.array(self.loss_dict_tr[key]).reshape(-1, tr_batch_size), axis = 1).tolist()
            val_dict[key] = np.mean(np.array(self.loss_dict_val[key]).reshape(-1, val_batch_size), axis=1).tolist()

        return tr_dict, val_dict






