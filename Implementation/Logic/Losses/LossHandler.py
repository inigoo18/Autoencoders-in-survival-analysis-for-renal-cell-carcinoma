import torch
import torch.nn.functional as F
from torch import nn

import numpy as np

from Logic.Losses.LossType import LossType


class LossHandler():

    def __init__(self, loss_type, args):
        self.loss_type = loss_type
        self.args = args
        self.loss_dict_tr = {'MSE': [], 'SPARSE': []}  # here we keep count of the different losses we have
        self.loss_dict_val = {'MSE': [], 'SPARSE': []}
        if loss_type == LossType.SPARSE:
            if not 'reg_param' in args:
                print("ERROR :: Reg_param is a necessary argument for sparse autoencoders")

    def clear(self):
        self.loss_dict = {}

    def _sparse_loss(self, params):
        return sum([p.abs().sum() for p in params])

    def _add_loss(self, mode, name, val):
        if (mode == 'Train'):
            self.loss_dict_tr[name] += [val.item()]
        else:
            self.loss_dict_val[name] += [val.item()]


    def compute_loss(self, mode, X, predX, params = None):
        criterion = nn.MSELoss(reduction='sum')

        total_loss = criterion(X, predX)
        self._add_loss(mode, 'MSE', total_loss)

        if self.loss_type == LossType.SPARSE:
            sparse_loss = self._sparse_loss(params)
            self._add_loss(mode, 'SPARSE', sparse_loss * self.args['reg_param'])
            total_loss += sparse_loss * self.args['reg_param']

        return total_loss

    def process_batch(self, tr_batch_size, val_batch_size):
        print("Batch size", "training: ", tr_batch_size, "val: ", val_batch_size)

        tr_dict = {'MSE': [], 'SPARSE': []} # this should be made dynamically
        val_dict = {'MSE': [], 'SPARSE': []}

        tr_dict['MSE'] = np.mean(np.array(self.loss_dict_tr['MSE']).reshape(-1, tr_batch_size), axis = 1).tolist()
        tr_dict['SPARSE'] = np.mean(np.array(self.loss_dict_tr['SPARSE']).reshape(-1, tr_batch_size), axis=1).tolist()
        val_dict['MSE'] = np.mean(np.array(self.loss_dict_val['MSE']).reshape(-1, val_batch_size), axis=1).tolist()
        val_dict['SPARSE'] = np.mean(np.array(self.loss_dict_val['SPARSE']).reshape(-1, val_batch_size), axis=1).tolist()

        return tr_dict, val_dict






