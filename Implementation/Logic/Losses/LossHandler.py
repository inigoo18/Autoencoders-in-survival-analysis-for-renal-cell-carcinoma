import torch
import torch.nn.functional as F
from torch import nn

import numpy as np

from Logic.Losses.LossType import LossType


class LossHandler():

    def __init__(self, loss_type, args):
        self.loss_type = loss_type
        self.args = args
        self.loss_dict_tr = {}  # here we keep count of the different losses we have
        self.loss_dict_val = {}
        self.loss_dict_tr['MSE'] = []
        self.loss_dict_val['MSE'] = []
        if loss_type == LossType.SPARSE:
            self.loss_dict_tr['SPARSE'] = []
            self.loss_dict_val['SPARSE'] = []
            if not 'reg_param' in args:
                print("ERROR :: Reg_param is a necessary argument for sparse autoencoders, fixing to 0.1")
                self.args['reg_param'] = 0.1

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
        keys = list(self.loss_dict_tr.keys())

        tr_dict = {}
        val_dict = {}

        for key in keys:
            tr_dict[key] = np.mean(np.array(self.loss_dict_tr[key]).reshape(-1, tr_batch_size), axis = 1).tolist()
            val_dict[key] = np.mean(np.array(self.loss_dict_val[key]).reshape(-1, val_batch_size), axis=1).tolist()

        return tr_dict, val_dict






