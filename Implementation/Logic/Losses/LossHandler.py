import torch
import torch.nn.functional as F
from torch import nn

from Logic.Losses.LossType import LossType


class LossHandler():

    def __init__(self, loss_type, args):
        self.loss_type = loss_type
        self.args = args
        if loss_type == LossType.SPARSE:
            if not 'reg_param' in args:
                print("ERROR :: Reg_param is a necessary argument for sparse autoencoders")


    def _sparse_loss(self, params):
        return sum([p.abs().sum() for p in params])

    def compute_loss(self, X, predX, params = None):
        criterion = nn.MSELoss(reduction='sum')

        total_loss = criterion(X, predX)

        if self.loss_type == LossType.SPARSE:
            sparse_loss = self._sparse_loss(params)
            total_loss += sparse_loss * self.args['reg_param']

        return total_loss
