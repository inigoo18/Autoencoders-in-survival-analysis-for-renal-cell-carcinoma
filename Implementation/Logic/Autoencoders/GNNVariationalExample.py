from typing import Type

import torch
import torch_geometric

from torch.nn import Linear, ReLU,Dropout
from torch_geometric.nn import Sequential, GCNConv, TopKPooling, SimpleConv
import torch.nn.functional as F
import torch.nn as nn

from Logic.Autoencoders.VariationalExample import VariationalExample


class GNNVariationalExample(nn.Module):
    def __init__(self, num_features, input_dim, L, batch_size):
        super(GNNVariationalExample, self).__init__()
        self.conv = GCNConv(num_features,
                            num_features)  # SimpleConv(aggr = "median", combine_root = "self_loop") # aggr :: [sum, mean, mul]
        self.model = VariationalExample(input_dim, L)
        self.encoder = self.model.encoder
        self.decoder = self.model.decoder
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.num_features = num_features
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dropout = nn.Dropout(p=0.15)
        self.lrelu = nn.LeakyReLU()
        self.batchNorm = nn.BatchNorm1d(num_features)

    def convolute(self, data):
        xs = torch.tensor([]).to(self.device)
        for i in range(len(data)):
            x, edge_index = data[i].x, data[i].edge_index
            h = self.conv(x, edge_index)
            h = self.batchNorm(h)  # todo:: if no good results, remove batchnorm and tanh and put relu instead
            h = h.tanh()
            h = self.dropout(h)
            xs = torch.cat([xs, h])
        return xs

    def forward(self, data):
        xs = self.convolute(data)
        xs = torch.reshape(xs, (len(data), self.input_dim))
        x_hat, mean, log_var = self.model.forward(xs)
        return x_hat, mean, log_var

    def get_latent_space(self, data):
        xs = self.convolute(data)
        xs = torch.reshape(xs, (-1, self.input_dim))
        z = self.model.get_latent_space(xs)
        return z
