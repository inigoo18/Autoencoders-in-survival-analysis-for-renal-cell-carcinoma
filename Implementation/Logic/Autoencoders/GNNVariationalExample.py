from typing import Type

import torch
import torch_geometric

from torch.nn import Linear, ReLU,Dropout
from torch_geometric.nn import Sequential, GCNConv, TopKPooling, SimpleConv, GeneralConv
import torch.nn.functional as F
import torch.nn as nn

from Logic.Autoencoders.VariationalExample import VariationalExample


class GNNVariationalExample(nn.Module):
    def __init__(self, num_features, input_dim, L, batch_size):
        super(GNNVariationalExample, self).__init__()
        self.genConv = GeneralConv(num_features, num_features, aggr='mean')
        # We instantiate the variational autoencoder
        self.model = VariationalExample(input_dim, L)
        self.encoder = self.model.encoder
        self.decoder = self.model.decoder
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.num_features = num_features
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dropout = nn.Dropout(p=0.1)
        self.lrelu = nn.LeakyReLU()

    def convolute(self, data):
        xs = torch.tensor([]).to(self.device)
        for i in range(len(data)):
            x, edge_index = data[i].x.to(self.device), data[i].edge_index.to(self.device)
            h = self.genConv(x, edge_index)
            h = self.lrelu(h)
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
