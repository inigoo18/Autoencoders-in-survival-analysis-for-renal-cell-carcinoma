from typing import Type

import torch
import torch_geometric

from torch.nn import Linear, ReLU,Dropout
from torch_geometric.nn import Sequential, GCNConv


class GNNExample(torch.nn.Module):

    def __init__(self, input_dim, L):
        super().__init__()

        print("Initializing GRAPH Example AE with input dim: ", input_dim)

        # TODO :: understand how Loader is supposed to work these things

        self.encoder = Sequential("x, edge_index",
            [
                (GCNConv(input_dim, 2000), 'x, edge_index -> x1'),
                (torch.nn.ReLU(), 'x1 -> x1a'),
                (GCNConv(1000, L), 'x1a, edge_index -> x2'),
                (torch.nn.Sigmoid(), 'x2 -> x2a'),
            ]
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        encoded = self.encoder(x, edge_index)
        return encoded

    def get_latent_space(self, x, edge_index, batch_index):
        print("WARNING:: Need to be implemented in GNN Example (the arguments it gets called with)")
        return self.encoder(x, edge_index, batch_index)
