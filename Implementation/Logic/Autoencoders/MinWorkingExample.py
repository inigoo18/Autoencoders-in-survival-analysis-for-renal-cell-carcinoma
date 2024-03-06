from typing import Type

import torch
import math


class MWE_AE(torch.nn.Module):

    def __init__(self, input_dim, L):
        super().__init__()

        print("Initializing Minimal Working Example AE with input dim: ", input_dim)


        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 2500),
            torch.nn.BatchNorm1d(2500),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(2500,2000),
            torch.nn.BatchNorm1d(2000),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(2000, 1500),
            torch.nn.BatchNorm1d(1500),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(1500,1000),
            torch.nn.BatchNorm1d(1000),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(1000,500),
            torch.nn.BatchNorm1d(500),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(500,L),
            torch.nn.BatchNorm1d(L),
            torch.nn.Sigmoid()
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(L,500),
            torch.nn.BatchNorm1d(500),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(500,1000),
            torch.nn.BatchNorm1d(1000),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(1000, 1500),
            torch.nn.BatchNorm1d(1500),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(1500,2000),
            torch.nn.BatchNorm1d(2000),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(2000,2500),
            torch.nn.BatchNorm1d(2500),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(2500, input_dim),
            torch.nn.BatchNorm1d(input_dim),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def get_latent_space(self, x):
        return self.encoder(x)
