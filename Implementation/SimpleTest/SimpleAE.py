from typing import Type

import torch
import math


class AE(torch.nn.Module):

    def __init__(self, input_dim):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 254),
            torch.nn.ReLU(),
            torch.nn.Linear(254,128),
            torch.nn.ReLU(),
            torch.nn.Linear(128,64),
            torch.nn.ReLU(),
            torch.nn.Linear(64,32),
            torch.nn.ReLU(),
            torch.nn.Linear(32,16)
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(16,32),
            torch.nn.ReLU(),
            torch.nn.Linear(32,64),
            torch.nn.ReLU(),
            torch.nn.Linear(64,128),
            torch.nn.ReLU(),
            torch.nn.Linear(128,254),
            torch.nn.ReLU(),
            torch.nn.Linear(254, input_dim)
        )

    def forward(self, x):
        print("Input dim:", len(x))
        encoded = self.encoder(x)
        print("WOOHOO")
        decoded = self.decoder(encoded)
        return decoded
