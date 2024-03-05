from typing import Type

import torch
import math

from torch import nn
import torch.nn.functional as F

class SparseAE(torch.nn.Module):

    def __init__(self, input_dim, L):
        super().__init__()

        print("Initializing Sparse AE with input dim: ", input_dim)

        self.enc1 = nn.Linear(input_dim, 2500)
        self.enc2 = nn.Linear(2500, 2000)
        self.enc3 = nn.Linear(2000, 1500)
        self.enc4 = nn.Linear(1500, 1000)
        self.enc5 = nn.Linear(1000, 500)
        self.enc6 = nn.Linear(500,L)

        self.dec1 = nn.Linear(L, 500)
        self.dec2 = nn.Linear(500, 1000)
        self.dec3 = nn.Linear(1000, 1500)
        self.dec4 = nn.Linear(1500, 2000)
        self.dec5 = nn.Linear(2000, 2500)
        self.dec6 = nn.Linear(2500, input_dim)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.sigmoid(self.dropout(self.enc1(x)))
        x = F.sigmoid(self.dropout(self.enc2(x)))
        x = F.sigmoid(self.dropout(self.enc3(x)))
        x = F.sigmoid(self.dropout(self.enc4(x)))
        x = F.sigmoid(self.dropout(self.enc5(x)))
        x = F.sigmoid(self.dropout(self.enc6(x)))

        x = F.sigmoid(self.dropout(self.dec1(x)))
        x = F.sigmoid(self.dropout(self.dec2(x)))
        x = F.sigmoid(self.dropout(self.dec3(x)))
        x = F.sigmoid(self.dropout(self.dec4(x)))
        x = F.sigmoid(self.dropout(self.dec5(x)))
        x = F.sigmoid(self.dropout(self.dec6(x)))
        return x

    def get_latent_space(self, x):
        x = F.sigmoid(self.dropout(self.enc1(x)))
        x = F.sigmoid(self.dropout(self.enc2(x)))
        x = F.sigmoid(self.dropout(self.enc3(x)))
        x = F.sigmoid(self.dropout(self.enc4(x)))
        x = F.sigmoid(self.dropout(self.enc5(x)))
        x = F.sigmoid(self.dropout(self.enc6(x)))
        return x
