from typing import Type

import torch


class MWE_AE(torch.nn.Module):

    def __init__(self, input_dim, L):
        super().__init__()

        print("Initializing Minimal Working Example AE with input dim: ", input_dim)

        self.encoder = torch.nn.Sequential(
            custom_block(input_dim, 2000),
            custom_block(2000, 1000),
            custom_block(1000, 500),
            custom_block_encoder(500, L)
        )

        self.decoder = torch.nn.Sequential(
            custom_block(L, 500),
            custom_block(500, 1000),
            custom_block(1000, 2000),
            custom_block_decoder(2000, input_dim)
        )


    def forward(self, x):
        encoded = self.encoder(x)

        decoded = self.decoder(encoded)
        return decoded

    def get_latent_space(self, x):
        return self.encoder(x)


def custom_block(input_dim, output_dim, dropout_rate=0.2):
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, output_dim),
        torch.nn.BatchNorm1d(output_dim),
        torch.nn.Dropout(dropout_rate),
        torch.nn.Tanh(),
        #torch.nn.Tanh(),
    )

def custom_block_encoder(input_dim, output_dim, dropout_rate = 0.1):
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, output_dim),
        torch.nn.Sigmoid(),
        torch.nn.Dropout(dropout_rate),
    )

def custom_block_decoder(input_dim, output_dim, dropout_rate = 0.1):
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, output_dim),
        torch.nn.Sigmoid(),
        torch.nn.Dropout(dropout_rate),
    )