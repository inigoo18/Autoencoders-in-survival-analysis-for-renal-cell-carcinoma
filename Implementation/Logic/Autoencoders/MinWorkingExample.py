from typing import Type

import torch


class MWE_AE(torch.nn.Module):

    def __init__(self, input_dim, L):
        super().__init__()

        print("Initializing Minimal Working Example AE with input dim: ", input_dim)

        self.encoder = torch.nn.Sequential(
            custom_block(input_dim, 2000),
            custom_block(2000, 1500),
            custom_block(1500, 1000),
            custom_block(1000, 800),
            custom_block_encoder(800, 500),
            custom_block_encoder(500, L)
        )

        self.decoder = torch.nn.Sequential(
            custom_block(L, 500),
            custom_block(500, 800),
            custom_block(800, 1000),
            custom_block(1000, 1500),
            custom_block_decoder(1500, 2000),
            custom_block_decoder(2000, input_dim)
        )


    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def get_latent_space(self, x):
        return self.encoder(x)


def custom_block(input_dim, output_dim, dropout_rate=0.15):
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, output_dim),
        torch.nn.BatchNorm1d(output_dim),
        torch.nn.Tanh(),
        torch.nn.Dropout(dropout_rate),
        #torch.nn.Tanh(),
    )

def custom_block_encoder(input_dim, output_dim, dropout_rate = 0.05):
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, output_dim),
        torch.nn.Sigmoid(),
        torch.nn.Dropout(dropout_rate),
    )

def custom_block_decoder(input_dim, output_dim, dropout_rate = 0.05):
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, output_dim),
        torch.nn.Sigmoid(),
    )