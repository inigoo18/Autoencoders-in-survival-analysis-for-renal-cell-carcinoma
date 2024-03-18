from typing import Type

import torch


class MWE_AE(torch.nn.Module):

    def __init__(self, input_dim, L):
        super().__init__()

        print("Initializing Minimal Working Example AE with input dim: ", input_dim)

        self.encoder = torch.nn.Sequential(
            custom_block(input_dim, 3000),
            custom_block(3000, 2500),
            custom_block(2500, 2000),
            custom_block(2000, 1500),
            custom_block(1500, 1200),
            custom_block(1200, 1000),
            custom_block(1000, 800),
            custom_block(800, 600),
            custom_block(600, L),
            torch.nn.Sigmoid()
        )

        self.decoder = torch.nn.Sequential(
            custom_block(L, 600),
            custom_block(600, 800),
            custom_block(800, 1000),
            custom_block(1000, 1200),
            custom_block(1200, 1500),
            custom_block(1500, 2000),
            custom_block(2000, 2500),
            custom_block(2500, 3000),
            custom_block(3000, input_dim),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)

        decoded = self.decoder(encoded)
        return decoded

    def get_latent_space(self, x):
        return self.encoder(x)


def custom_block(input_dim, output_dim, dropout_rate=0.1):
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, output_dim),
        torch.nn.BatchNorm1d(output_dim),
        torch.nn.PReLU(),
        torch.nn.Dropout(dropout_rate)
    )
