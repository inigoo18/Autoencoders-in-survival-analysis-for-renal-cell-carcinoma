from typing import Type

import torch


class MWE_AE(torch.nn.Module):

    def __init__(self, input_dim, L):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            custom_block(input_dim, 1500),
            custom_block_final_dropout(1500, 500),
            custom_block_final(500, L)
        )

        self.decoder = torch.nn.Sequential(
            custom_block_final_dropout(L, 500),
            custom_block_final_dropout(500, 1500),
            custom_block_final(1500, input_dim)
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
        torch.nn.Tanh(),
        torch.nn.Dropout(dropout_rate),
    )

def custom_block_final_dropout(input_dim, output_dim, dropout_rate = 0.5):
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, output_dim),
        torch.nn.Tanh(),
        torch.nn.Dropout(dropout_rate)
    )

def custom_block_final(input_dim, output_dim):
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, output_dim),
        torch.nn.Sigmoid(),
    )