from typing import Type

import torch


class MWE_AE(torch.nn.Module):

    def __init__(self, input_dim, L):
        super().__init__()

        print("Initializing Minimal Working Example AE with input dim: ", input_dim)

        self.encoder = torch.nn.Sequential(
            custom_block(input_dim, 2000),
            custom_block_final_dropout(2000, 500),
            custom_block_final(500, L)
        )

        self.decoder = torch.nn.Sequential(
            custom_block(L, 500),
            custom_block_final_dropout(500, 2000),
            custom_block_final(2000, input_dim)
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
        torch.nn.Tanh(),
        torch.nn.Dropout(dropout_rate),
        #torch.nn.Tanh(),
    )

def custom_block_final_dropout(input_dim, output_dim, dropout_rate = 0.1):
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, output_dim),
        torch.nn.Sigmoid(),
        torch.nn.Dropout(dropout_rate)
    )

def custom_block_final(input_dim, output_dim):
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, output_dim),
        torch.nn.Sigmoid(),
    )