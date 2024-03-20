import torch
import torch.nn as nn

class VariationalExample(nn.Module):

    def __init__(self, input_dim, L):
        super(VariationalExample, self).__init__()

        # encoder
        self.encoder = nn.Sequential(
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

        # latent mean and variance
        self.mean_layer = nn.Linear(L, 2)
        self.logvar_layer = nn.Linear(L, 2)

        # decoder
        self.decoder = nn.Sequential(
            custom_block(2, L),
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

    def encode(self, x):
        x = self.encoder(x)
        mean, log_var = self.mean_layer(x), self.logvar_layer(x)
        return mean, log_var

    def get_latent_space(self, x):
        return self.encoder(x)

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var)
        z = mean + var * epsilon
        return z

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterization(mean, log_var)
        x_hat = self.decode(z)
        return x_hat, mean, log_var

def custom_block(input_dim, output_dim, dropout_rate=0.1):
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, output_dim),
        torch.nn.BatchNorm1d(output_dim),
        torch.nn.PReLU(),
        torch.nn.Dropout(dropout_rate)
    )