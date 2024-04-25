import torch
import torch.nn as nn
import torch.nn.functional as F

class VariationalExample(nn.Module):

    def __init__(self, input_dim, L):
        super(VariationalExample, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.encoder = torch.nn.Sequential(
            custom_block(input_dim, 2000),
            custom_block(2000, 1000),
            custom_block(1000, 500),
        )

        self.decoder = torch.nn.Sequential(
            custom_block(L, 500),
            custom_block(500, 1000),
            custom_block(1000, 2000),
            custom_block_decoder(2000, input_dim)
        )

        # latent mean and variance
        self.mean_layer = nn.Linear(500, L)
        self.logvar_layer = nn.Linear(500, L)


    def encode(self, x):
        x = self.encoder(x)
        mean, log_var = self.mean_layer(x), self.logvar_layer(x)
        return mean, log_var


    def get_latent_space(self, x):
        encoded = self.encoder(x)
        mean, log_var = self.mean_layer(encoded), self.logvar_layer(encoded)
        return self.reparameterization(mean, torch.exp(0.5 * log_var)) # log var -> var


    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(self.device)
        z = mean + var * epsilon
        z = (z - z.min()) / (z.max() - z.min())
        return z

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterization(mean, log_var)
        x_hat = self.decode(z)
        return x_hat, mean, log_var

def custom_block(input_dim, output_dim, dropout_rate=0.35):
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, output_dim),
        torch.nn.BatchNorm1d(output_dim),
        torch.nn.PReLU(),
        torch.nn.Dropout(dropout_rate)
    )

def custom_block_encoder(input_dim, output_dim):
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, output_dim),
        torch.nn.Sigmoid()
    )

def custom_block_decoder(input_dim, output_dim, dropout_rate = 0.25):
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, output_dim),
        torch.nn.Dropout(dropout_rate),
        torch.nn.Sigmoid()
    )