import torch
import torch.nn as nn

class VariationalExample(nn.Module):

    def __init__(self, input_dim, L):
        super(VariationalExample, self).__init__()

        # encoder
        self.encoder = nn.Sequential(
            torch.nn.Linear(input_dim, 2500),
            torch.nn.BatchNorm1d(2500),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(2500, 2000),
            torch.nn.BatchNorm1d(2000),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(2000, 1500),
            torch.nn.BatchNorm1d(1500),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(1500, 1000),
            torch.nn.BatchNorm1d(1000),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(1000, 500),
            torch.nn.BatchNorm1d(500),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(500, L),
            torch.nn.BatchNorm1d(L),
            torch.nn.ReLU()
        )

        # latent mean and variance
        self.mean_layer = nn.Linear(L, 2)
        self.logvar_layer = nn.Linear(L, 2)

        # decoder
        self.decoder = nn.Sequential(
            torch.nn.Linear(2, L),
            torch.nn.ReLU(),
            torch.nn.Linear(L, 500),
            torch.nn.BatchNorm1d(500),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(500, 1000),
            torch.nn.BatchNorm1d(1000),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(1000, 1500),
            torch.nn.BatchNorm1d(1500),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(1500, 2000),
            torch.nn.BatchNorm1d(2000),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(2000, 2500),
            torch.nn.BatchNorm1d(2500),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(2500, input_dim),
            torch.nn.BatchNorm1d(input_dim),
            torch.nn.ReLU()
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