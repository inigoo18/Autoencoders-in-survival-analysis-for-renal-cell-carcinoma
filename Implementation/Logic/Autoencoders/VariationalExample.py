import torch
import torch.nn as nn

class VariationalExample(nn.Module):

    def __init__(self, input_dim, L):
        super(VariationalExample, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.encoder = torch.nn.Sequential(
            custom_block(input_dim, 1500),
            custom_block_final_dropout(1500, 500),
        )

        # Latent mean and variance
        self.mean_layer = nn.Linear(500, L)
        self.logvar_layer = nn.Linear(500, L)

        self.decoder = torch.nn.Sequential(
            custom_block_final_dropout(L, 500),
            custom_block_final_dropout(500, 1500),
            custom_block_final(1500, input_dim)
        )


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

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterization(mean, log_var)
        x_hat = self.decoder(z)
        return x_hat, mean, log_var


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
        torch.nn.Sigmoid(),
        torch.nn.Dropout(dropout_rate)
    )

def custom_block_final(input_dim, output_dim):
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, output_dim),
        torch.nn.Sigmoid(),
    )