from __future__ import print_function
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, x_dim=784, h_dim=400, z_dim=20):
        super(VAE, self).__init__()

        self.input_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim

        # encoder
        self.fc1 = nn.Linear(x_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(h_dim, z_dim)
        # decoder
        self.fc4 = nn.Linear(z_dim, h_dim)
        self.fc5 = nn.Linear(h_dim, x_dim)

    def forward(self, x):
        batch_size = x.shape[0]
        mu, log_var = self.encode(x.view(batch_size, self.input_dim))
        sampled_z = self.reparameterization(mu, log_var)
        recon_x = self.decode(sampled_z).view(batch_size, 1, 28, 28)
        return recon_x, mu, log_var

    def encode(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc2(h)
        log_var = self.fc3(h)

        return mu, log_var

    def reparameterization(self, mu, log_var):
        sigma = torch.exp(log_var * 0.5)
        eps = torch.randn_like(sigma)
        return mu + sigma * eps

    def decode(self, z):
        h = F.relu(self.fc4(z))
        x_hat = torch.sigmoid(self.fc5(h))
        return x_hat


def loss_function(recon_x, x, mu, logVar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    # BCE = F.mse_loss(recon_x, x)
    KLD_element = (-1 * ((mu ** 2) + logVar.exp())) + 1 + logVar
    KLD = torch.sum(KLD_element).mul_(-0.5)

    return BCE + KLD