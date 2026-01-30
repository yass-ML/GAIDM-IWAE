import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self,input_size, hidden_size, latent_size, output_size):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.Tanh(),
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.Tanh(),
            nn.Linear(in_features=hidden_size, out_features=2*latent_size)
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features=latent_size, out_features=hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.Tanh(),
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.Tanh(),
            nn.Linear(in_features=hidden_size, out_features=output_size),
            nn.Sigmoid()
        )

        self.latent_size = latent_size


    def forward(self,x):
        x = x.view(x.size(0), -1)
        h = self.encoder(x)

        mu, logvar = h.chunk(2,dim=1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        z = mu + std * eps

        return self.decoder(z), mu, logvar, z

    def compute_loss(self, x, recon_x, mu, logvar, z=None):
        x = x.view(x.size(0), -1)
        bce = F.binary_cross_entropy(recon_x,x, reduction='sum')
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return bce + kld
