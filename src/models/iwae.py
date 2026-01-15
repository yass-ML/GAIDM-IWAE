import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.vae import VAE

import numpy as np
import math



class IWAE(VAE):
    def __init__(self, K, input_size, hidden_size, latent_size, output_size):
        super().__init__(input_size=input_size, hidden_size=hidden_size,latent_size=latent_size,output_size=output_size)
        self.K = K


    def forward(self, x):
        x = x.view(x.size(0), -1)
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=1) # Size (B, Latent)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(self.K, x.size(0), self.latent_size).to(x.device)

        z = mu + eps * std

        z_flat = z.view(-1, self.latent_size)
        recon_x_flat = self.decoder(z_flat)
        recon_x = recon_x_flat.view(self.K, x.size(0), -1)

        return recon_x, mu, logvar, z


    def compute_loss(self, x, recon_x, mu, logvar, z):
        x = x.view(x.size(0), -1)
        x_k = x.unsqueeze(0).repeat(self.K, 1, 1)

        log_p_x_given_z = -F.binary_cross_entropy(recon_x,x_k, reduction="none").sum(dim=2)

        log_q_z_given_x = -0.5 * (torch.log(2 * torch.tensor(np.pi)) + logvar + (z - mu).pow(2) / torch.exp(logvar))
        log_q_z_given_x = log_q_z_given_x.sum(dim=2) # (K, B)

        log_p_z = -0.5 * (torch.log(2 * torch.tensor(np.pi)) + z.pow(2))
        log_p_z = log_p_z.sum(dim=2) #  (K, B)

        log_w = log_p_x_given_z + log_p_z - log_q_z_given_x

        loss = - (torch.logsumexp(log_w, dim=0) - torch.log(torch.tensor(self.K))).mean()

        return loss
