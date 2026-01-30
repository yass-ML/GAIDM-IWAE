import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.vae import VAE # Hérite de la structure de base du VAE

import numpy as np
import math

class IWAE(VAE):
    def __init__(self, K, input_size, hidden_size, latent_size, output_size):
        """
        Initialise l'IWAE.
        Args:
            K (int): Nombre d'échantillons d'importance par donnée.
        """
        super().__init__(input_size=input_size, hidden_size=hidden_size, 
                         latent_size=latent_size, output_size=output_size)
        self.K = K

    def forward(self, x):
        """
        Passage vers l'avant avec échantillonnage multiple.
        """
        # Aplatissement de l'entrée : (Batch, ...) -> (Batch, Input_size)
        x = x.view(x.size(0), -1)
        
        # Encodage pour obtenir les paramètres de la distribution latente
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=1) # Dimensions : (Batch, Latent)
        std = torch.exp(0.5 * logvar)
        
        # --- ÉCHANTILLONNAGE MULTIPLE (K échantillons) ---
        # On génère K bruits blancs pour chaque élément du batch.
        # Dimension finale de eps : (K, Batch, Latent)
        eps = torch.randn(self.K, x.size(0), self.latent_size).to(x.device)

        # Reparamétrisation : z a maintenant une dimension (K, Batch, Latent)
        z = mu + eps * std

        # --- DÉCODAGE MASSIVE ---
        # On aplatit z pour passer tous les échantillons (K * Batch) dans le décodeur d'un coup
        z_flat = z.view(-1, self.latent_size)
        recon_x_flat = self.decoder(z_flat)
        
        # On redonne à la reconstruction sa forme multi-échantillons : (K, Batch, Output_size)
        recon_x = recon_x_flat.view(self.K, x.size(0), -1)

        return recon_x, mu, logvar, z

    def compute_loss(self, x, recon_x, mu, logvar, z):
        """
        Calcule la perte IWAE basée sur le Log-Sum-Exp.
        L'objectif est de maximiser la borne : E[log( (1/K) * sum(p(x,z)/q(z|x)) )]
        """
        x = x.view(x.size(0), -1)
        # Répète x pour correspondre aux K échantillons de recon_x : (K, Batch, Input_size)
        x_k = x.unsqueeze(0).repeat(self.K, 1, 1)

        # 1. Log p(x | z) : Log-vraisemblance de la reconstruction (Log-Bernoulli)
        # On somme sur la dimension des pixels (dim=2)
        log_p_x_given_z = -F.binary_cross_entropy(recon_x, x_k, reduction="none").sum(dim=2)

        # 2. Log q(z | x) : Log-densité de la distribution de l'encodeur (Gaussienne)
        # Formule : log N(z; mu, sigma^2)
        log_q_z_given_x = -0.5 * (torch.log(2 * torch.tensor(np.pi)) + logvar + (z - mu).pow(2) / torch.exp(logvar))
        log_q_z_given_x = log_q_z_given_x.sum(dim=2) # Somme sur les dimensions latentes -> (K, Batch)

        # 3. Log p(z) : Log-densité de la priorité (Prior) N(0, 1)
        log_p_z = -0.5 * (torch.log(2 * torch.tensor(np.pi)) + z.pow(2))
        log_p_z = log_p_z.sum(dim=2) # Somme sur les dimensions latentes -> (K, Batch)

        # --- CALCUL DES POIDS D'IMPORTANCE ---
        # log_w = log( p(x,z) / q(z|x) ) = log p(x|z) + log p(z) - log q(z|x)
        log_w = log_p_x_given_z + log_p_z - log_q_z_given_x

        # --- LOG-SUM-EXP TRICK ---
        # Pour éviter les instabilités numériques, on utilise logsumexp pour calculer le log de la moyenne
        # loss = - Moyenne_sur_batch ( log ( (1/K) * sum(exp(log_w)) ) )
        loss = - (torch.logsumexp(log_w, dim=0) - torch.log(torch.tensor(float(self.K)))).mean()

        return loss