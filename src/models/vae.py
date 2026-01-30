import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, output_size):
        """
        Initialise l'architecture du VAE.
        
        Args:
            input_size (int): Dimension de l'entrée (ex: 784 pour MNIST 28x28).
            hidden_size (int): Nombre de neurones dans les couches cachées.
            latent_size (int): Dimension de l'espace latent (le 'goulot d'étranglement').
            output_size (int): Dimension de la sortie reconstruite (généralement = input_size).
        """
        super().__init__()
        
        # --- ENCODEUR ---
        # Compresse l'entrée vers l'espace latent
        self.encoder = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.Tanh(),
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.Tanh(),
            # La sortie finale de l'encodeur est 2 * latent_size car on doit
            # prédire à la fois la moyenne (mu) et le log-variance (logvar).
            nn.Linear(in_features=hidden_size, out_features=2 * latent_size)
        )

        # --- DÉCODEUR ---
        # Reconstruit l'image à partir d'un point de l'espace latent
        self.decoder = nn.Sequential(
            nn.Linear(in_features=latent_size, out_features=hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.Tanh(),
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.Tanh(),
            nn.Linear(in_features=hidden_size, out_features=output_size),
            # Sigmoid assure que la sortie est entre 0 et 1 (idéal pour des pixels normalisés).
            nn.Sigmoid() 
        )

        self.latent_size = latent_size

    def forward(self, x):
        """
        Passage vers l'avant (Forward pass).
        """
        # Aplatit l'entrée au cas où elle arrive sous forme d'image (B, C, H, W) -> (B, D)
        x = x.view(x.size(0), -1)
        
        # 1. ENCODAGE : Extraction des paramètres de la distribution gaussienne
        h = self.encoder(x)
        # Sépare le vecteur en deux : mu (moyenne) et logvar (log-variance)
        mu, logvar = h.chunk(2, dim=1)

        # 2. REPARAMÉTRISATION (The Reparameterization Trick)
        # On ne peut pas faire de backpropagation à travers un échantillonnage aléatoire.
        # On utilise donc : z = mu + sigma * epsilon, où epsilon ~ N(0, 1)
        std = torch.exp(0.5 * logvar) # Calcul de l'écart-type (sigma)
        eps = torch.randn_like(std)   # Échantillonnage d'un bruit blanc
        z = mu + std * eps            # Échantillonnage de l'espace latent

        # 3. DÉCODAGE : Reconstruction à partir de z
        recon_x = self.decoder(z)
        
        return recon_x, mu, logvar, z

    def compute_loss(self, x, recon_x, mu, logvar):
        """
        Calcule la perte ELBO (Evidence Lower Bound).
        Loss = Erreur de reconstruction + Divergence KL
        """
        x = x.view(x.size(0), -1)
        
        # A. Perte de Reconstruction (BCE) : Compare l'original et la copie
        # On utilise la somme (reduction='sum') pour rester cohérent avec la formule mathématique de la KL.
        bce = F.binary_cross_entropy(recon_x, x, reduction='sum')

        # B. Divergence de Kullback-Leibler (KLD)
        # Force la distribution latente apprise à être proche d'une distribution normale standard N(0,1).
        # Formule fermée : -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return bce + kld