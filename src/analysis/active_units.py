
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import argparse
import os
import sys
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.vae import VAE
from src.models.iwae import IWAE
from src.data.mnist_loader import get_dataloaders

def compute_batch_kl(mu, logvar):
    """
    Computes individual KL divergence for each dimension z_j.
    Formula: KL = -0.5 * (1 + logvar - mu^2 - exp(logvar))

    Args:
        mu: (BATCH_SIZE, LATENT_SIZE)
        logvar: (BATCH_SIZE, LATENT_SIZE)

    Returns:
        kl_per_dim: (LATENT_SIZE,) - Average KL for each dimension across the batch
    """
    # 1. Compute KL for every element in the batch and every dimension
    # Shape: (B, L)
    kl_elementwise = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())

    # 2. Average across the batch (but keep dimensions separate!)
    # Shape: (L,)
    kl_per_dim = kl_elementwise.mean(dim=0)

    return kl_per_dim

def calc_active_units(model, dataloader, device, threshold=0.01):
    """
    Runs over the dataset and computes the number of active units.
    A unit is active if Avg_KL(z_j) > threshold.
    """
    model.eval()
    total_kl = 0
    num_batches = 0

    with torch.no_grad():
        for data, _ in tqdm(dataloader, desc="Checking Latents", leave=False):
            data = data.to(device)

            # Forward pass just to get mu and logvar
            # Note: We don't care about K samples here, we just want q(z|x) parameters
            if isinstance(model, IWAE):
                # IWAE encoder works same as VAE: x -> mu, logvar
                x_flat = data.view(data.size(0), -1)
                h = model.encoder(x_flat)
                mu, logvar = h.chunk(2, dim=1)
            else:
                _, mu, logvar, _ = model(data)

            # Compute KL for this batch
            batch_kl = compute_batch_kl(mu, logvar)
            total_kl += batch_kl
            num_batches += 1

    # Average over all batches
    final_avg_kl = total_kl / num_batches

    # Count active units (KL > threshold)
    num_active = (final_avg_kl > threshold).sum().item()

    return num_active, final_avg_kl

def plot_kl_stats(avg_kl_values, model_name, save_path):
    """
    Plots the KL value for each dimension sorted.
    """
    # Sort values to make the plot readable (Scree plot style)
    sorted_kl, _ = torch.sort(avg_kl_values, descending=True)
    sorted_kl = sorted_kl.cpu().numpy()

    plt.figure(figsize=(8, 4))
    plt.bar(range(len(sorted_kl)), sorted_kl)
    plt.axhline(y=0.01, color='r', linestyle='--', label='Threshold (0.01)')
    plt.xlabel('Latent Dimensions (Sorted)')
    plt.ylabel('Average KL Divergence (nats)')
    plt.title(f'Effective KL per Dimension - {model_name}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--model_type', type=str, default='vae', choices=['vae', 'iwae'])
    parser.add_argument('--k', type=int, default=1)
    parser.add_argument('--threshold', type=float, default=0.01, help='Threshold for active unit')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_dir', type=str, default='./results')

    args = parser.parse_args()

    # Init
    input_size = 784
    output_size = 784
    hidden_size = 200 # Assuming standard
    latent_size = 50   # Assuming standard

    if args.model_type == 'vae':
         model = VAE(input_size, hidden_size, latent_size, output_size)
    else:
         model = IWAE(args.k, input_size, hidden_size, latent_size, output_size)

    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    model.to(args.device)

    # Data
    _, _, test_loader = get_dataloaders(batch_size=128)

    # Calculate
    print(f"Analyzing {args.model_type.upper()}...")
    n_active, avg_kls = calc_active_units(model, test_loader, args.device, args.threshold)

    print(f"-" * 40)
    print(f"Total Latent Dimensions: {latent_size}")
    print(f"Active Units (KL > {args.threshold}): {n_active}")
    print(f"-" * 40)

    # Plot
    os.makedirs(args.output_dir, exist_ok=True)
    base_name = os.path.basename(args.model_path).replace('.pt', '')
    plot_kl_stats(avg_kls, f"{args.model_type.upper()} (Active: {n_active})",
                  os.path.join(args.output_dir, f"{base_name}_kl.png"))
