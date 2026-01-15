
import torch
import torch.nn as nn
import argparse
import os
import sys
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.vae import VAE
from src.models.iwae import IWAE
from src.data.mnist_loader import get_dataloaders

def compute_gradient_variance(model, data, n_runs=100):
    """
    Computes the variance of the gradients for the encoder parameters
    over multiple runs on the SAME batch of data (varying random sampling).
    """
    model.train() # Make sure we can compute gradients

    # We will track gradients for the first encoder layer
    # layer 0 is Linear(784 -> 200)
    target_layer_param = model.encoder[0].weight

    grads = []

    print(f"Collecting gradients over {n_runs} runs...")
    for _ in range(n_runs):
        model.zero_grad()

        # Forward
        recon_x, mu, logvar, z = model(data)

        # Loss
        loss = model.compute_loss(data, recon_x, mu, logvar, z)

        # Backward
        loss.backward()

        # Save grad copy
        if target_layer_param.grad is not None:
            grads.append(target_layer_param.grad.clone().cpu().numpy())

    # Convert to array: (n_runs, out_features, in_features)
    grads = np.array(grads)
    # Shape: (100, 200, 784)

    # 1. Compute Variance per parameter
    # shape: (200, 784)
    var_per_param = np.var(grads, axis=0)

    # 2. Compute Mean gradient (Signal)
    mean_grad = np.mean(grads, axis=0)

    # 3. Overall stats
    # Average Variance across all parameters in the layer
    avg_variance = np.mean(var_per_param)

    # SNR (Signal to Noise Ratio): Mean( |Mean| / Std )
    # Avoid div by zero
    std_per_param = np.std(grads, axis=0) + 1e-10
    snr = np.mean(np.abs(mean_grad) / std_per_param)

    return avg_variance, snr

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--model_type', type=str, default='vae', choices=['vae', 'iwae'])
    parser.add_argument('--k', type=int, default=1)
    parser.add_argument('--n_runs', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    # Load Model
    input_size = 784
    output_size = 784
    hidden_size = 200
    latent_size = 50

    if args.model_type == 'vae':
         model = VAE(input_size, hidden_size, latent_size, output_size)
    else:
         model = IWAE(args.k, input_size, hidden_size, latent_size, output_size)

    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    model.to(args.device)

    # Data: Get one batch
    train_loader, _, _ = get_dataloaders(batch_size=32)
    data, _ = next(iter(train_loader))
    data = data.to(args.device)

    # Analyze
    print(f"Analyzing Gradient Variance for {args.model_type.upper()}...")
    variance, snr = compute_gradient_variance(model, data, n_runs=args.n_runs)

    print("-" * 40)
    print(f"Gradient Stats (Encoder Layer 1):")
    print(f"Average Variance: {variance:.2e}")
    print(f"Signal-to-Noise Ratio (SNR): {snr:.4f}")
    print("-" * 40)
