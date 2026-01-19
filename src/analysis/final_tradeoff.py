
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.models.vae import VAE
from src.models.iwae import IWAE
from src.data.mnist_loader import get_dataloaders
from src.analysis.gradient_variance import compute_gradient_variance

def analyze_tradeoff():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Config
    input_size = 784
    hidden_size = 200
    latent_size = 50
    output_size = 784

    # Models to evaluate
    models_config = [
        {'name': 'VAE (K=1)', 'type': 'vae', 'k': 1, 'path': 'checkpoints/vae_k1_epochs50_seed42.pt', 'll': -81.06},
        {'name': 'IWAE (K=5)', 'type': 'iwae', 'k': 5, 'path': 'checkpoints/iwae_k5_epochs50_seed42.pt', 'll': -78.36},
        {'name': 'IWAE (K=20)', 'type': 'iwae', 'k': 20, 'path': 'checkpoints/iwae_k20_epochs50_seed42.pt', 'll': -77.33}
    ]

    # Data (single batch for gradient variance)
    train_loader, _, _ = get_dataloaders(batch_size=32)
    data, _ = next(iter(train_loader))
    data = data.to(device)

    results = []

    print("-" * 60)
    print(f"{'Model':<15} | {'Test LL':<10} | {'Grad Var':<10} | {'Grad SNR':<10}")
    print("-" * 60)

    for cfg in models_config:
        # Load Model
        if cfg['type'] == 'vae':
            model = VAE(input_size, hidden_size, latent_size, output_size)
        else:
            model = IWAE(cfg['k'], input_size, hidden_size, latent_size, output_size)

        try:
            model.load_state_dict(torch.load(cfg['path'], map_location=device))
        except FileNotFoundError:
            print(f"Checkpoint not found: {cfg['path']}")
            continue

        model.to(device)

        # Compute Gradient Variance
        variance, snr = compute_gradient_variance(model, data, n_runs=100)

        results.append({
            'k': cfg['k'],
            'll': cfg['ll'],
            'variance': variance,
            'snr': snr
        })

        print(f"{cfg['name']:<15} | {cfg['ll']:<10.2f} | {variance:<10.2e} | {snr:<10.4f}")

    print("-" * 60)

    # Plotting
    ks = [r['k'] for r in results]
    lls = [r['ll'] for r in results]
    snrs = [r['snr'] for r in results]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Number of Importance Samples (K)')
    ax1.set_ylabel('Test Log-Likelihood (Higher is Better)', color=color)
    ax1.plot(ks, lls, marker='o', color=color, linewidth=2, label='Log-Likelihood')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(ks)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:red'
    ax2.set_ylabel('Gradient SNR (Higher is Stable)', color=color)  # we already handled the x-label with ax1
    ax2.plot(ks, snrs, marker='s', color=color, linewidth=2, linestyle='--', label='Gradient SNR')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Trade-off: Performance vs Optimization Stability')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    output_path = './notebook_results/tradeoff_analysis.png'
    plt.savefig(output_path)
    print(f"Trade-off plot saved to {output_path}")

if __name__ == "__main__":
    analyze_tradeoff()
