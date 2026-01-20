
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import argparse

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.models.vae import VAE
from src.models.iwae import IWAE
from src.data.mnist_loader import get_dataloaders
from src.analysis.gradient_variance import compute_gradient_variance
from src.utils.checkpoint_utils import get_models_config, save_results, get_model_key


def analyze_tradeoff(
    checkpoint_dir: str = 'checkpoints',
    results_path: str = 'results/evaluations.yaml',
    output_path: str = './notebook_results/tradeoff_analysis.png'
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Config
    input_size = 784
    hidden_size = 200
    latent_size = 50
    output_size = 784

    # Get models config from checkpoint discovery + stored results
    models_config = get_models_config(checkpoint_dir, results_path)

    if not models_config:
        print(f"No checkpoints found in {checkpoint_dir}")
        print("Run trainer.py first to create checkpoints.")
        return

    # Filter for valid models (with parseable k values)
    models_config = [m for m in models_config if m.get('k') is not None]

    if not models_config:
        print("No valid checkpoints found with standard naming convention.")
        return

    # Check for missing log-likelihoods
    missing_ll = [m['name'] for m in models_config if m.get('log_likelihood') is None]
    if missing_ll:
        print(f"\nWarning: The following models are missing log-likelihood values:")
        for name in missing_ll:
            print(f"  - {name}")
        print(f"\nRun: python src/analysis/evaluate_likelihood.py --checkpoint_dir {checkpoint_dir}")
        print("to compute log-likelihoods first.\n")

    # Data (single batch for gradient variance)
    train_loader, _, _ = get_dataloaders(batch_size=32)
    data, _ = next(iter(train_loader))
    data = data.to(device)

    results = []

    print("-" * 70)
    print(f"{'Model':<15} | {'Test LL':<12} | {'Grad Var':<12} | {'Grad SNR':<10}")
    print("-" * 70)

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

        # Save gradient metrics to YAML
        model_key = get_model_key(cfg['path'])
        save_results(
            results_path=results_path,
            model_key=model_key,
            metrics={
                'gradient_variance': float(f"{variance:.2e}"),
                'gradient_snr': round(snr, 4)
            }
        )

        ll_value = cfg.get('log_likelihood')
        ll_display = f"{ll_value:.2f}" if ll_value is not None else "N/A"

        results.append({
            'name': cfg['name'],
            'k': cfg['k'],
            'll': ll_value,
            'variance': variance,
            'snr': snr
        })

        print(f"{cfg['name']:<15} | {ll_display:<12} | {variance:<12.2e} | {snr:<10.4f}")

    print("-" * 70)

    # Filter results with LL for plotting
    results_with_ll = [r for r in results if r['ll'] is not None]

    if not results_with_ll:
        print("\nNo models have log-likelihood computed. Run evaluate_likelihood.py first.")
        print("Skipping plot generation.")
        return

    # Sort by K for proper line plot
    results_with_ll.sort(key=lambda x: x['k'])

    # Plotting
    ks = [r['k'] for r in results_with_ll]
    lls = [r['ll'] for r in results_with_ll]
    snrs = [r['snr'] for r in results_with_ll]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Number of Importance Samples (K)')
    ax1.set_ylabel('Test Log-Likelihood (Higher is Better)', color=color)
    ax1.plot(ks, lls, marker='o', color=color, linewidth=2, label='Log-Likelihood')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(ks)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()

    color = 'tab:red'
    ax2.set_ylabel('Gradient SNR (Higher is Stable)', color=color)
    ax2.plot(ks, snrs, marker='s', color=color, linewidth=2, linestyle='--', label='Gradient SNR')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Trade-off: Performance vs Optimization Stability')
    fig.tight_layout()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"\nTrade-off plot saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Performance vs Stability Trade-off")
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory containing model checkpoints')
    parser.add_argument('--results_path', type=str, default='results/evaluations.yaml',
                        help='Path to evaluations YAML file')
    parser.add_argument('--output', type=str, default='./results/tradeoff_analysis.png',
                        help='Path to save the plot')

    args = parser.parse_args()

    analyze_tradeoff(
        checkpoint_dir=args.checkpoint_dir,
        results_path=args.results_path,
        output_path=args.output
    )
