
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
from src.utils.checkpoint_utils import (
    discover_checkpoints,
    save_results,
    get_model_key
)


def compute_gradient_variance(model, data, n_runs=100):
    """
    Computes the variance of the gradients for the encoder parameters
    over multiple runs on the SAME batch of data (varying random sampling).
    """
    model.train()

    # Target the first encoder layer
    target_layer_param = model.encoder[0].weight
    grads = []

    for _ in range(n_runs):
        model.zero_grad()
        recon_x, mu, logvar, z = model(data)
        loss = model.compute_loss(data, recon_x, mu, logvar, z)
        loss.backward()

        if target_layer_param.grad is not None:
            grads.append(target_layer_param.grad.clone().cpu().numpy())

    grads = np.array(grads)

    var_per_param = np.var(grads, axis=0)
    mean_grad = np.mean(grads, axis=0)
    avg_variance = np.mean(var_per_param)

    std_per_param = np.std(grads, axis=0) + 1e-10
    snr = np.mean(np.abs(mean_grad) / std_per_param)

    return avg_variance, snr


def analyze_single_checkpoint(
    checkpoint_path: str,
    model_type: str,
    k: int,
    device: str,
    data: torch.Tensor,
    n_runs: int = 50,
    results_path: str = None,
    hidden_size: int = 200,
    latent_size: int = 50
) -> tuple:
    """Analyze gradient variance for a single checkpoint."""
    input_size = 784
    output_size = 784

    if model_type == 'vae':
        model = VAE(input_size, hidden_size, latent_size, output_size)
    else:
        model = IWAE(k, input_size, hidden_size, latent_size, output_size)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)

    variance, snr = compute_gradient_variance(model, data, n_runs=n_runs)

    # Save to YAML if path provided
    if results_path:
        model_key = get_model_key(checkpoint_path)
        save_results(
            results_path=results_path,
            model_key=model_key,
            metrics={
                'gradient_variance': float(f"{variance:.2e}"),
                'gradient_snr': round(snr, 4)
            }
        )

    return variance, snr


def analyze_all_checkpoints(
    checkpoint_dir: str,
    device: str,
    data: torch.Tensor,
    n_runs: int = 50,
    results_path: str = 'results/evaluations.yaml'
):
    """Discover and analyze all checkpoints."""
    checkpoints = discover_checkpoints(checkpoint_dir)

    if not checkpoints:
        print(f"No checkpoints found in {checkpoint_dir}")
        return

    print(f"Found {len(checkpoints)} checkpoints")
    print("-" * 60)

    results = []
    for cp in checkpoints:
        if cp['k'] is None:
            print(f"Skipping {cp['name']} - could not parse K value")
            continue

        print(f"\nAnalyzing {cp['name']}...")

        variance, snr = analyze_single_checkpoint(
            checkpoint_path=cp['path'],
            model_type=cp['type'],
            k=cp['k'],
            device=device,
            data=data,
            n_runs=n_runs,
            results_path=results_path
        )

        results.append((cp['name'], variance, snr))
        print(f"  Variance: {variance:.2e}, SNR: {snr:.4f}")

    print("-" * 60)
    print("\nSummary:")
    print(f"{'Model':<20} | {'Variance':<12} | {'SNR':<10}")
    print("-" * 50)
    for name, var, snr in results:
        print(f"{name:<20} | {var:<12.2e} | {snr:<10.4f}")

    if results_path:
        print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--model_path', type=str, help='Path to single checkpoint')
    input_group.add_argument('--checkpoint_dir', type=str, help='Directory to analyze all checkpoints')

    # For single checkpoint mode
    parser.add_argument('--model_type', type=str, default='vae', choices=['vae', 'iwae'])
    parser.add_argument('--k', type=int, default=1)

    # Analysis params
    parser.add_argument('--n_runs', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    # Results storage
    parser.add_argument('--results_path', type=str, default='results/evaluations.yaml')
    parser.add_argument('--no_save', action='store_true', help="Don't save to YAML")

    args = parser.parse_args()

    # Data: Get one batch
    train_loader, _, _ = get_dataloaders(batch_size=32)
    data, _ = next(iter(train_loader))
    data = data.to(args.device)

    if args.checkpoint_dir:
        analyze_all_checkpoints(
            checkpoint_dir=args.checkpoint_dir,
            device=args.device,
            data=data,
            n_runs=args.n_runs,
            results_path=args.results_path if not args.no_save else None
        )
    else:
        print(f"Analyzing Gradient Variance for {args.model_type.upper()}...")
        variance, snr = analyze_single_checkpoint(
            checkpoint_path=args.model_path,
            model_type=args.model_type,
            k=args.k,
            device=args.device,
            data=data,
            n_runs=args.n_runs,
            results_path=args.results_path if not args.no_save else None
        )

        print("-" * 40)
        print(f"Gradient Stats (Encoder Layer 1):")
        print(f"Average Variance: {variance:.2e}")
        print(f"Signal-to-Noise Ratio (SNR): {snr:.4f}")
        print("-" * 40)
