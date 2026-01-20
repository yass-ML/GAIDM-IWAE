
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
from src.utils.checkpoint_utils import (
    discover_checkpoints,
    save_results,
    get_model_key
)


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
    kl_elementwise = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
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

            if isinstance(model, IWAE):
                x_flat = data.view(data.size(0), -1)
                h = model.encoder(x_flat)
                mu, logvar = h.chunk(2, dim=1)
            else:
                _, mu, logvar, _ = model(data)

            batch_kl = compute_batch_kl(mu, logvar)
            total_kl += batch_kl
            num_batches += 1

    final_avg_kl = total_kl / num_batches
    num_active = (final_avg_kl > threshold).sum().item()

    return num_active, final_avg_kl


def plot_kl_stats(avg_kl_values, model_name, save_path):
    """Plots the KL value for each dimension sorted."""
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


def analyze_single_checkpoint(
    checkpoint_path: str,
    model_type: str,
    k: int,
    device: str,
    test_loader,
    threshold: float = 0.01,
    output_dir: str = './results',
    results_path: str = None,
    hidden_size: int = 200,
    latent_size: int = 50
) -> int:
    """Analyze a single checkpoint and optionally save results."""
    input_size = 784
    output_size = 784

    if model_type == 'vae':
        model = VAE(input_size, hidden_size, latent_size, output_size)
    else:
        model = IWAE(k, input_size, hidden_size, latent_size, output_size)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)

    n_active, avg_kls = calc_active_units(model, test_loader, device, threshold)

    # Save to YAML if path provided
    if results_path:
        model_key = get_model_key(checkpoint_path)
        save_results(
            results_path=results_path,
            model_key=model_key,
            metrics={'active_units': n_active}
        )

    # Plot
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.basename(checkpoint_path).replace('.pt', '')
    plot_kl_stats(
        avg_kls,
        f"{model_type.upper()} (Active: {n_active})",
        os.path.join(output_dir, f"{base_name}_kl.png")
    )

    return n_active


def analyze_all_checkpoints(
    checkpoint_dir: str,
    device: str,
    test_loader,
    threshold: float = 0.01,
    output_dir: str = './results',
    results_path: str = 'results/evaluations.yaml'
):
    """Discover and analyze all checkpoints."""
    checkpoints = discover_checkpoints(checkpoint_dir)

    if not checkpoints:
        print(f"No checkpoints found in {checkpoint_dir}")
        return

    print(f"Found {len(checkpoints)} checkpoints")
    print("-" * 50)

    results = []
    for cp in checkpoints:
        if cp['k'] is None:
            print(f"Skipping {cp['name']} - could not parse K value")
            continue

        print(f"\nAnalyzing {cp['name']}...")

        n_active = analyze_single_checkpoint(
            checkpoint_path=cp['path'],
            model_type=cp['type'],
            k=cp['k'],
            device=device,
            test_loader=test_loader,
            threshold=threshold,
            output_dir=output_dir,
            results_path=results_path
        )

        results.append((cp['name'], n_active))
        print(f"  Active Units: {n_active}/50")

    print("-" * 50)
    print("\nSummary:")
    for name, n_active in results:
        print(f"  {name}: {n_active} active units")

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
    parser.add_argument('--threshold', type=float, default=0.01, help='Threshold for active unit')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_dir', type=str, default='./results')

    # Results storage
    parser.add_argument('--results_path', type=str, default='results/evaluations.yaml')
    parser.add_argument('--no_save', action='store_true', help="Don't save to YAML")

    args = parser.parse_args()

    # Data
    _, _, test_loader = get_dataloaders(batch_size=128)

    if args.checkpoint_dir:
        analyze_all_checkpoints(
            checkpoint_dir=args.checkpoint_dir,
            device=args.device,
            test_loader=test_loader,
            threshold=args.threshold,
            output_dir=args.output_dir,
            results_path=args.results_path if not args.no_save else None
        )
    else:
        print(f"Analyzing {args.model_type.upper()}...")
        n_active = analyze_single_checkpoint(
            checkpoint_path=args.model_path,
            model_type=args.model_type,
            k=args.k,
            device=args.device,
            test_loader=test_loader,
            threshold=args.threshold,
            output_dir=args.output_dir,
            results_path=args.results_path if not args.no_save else None
        )

        print("-" * 40)
        print(f"Total Latent Dimensions: 50")
        print(f"Active Units (KL > {args.threshold}): {n_active}")
        print("-" * 40)
