import torch
import torch.nn as nn
import argparse
import os
import sys
from tqdm import tqdm
import math

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.iwae import IWAE
from src.data.mnist_loader import get_dataloaders
from src.utils.checkpoint_utils import (
    discover_checkpoints,
    save_results,
    get_model_key
)


def evaluate_model(model, dataloader, device, k_eval):
    """
    Evaluates model using IWAE bound with k_eval samples.
    Returns average Log-Likelihood (nats).
    """
    model.eval()
    total_ll = 0
    total_recon_error = 0
    total_weighted_recon_error = 0
    total_samples = 0

    # Check if model has 'K' attribute and update it temporarily
    original_k = getattr(model, 'K', 1)
    if hasattr(model, 'K'):
        model.K = k_eval

    with torch.no_grad():
        for data, _ in tqdm(dataloader, desc=f"Evaluating with K={k_eval}", leave=False):
            data = data.to(device)
            recon_x, mu, logvar, z = model(data)
            loss = model.compute_loss(data, recon_x, mu, logvar, z)

            # Calculate Reconstruction Error (BCE)
            # Shape of recon_x: (K, B, L) for IWAE/VAE(wrapped), or (B, L) for VAE(native)
            # We standardize to (K, B, L) for computation
            x_flat = data.view(data.size(0), -1)

            if len(recon_x.shape) == 2: # (B, L)
                 # VAE standard
                 bce = torch.nn.functional.binary_cross_entropy(recon_x, x_flat, reduction='sum')
                 recon_error = bce.item()
                 weighted_recon_error = recon_error
            else: # (K, B, L)
                 # IWAE: Compute expected reconstruction error over K samples
                 # Sum over Batch & Latent, Mean over K
                 x_expanded = x_flat.unsqueeze(0).expand_as(recon_x) # (K, B, L)
                 # BCE per sample per batch: (K, B)
                 bce_per_sample = torch.nn.functional.binary_cross_entropy(recon_x, x_expanded, reduction='none').sum(dim=2)

                 # 1. Unweighted Reconstruction Error (Expected Value)
                 recon_error = bce_per_sample.mean(dim=0).sum().item()

                 # 2. Weighted Reconstruction Error (Importance Sampled)
                 log_p_x_given_z = -bce_per_sample # (K, B)

                 # q(z|x) and p(z)
                 if mu.dim() == 2:
                     mu_k = mu.unsqueeze(0).expand_as(z)
                     logvar_k = logvar.unsqueeze(0).expand_as(z)
                 else:
                     mu_k, logvar_k = mu, logvar

                 log_q_z_given_x = -0.5 * (torch.log(2 * torch.tensor(math.pi)) + logvar_k + (z - mu_k).pow(2) / torch.exp(logvar_k))
                 log_q_z_given_x = log_q_z_given_x.sum(dim=2) # (K, B)

                 log_p_z = -0.5 * (torch.log(2 * torch.tensor(math.pi)) + z.pow(2))
                 log_p_z = log_p_z.sum(dim=2) # (K, B)

                 log_w = log_p_x_given_z + log_p_z - log_q_z_given_x # (K, B)
                 w_tilde = torch.softmax(log_w, dim=0) # (K, B)

                 weighted_bce = (w_tilde * bce_per_sample).sum(dim=0) # (B,)
                 weighted_recon_error = weighted_bce.sum().item()

            batch_img_count = data.size(0)
            total_ll += -loss.item() * batch_img_count
            total_samples += batch_img_count
            total_recon_error += recon_error
            total_weighted_recon_error += weighted_recon_error

    # Restore K just in case
    if hasattr(model, 'K'):
        model.K = original_k

    return total_ll / total_samples, total_recon_error / total_samples, total_weighted_recon_error / total_samples


def evaluate_single_checkpoint(
    checkpoint_path: str,
    model_type: str,
    k_train: int,
    k_eval: int,
    device: str,
    test_loader,
    hidden_size: int = 200,
    latent_size: int = 50,
    results_path: str = None
) -> tuple:
    """
    Evaluate a single checkpoint and optionally save results.

    Returns:
        Log-likelihood estimate
    """
    input_size = 784
    output_size = 784

    # Load weights into IWAE evaluator (works for both VAE and IWAE checkpoints)
    evaluator = IWAE(
        K=k_eval,
        input_size=input_size,
        hidden_size=hidden_size,
        latent_size=latent_size,
        output_size=output_size
    )
    evaluator.load_state_dict(torch.load(checkpoint_path, map_location=device))
    evaluator.to(device)

    ll, recon_error, weighted_recon_error = evaluate_model(evaluator, test_loader, device, k_eval)

    # Save to YAML if path provided
    if results_path:
        model_key = get_model_key(checkpoint_path)

        metrics = {
            'log_likelihood': round(ll, 2),
            'reconstruction_error': round(recon_error, 2)
        }

        # Add weighted recon error only for IWAE (K>1)
        if model_type == 'iwae' and k_train > 1:
            metrics['weighted_reconstruction_error'] = round(weighted_recon_error, 2)

        save_results(
            results_path=results_path,
            model_key=model_key,
            metrics=metrics,
            model_info={
                'path': checkpoint_path,
                'type': model_type,
                'k': k_train
            }
        )

    return ll, recon_error, weighted_recon_error


def evaluate_all_checkpoints(
    checkpoint_dir: str,
    k_eval: int,
    device: str,
    test_loader,
    hidden_size: int = 200,
    latent_size: int = 50,
    results_path: str = 'results/evaluations.yaml'
):
    """
    Discover and evaluate all checkpoints in directory.
    """
    checkpoints = discover_checkpoints(checkpoint_dir)

    if not checkpoints:
        print(f"No checkpoints found in {checkpoint_dir}")
        return

    print(f"Found {len(checkpoints)} checkpoints")
    print(f"Evaluation on Test Set (10k images) using IWAE bound with K={k_eval}")
    print("-" * 60)

    results = []
    for cp in checkpoints:
        print(f"\nEvaluating {cp['name']}...")

        ll, recon_error, weighted_recon = evaluate_single_checkpoint(
            checkpoint_path=cp['path'],
            model_type=cp['type'],
            k_train=cp['k'] if cp['k'] else 1,
            k_eval=k_eval,
            device=device,
            test_loader=test_loader,
            hidden_size=hidden_size,
            latent_size=latent_size,
            results_path=results_path
        )

        results.append((cp['name'], ll, recon_error, weighted_recon, cp.get('type'), cp.get('k')))

        if cp.get('type') == 'iwae' and (cp.get('k') or 0) > 1:
            print(f"  {cp['name']} Log-Likelihood: {ll:.2f} nats | Recon: {recon_error:.2f} | Weighted Recon: {weighted_recon:.2f}")
        else:
            print(f"  {cp['name']} Log-Likelihood: {ll:.2f} nats | Recon: {recon_error:.2f}")

    print("-" * 60)
    print("\nSummary:")
    # Sort by Log Likelihood
    for name, ll, recon, weighted, mtype, k in sorted(results, key=lambda x: x[1], reverse=True):
        if mtype == 'iwae' and k and k > 1:
            print(f"  {name:<15}: LL={ll:.2f} | Recon={recon:.2f} | Weighted={weighted:.2f}")
        else:
            print(f"  {name:<15}: LL={ll:.2f} | Recon={recon:.2f}")

    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Estimate Log-Likelihood")

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--checkpoint', type=str, help='Path to single checkpoint')
    input_group.add_argument('--checkpoint_dir', type=str, help='Directory to discover all checkpoints')

    # Legacy options for single checkpoint mode
    parser.add_argument('--model_type', type=str, default='vae', choices=['vae', 'iwae'],
                        help='Model type (for single checkpoint mode)')
    parser.add_argument('--k_train', type=int, default=1,
                        help='K used during training (for single checkpoint mode)')

    # Evaluation params
    parser.add_argument('--k_eval', type=int, default=200,
                        help='Number of samples for evaluation (default: 200)')
    parser.add_argument('--hidden_size', type=int, default=200)
    parser.add_argument('--latent_size', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (reduce if K is large to avoid OOM)')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')

    # Output
    parser.add_argument('--results_path', type=str, default='results/evaluations.yaml',
                        help='Path to save results YAML')
    parser.add_argument('--no_save', action='store_true',
                        help="Don't save results to YAML")

    args = parser.parse_args()

    # Data
    _, _, test_loader = get_dataloaders(batch_size=args.batch_size)

    if args.checkpoint_dir:
        # Evaluate all checkpoints in directory
        evaluate_all_checkpoints(
            checkpoint_dir=args.checkpoint_dir,
            k_eval=args.k_eval,
            device=args.device,
            test_loader=test_loader,
            hidden_size=args.hidden_size,
            latent_size=args.latent_size,
            results_path=args.results_path if not args.no_save else None
        )
    else:
        # Single checkpoint mode
        print(f"Evaluation on Test Set using IWAE bound with K={args.k_eval}")
        print("-" * 60)

        ll, recon_error, weighted_recon = evaluate_single_checkpoint(
            checkpoint_path=args.checkpoint,
            model_type=args.model_type,
            k_train=args.k_train,
            k_eval=args.k_eval,
            device=args.device,
            test_loader=test_loader,
            hidden_size=args.hidden_size,
            latent_size=args.latent_size,
            results_path=args.results_path if not args.no_save else None
        )

        print(f"{args.model_type.upper()} (K={args.k_train}) Log-Likelihood: {ll:.2f} nats")
        print(f"Reconstruction Error: {recon_error:.2f}")
        if args.model_type == 'iwae' and args.k_train > 1:
             print(f"Weighted Reconstruction Error: {weighted_recon:.2f}")

        if not args.no_save:
            print(f"Results saved to {args.results_path}")
