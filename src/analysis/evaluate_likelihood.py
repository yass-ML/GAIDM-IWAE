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

def evaluate_model(model, dataloader, device, k_eval):
    """
    Evaluates model using IWAE bound with k_eval samples.
    Returns average Log-Likelihood (nats).
    """
    model.eval()
    total_ll = 0
    total_samples = 0

    # We essentially treat the model as an IWAE(k_eval) for evaluation
    # To do this cleanly without modifying the model instance structure too much,
    # we can manually run the forward/loss logic or just update model.K if it's an IWAE instance.
    # The safest way is to wrap it or ensure the model passed acts like IWAE with K=k_eval.

    # Check if model has 'K' attribute and update it temporarily
    original_k = getattr(model, 'K', 1)
    if hasattr(model, 'K'):
        model.K = k_eval

    with torch.no_grad():
        for data, _ in tqdm(dataloader, desc=f"Evaluating with K={k_eval}", leave=False):
            data = data.to(device)
            # Forward pass: we need iwae forward behavior (mu, logvar, z samples)
            # But wait, VAE forward returns (recon, mu, logvar, z) with z shape (B, L)
            # IWAE forward returns (recon, mu, logvar, z) with z shape (K, B, L)

            # Implementation detail:
            # If we loaded a VAE checkpoint into an IWAE class with K=200, it works perfectly.
            # If we passed a VAE class, its forward does NOT sample K times.

            # Robust approach:
            # We assume the 'model' object passed here is an instance of IWAE (even if weights came from VAE)
            # OR we manually do the sampling logic here.

            # Let's rely on the caller to load weights into an IWAE(k_eval) instance.
            recon_x, mu, logvar, z = model(data)

            # compute_loss returns the NEGATIVE bound (loss)
            loss = model.compute_loss(data, recon_x, mu, logvar, z)

            # LL = -Loss
            batch_img_count = data.size(0)
            total_ll += -loss.item() * batch_img_count
            total_samples += batch_img_count

    # Restore K just in case
    if hasattr(model, 'K'):
        model.K = original_k

    return total_ll / total_samples

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Estimate Log-Likelihood")
    parser.add_argument('--vae_path', type=str, required=True, help='Path to VAE checkpoint')
    parser.add_argument('--iwae_path', type=str, required=True, help='Path to IWAE checkpoint')
    parser.add_argument('--k_eval', type=int, default=200, help='Number of samples for evaluation (default: 200)')
    parser.add_argument('--hidden_size', type=int, default=200)
    parser.add_argument('--latent_size', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (reduce if K is large to avoid OOM)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    # Data
    _, _, test_loader = get_dataloaders(batch_size=args.batch_size)

    input_size = 784
    output_size = 784

    print(f"Evaluation on Test Set (10k images) using IWAE bound with K={args.k_eval}")
    print("-" * 60)

    # 1. Evaluate VAE
    # We load VAE weights into an IWAE model structure to enable K-sampling
    print("Loading VAE weights into IWAE evaluator...")
    vae_evaluator = IWAE(K=args.k_eval, input_size=input_size, hidden_size=args.hidden_size,
                        latent_size=args.latent_size, output_size=output_size)
    vae_evaluator.load_state_dict(torch.load(args.vae_path, map_location=args.device))
    vae_evaluator.to(args.device)

    vae_ll = evaluate_model(vae_evaluator, test_loader, args.device, args.k_eval)
    print(f"VAE (trained K=1) Log-Likelihood estimate: {vae_ll:.4f} nats")

    # 2. Evaluate IWAE
    print("Loading IWAE weights into IWAE evaluator...")
    iwae_evaluator = IWAE(K=args.k_eval, input_size=input_size, hidden_size=args.hidden_size,
                         latent_size=args.latent_size, output_size=output_size)
    iwae_evaluator.load_state_dict(torch.load(args.iwae_path, map_location=args.device))
    iwae_evaluator.to(args.device)

    iwae_ll = evaluate_model(iwae_evaluator, test_loader, args.device, args.k_eval)
    print(f"IWAE (trained K=5) Log-Likelihood estimate: {iwae_ll:.4f} nats")

    print("-" * 60)
    print(f"Improvement: {iwae_ll - vae_ll:.4f} nats")
