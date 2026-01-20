import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import argparse
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.vae import VAE
from src.models.iwae import IWAE
from src.data.mnist_loader import get_dataloaders

def plot_reconstruction(model, dataloader, device, save_path):
    model.eval()
    data, _ = next(iter(dataloader))
    data = data.to(device)

    with torch.no_grad():
        recon, _, _, _ = model(data)
        # If IWAE, recon is (K, B, 784), take mean
        if len(recon.shape) == 3:
            recon = recon.mean(dim=0)

    # Randomly select 8 samples
    batch_size = data.size(0)
    num_samples = min(8, batch_size)
    indices = torch.randperm(batch_size)[:num_samples]
    
    # Reshape
    input_imgs = data[indices].view(-1, 28, 28).cpu()
    recon_imgs = recon[indices].view(-1, 28, 28).cpu()

    fig, axes = plt.subplots(2, 8, figsize=(12, 3))
    for i in range(8):
        # Original
        axes[0, i].imshow(input_imgs[i], cmap='gray')
        axes[0, i].axis('off')
        if i == 0: axes[0, i].set_title("Original")

        # Recon
        axes[1, i].imshow(recon_imgs[i], cmap='gray')
        axes[1, i].axis('off')
        if i == 0: axes[1, i].set_title("Recon")

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Reconstructions saved to {save_path}")
    plt.close()

def plot_samples(model, device, save_path, n=64):
    model.eval()
    # Sample from prior p(z) ~ N(0, I)
    z = torch.randn(n, model.latent_size).to(device)

    with torch.no_grad():
        samples = model.decoder(z)

    samples = samples.view(-1, 28, 28).cpu()

    # Grid size ( sqrt(n) )
    grid_size = int(n**0.5)
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))

    for i, ax in enumerate(axes.flatten()):
        ax.imshow(samples[i], cmap='gray')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Samples saved to {save_path}")
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to .pt checkpoint')
    parser.add_argument('--model_type', type=str, default='vae', choices=['vae', 'iwae'])
    parser.add_argument('--k', type=int, default=1, help='K for IWAE model loading (arg required for init)')
    parser.add_argument('--hidden_size', type=int, default=200)
    parser.add_argument('--latent_size', type=int, default=50)
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load Model
    input_size = 784
    output_size = 784

    if args.model_type == 'vae':
         model = VAE(input_size, args.hidden_size, args.latent_size, output_size)
    else:
         model = IWAE(args.k, input_size, args.hidden_size, args.latent_size, output_size)

    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    model.to(args.device)
    print(f"Loaded model from {args.model_path}")

    # Data for reconstruction
    _, _, test_loader = get_dataloaders(batch_size=32)

    # Plot
    base_name = os.path.basename(args.model_path).replace('.pt', '')
    plot_reconstruction(model, test_loader, args.device, os.path.join(args.output_dir, f'{base_name}_{args.model_type}_recon.png'))
    plot_samples(model, args.device, os.path.join(args.output_dir, f'{base_name}_{args.model_type}_samples.png'))
