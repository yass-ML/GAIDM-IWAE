
import torch
import torch.optim as optim
import argparse
import os
import time
from tqdm import tqdm
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.vae import VAE
from src.models.iwae import IWAE
from src.data.mnist_loader import get_dataloaders

class Trainer:
    def __init__(self, model, optimizer, train_loader, val_loader, device='cpu'):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.val_loader = val_loader
        self.device = device
        self.model.to(self.device)

    def save_checkpoint(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]", leave=False)

        for batch_idx, (data, _) in enumerate(pbar):
            data = data.to(self.device)
            # Flatten is handled in model forward/loss, but loss expects (B, 784) if using VAE default
            # Actually our models handle flattening internally or expect specific shapes.
            # VAE/IWAE forward handles flattening.

            self.optimizer.zero_grad()

            # Forward pass
            # Returns: recon_x, mu, logvar, z
            recon_x, mu, logvar, z = self.model(data)

            # Compute Loss
            # IWAE needs z, VAE accepts z (optional)
            loss = self.model.compute_loss(data, recon_x, mu, logvar, z)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for data, _ in self.train_loader: # Validate on training data? Or val? Usually val.
                # User asked for "Compare VAE vs IWAE", usually we check ELBO on held-out data
                pass

            # Use val_loader
            for data, _ in self.val_loader:
                data = data.to(self.device)
                recon_x, mu, logvar, z = self.model(data)
                loss = self.model.compute_loss(data, recon_x, mu, logvar, z)
                total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)
        return avg_loss

    def fit(self, epochs):
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate(epoch)
            print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

def parse_args():
    parser = argparse.ArgumentParser(description="Train VAE/IWAE on MNIST")

    # Model params
    parser.add_argument('--model', type=str, default='vae', choices=['vae', 'iwae'], help='Model type')
    parser.add_argument('--k', type=int, default=1, help='Number of importance samples (K). For VAE K=1.')
    parser.add_argument('--hidden_size', type=int, default=200, help='Hidden layer size')
    parser.add_argument('--latent_size', type=int, default=50, help='Latent dimension size')

    # Training params
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device (cuda/cpu)')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save models')


    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Set seed
    torch.manual_seed(args.seed)

    # Data
    train_loader, val_loader, _ = get_dataloaders(batch_size=args.batch_size)

    # Model
    input_size = 784
    output_size = 784

    if args.model == 'vae' or (args.model == 'iwae' and args.k == 1):
        # Even if user says iwae K=1, it's effectively VAE, but let's stick to class names
        if args.model == 'iwae':
             model = IWAE(K=1, input_size=input_size, hidden_size=args.hidden_size,
                         latent_size=args.latent_size, output_size=output_size)
        else:
            model = VAE(input_size=input_size, hidden_size=args.hidden_size,
                        latent_size=args.latent_size, output_size=output_size)
    else:
        model = IWAE(K=args.k, input_size=input_size, hidden_size=args.hidden_size,
                     latent_size=args.latent_size, output_size=output_size)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Trainer
    print(f"Training {args.model.upper()} with K={args.k} on {args.device}...")
    trainer = Trainer(model, optimizer, train_loader, val_loader, device=args.device)

    start_time = time.time()
    trainer.fit(args.epochs)
    end_time = time.time()

    training_time = end_time - start_time
    print(f"Training finished in {training_time:.2f} seconds")

    # Save Model
    os.makedirs(args.save_dir, exist_ok=True)
    filename = f"{args.model}_k{args.k}_epochs{args.epochs}_seed{args.seed}.pt"
    save_path = os.path.join(args.save_dir, filename)
    trainer.save_checkpoint(save_path)

    # Save training time to centralized results
    try:
        from src.utils.checkpoint_utils import save_results, get_model_key
        model_key = get_model_key(save_path)
        save_results(
            results_path='results/evaluations.yaml',
            model_key=model_key,
            metrics={'training_time': training_time},
            model_info={
                'path': save_path,
                'type': args.model,
                'k': args.k,
                'epochs': args.epochs
            }
        )
        print(f"Training time saved to results/evaluations.yaml")
    except ImportError:
        print("Warning: Could not import checkpoint_utils to save training time.")
