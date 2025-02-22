import torch
from src.utils.utils import save_checkpoint, load_checkpoint, save_some_examples
import torch.nn as nn
import torch.optim as optim
from src.training.rgb2dsm.dataset import MapDataset
from src.training.rgb2dsm.generator import Generator
from src.training.rgb2dsm.discriminator import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.configs import rgb2dsm_config as config
import matplotlib.pyplot as plt
import argparse
from src.rgb2dsm.training.loss_function import ElevationLoss
import json
from pathlib import Path

torch.backends.cudnn.benchmark = True

# Argument parser
parser = argparse.ArgumentParser(description="Training Configuration")
parser.add_argument("--train_dir", type=str, required=True, help="Path to the training dataset directory")
parser.add_argument("--val_dir", type=str, required=True, help="Path to the validation dataset directory")
args = parser.parse_args()

TRAIN_DIR = args.train_dir
VAL_DIR = args.val_dir

def train_fn(
    disc, gen, loader, opt_disc, opt_gen, l1_loss, bce, g_scaler, d_scaler,
):
    loop = tqdm(loader, leave=True)
    metrics = {
        'total_disc_loss': 0,
        'total_gen_loss': 0,
        'total_l1_loss': 0,
        'total_g_fake_loss': 0,
        'elevation_metrics': {
            'low_range': {'count': 0, 'error': 0},    # 0-143
            'critical_range': {'count': 0, 'error': 0},  # 144-200
            'high_range': {'count': 0, 'error': 0},   # 201-255
        }
    }
    
    if len(loader) == 0:
        return metrics
    
    for idx, (x, y) in enumerate(loop):
        try:
            x = x.to(config.DEVICE)
            y = y.to(config.DEVICE)

            # Train Discriminator
            with torch.amp.autocast(config.DEVICE):
                y_fake = gen(x)
                D_real = disc(x, y)
                D_real_loss = bce(D_real, torch.ones_like(D_real))
                D_fake = disc(x, y_fake.detach())
                D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
                D_loss = (D_real_loss + D_fake_loss) / 2

            disc.zero_grad()
            d_scaler.scale(D_loss).backward()
            d_scaler.step(opt_disc)
            d_scaler.update()

            # Train generator
            with torch.amp.autocast(config.DEVICE):
                D_fake = disc(x, y_fake)
                G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
                L1 = l1_loss(y_fake, y) * config.L1_LAMBDA
                G_loss = G_fake_loss + L1

            opt_gen.zero_grad()
            g_scaler.scale(G_loss).backward()
            g_scaler.step(opt_gen)
            g_scaler.update()

            # Calculate and track elevation-specific errors
            with torch.no_grad():
                abs_diff = torch.abs(y_fake - y)
                
                # Track errors for different elevation ranges
                low_mask = y < 144
                critical_mask = (y >= 144) & (y <= 200)
                high_mask = y > 200
                
                metrics['elevation_metrics']['low_range']['error'] += abs_diff[low_mask].sum().item()
                metrics['elevation_metrics']['low_range']['count'] += low_mask.sum().item()
                
                metrics['elevation_metrics']['critical_range']['error'] += abs_diff[critical_mask].sum().item()
                metrics['elevation_metrics']['critical_range']['count'] += critical_mask.sum().item()
                
                metrics['elevation_metrics']['high_range']['error'] += abs_diff[high_mask].sum().item()
                metrics['elevation_metrics']['high_range']['count'] += high_mask.sum().item()

            # Accumulate losses
            metrics['total_disc_loss'] += D_loss.item()
            metrics['total_gen_loss'] += G_loss.item()
            metrics['total_l1_loss'] += L1.item()
            metrics['total_g_fake_loss'] += G_fake_loss.item()

            # Update progress bar
            if idx % 10 == 0:
                loop.set_postfix(
                    D_real=torch.sigmoid(D_real).mean().item(),
                    D_fake=torch.sigmoid(D_fake).mean().item(),
                    G_loss=G_loss.item(),
                    L1=L1.item(),
                )

        except Exception as e:
            print(f"Error during training iteration {idx}: {str(e)}")
            continue
    
    # Calculate averages
    num_batches = len(loader)
    avg_metrics = {
        'avg_disc_loss': metrics['total_disc_loss'] / num_batches,
        'avg_gen_loss': metrics['total_gen_loss'] / num_batches,
        'avg_l1_loss': metrics['total_l1_loss'] / num_batches,
        'avg_g_fake_loss': metrics['total_g_fake_loss'] / num_batches,
        'elevation_errors': {
            'low_range': metrics['elevation_metrics']['low_range']['error'] / max(metrics['elevation_metrics']['low_range']['count'], 1),
            'critical_range': metrics['elevation_metrics']['critical_range']['error'] / max(metrics['elevation_metrics']['critical_range']['count'], 1),
            'high_range': metrics['elevation_metrics']['high_range']['error'] / max(metrics['elevation_metrics']['high_range']['count'], 1),
        }
    }
    
    return avg_metrics

def save_metrics(metrics, epoch, save_dir):
    metrics_dir = Path(save_dir) / 'metrics'
    metrics_dir.mkdir(exist_ok=True, parents=True)
    
    # Save metrics for this epoch
    with open(metrics_dir / f'epoch_{epoch}.json', 'w') as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    disc = Discriminator(in_channels=1).to(config.DEVICE)
    gen = Generator(in_channels=1, features=64).to(config.DEVICE)
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999),)
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = ElevationLoss(base_weight=1.0,critical_range_weight=2.0,critical_range=(144, 200))

    # Loads pre-trained model weights if LOAD_MODEL is True.
    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE,
        )

    train_dataset = MapDataset(root_dir=TRAIN_DIR)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )
    g_scaler = torch.amp.GradScaler(config.DEVICE)
    d_scaler = torch.amp.GradScaler(config.DEVICE)
    val_dataset = MapDataset(root_dir=VAL_DIR)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)


    training_metrics = []
    
    for epoch in range(config.NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
        try:
            # Get detailed metrics from training
            metrics = train_fn(
                disc, gen, train_loader, opt_disc, opt_gen, L1_LOSS, BCE, g_scaler, d_scaler,
            )
            
            # Print detailed metrics
            print(f"\nEpoch {epoch+1} Metrics:")
            print(f"Generator Loss: {metrics['avg_gen_loss']:.4f}")
            print(f"Discriminator Loss: {metrics['avg_disc_loss']:.4f}")
            print(f"L1 Loss: {metrics['avg_l1_loss']:.4f}")
            print("\nElevation Range Errors:")
            print(f"Low Range (0-143): {metrics['elevation_errors']['low_range']:.4f}")
            print(f"Critical Range (144-200): {metrics['elevation_errors']['critical_range']:.4f}")
            print(f"High Range (201-255): {metrics['elevation_errors']['high_range']:.4f}")
            
            # Save metrics
            save_metrics(metrics, epoch, "training_logs")
            
            # Store for plotting
            disc_losses.append(metrics['avg_disc_loss'])
            gen_losses.append(metrics['avg_gen_loss'])
            
            # Save checkpoints
            if config.SAVE_MODEL and epoch % 5 == 0:
                save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
                save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)

            save_some_examples(gen, val_loader, epoch, folder="rgb_dsm_generated/")
            
        except Exception as e:
            print(f"Error in epoch {epoch}: {str(e)}")
            continue

    # Plot training metrics
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Losses
    plt.subplot(1, 2, 1)
    plt.plot(disc_losses, label="Discriminator Loss")
    plt.plot(gen_losses, label="Generator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training Losses")

    plt.tight_layout()
    plt.savefig("training_metrics.png")
    plt.show()