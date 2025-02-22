import torch
from src.utils.utils import save_checkpoint, load_checkpoint, save_some_examples
import torch.nn as nn
import torch.optim as optim
from src.rgb2dsm.training.dataset import MapDataset
from src.rgb2dsm.training.generator import Generator
from src.rgb2dsm.training.discriminator import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.configs import rgb2dsm_config as config
import matplotlib.pyplot as plt
import argparse
from src.rgb2dsm.training.loss_function import ElevationLoss

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
    total_disc_loss = 0
    total_gen_loss = 0
    
    # Add a check for empty loader
    if len(loader) == 0:
        return 0.0, 0.0  # Return default values if loader is empty
    
    for idx, (x, y) in enumerate(loop):
        try:
            x = x.to(config.DEVICE)
            y = y.to(config.DEVICE)

            # Train Discriminator. Computes real (D_real) and fake (D_fake) losses using Binary Cross-Entropy (BCE).
            with torch.amp.autocast(config.DEVICE):
                y_fake = gen(x)
                D_real = disc(x, y)
                D_real_loss = bce(D_real, torch.ones_like(D_real))
                D_fake = disc(x, y_fake.detach())
                D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
                D_loss = (D_real_loss + D_fake_loss) / 2

            # Updates the discriminator weights to better distinguish between real and fake samples.
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

            # Updates the generator weights to minimize the loss between the generated image and the target image.
            opt_gen.zero_grad()
            g_scaler.scale(G_loss).backward()
            g_scaler.step(opt_gen)
            g_scaler.update()

            # Accumulate losses
            total_disc_loss += D_loss.item()
            total_gen_loss += G_loss.item()

            if idx % 10 == 0:
                loop.set_postfix(
                    D_real=torch.sigmoid(D_real).mean().item(),
                    D_fake=torch.sigmoid(D_fake).mean().item(),
                )
        except Exception as e:
            print(f"Error during training iteration {idx}: {str(e)}")
            continue
    
    # Calculate average losses
    try:
        avg_disc_loss = total_disc_loss / len(loader)
        avg_gen_loss = total_gen_loss / len(loader)
    except ZeroDivisionError:
        print("Warning: Empty loader or division by zero")
        avg_disc_loss = total_disc_loss
        avg_gen_loss = total_gen_loss
        
    return avg_disc_loss, avg_gen_loss


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


    disc_losses = []
    gen_losses = []
    for epoch in range(config.NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
        try:
            avg_disc_loss, avg_gen_loss = train_fn(
                disc, gen, train_loader, opt_disc, opt_gen, L1_LOSS, BCE, g_scaler, d_scaler,
            )
            print(f"Returned losses - Disc: {avg_disc_loss:.4f}, Gen: {avg_gen_loss:.4f}")
            disc_losses.append(avg_disc_loss)
            gen_losses.append(avg_gen_loss)
        except Exception as e:
            print(f"Error in epoch {epoch}: {str(e)}")
            continue
        if config.SAVE_MODEL and epoch % 5 == 0:
            save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)

        save_some_examples(gen, val_loader, epoch, folder="/content/drive/MyDrive/ProjeDosyalari/rgb2dsm/v0.2/rgb_dsm_generated/")
    
    

    plt.plot(disc_losses, label="Discriminator Loss")
    plt.plot(gen_losses, label="Generator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss During Training")
    plt.savefig("training_loss_plot.png")  # Save plot as an image
    plt.show()