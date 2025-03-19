import argparse
import torch
import os
from src.BATC.model import get_model
import src.configs.maskrcnn_config as cfg
import numpy as np
from src.BATC.training.dataset import MaskRCNNDataset, custom_collate
import torch.optim as optim
from tqdm import tqdm

# Argument parser
parser = argparse.ArgumentParser(description="Training Configuration")
parser.add_argument(
    "--images_path",
    type=str,
    required=True,
    help="Path to the images directory",
)
parser.add_argument(
    "--masks_path",
    type=str,
    required=True,
    help="Path to the masks directory",
)
parser.add_argument(
    "--save_path",
    type=str,
    required=True,
    help="Path to save the trained model",
)
args = parser.parse_args()

IMAGES_PATH = args.images_path
MASKS_PATH = args.masks_path
SAVE_PATH = args.save_path

def train():
    print("Loading dataset...")
    images = sorted(os.listdir(IMAGES_PATH))
    masks = sorted(os.listdir(MASKS_PATH))

    num = int(0.9 * len(images))
    num = num if num % 2 == 0 else num + 1
    train_imgs_inds = np.random.choice(range(len(images)), num, replace=False)
    val_imgs_inds = np.setdiff1d(range(len(images)), train_imgs_inds)
    train_imgs = np.array(images)[train_imgs_inds]
    val_imgs = np.array(images)[val_imgs_inds]
    train_masks = np.array(masks)[train_imgs_inds]
    val_masks = np.array(masks)[val_imgs_inds]

    print("Creating dataloaders...")
    train_dl = torch.utils.data.DataLoader(MaskRCNNDataset(train_imgs, train_masks, IMAGES_PATH, MASKS_PATH),
                                           batch_size=cfg.BATCH_SIZE,
                                           shuffle=True,
                                           collate_fn=custom_collate,
                                           num_workers=cfg.NUM_WORKERS,
                                           pin_memory=True if torch.cuda.is_available() else False)
    val_dl = torch.utils.data.DataLoader(MaskRCNNDataset(val_imgs, val_masks, IMAGES_PATH, MASKS_PATH),
                                         batch_size=cfg.BATCH_SIZE,
                                         shuffle=False,
                                         collate_fn=custom_collate,
                                         num_workers=cfg.NUM_WORKERS,
                                         pin_memory=True if torch.cuda.is_available() else False)

    print("Initializing model...")
    model = get_model().to(cfg.DEVICE)
    optimizer = optim.SGD([p for p in model.parameters() if p.requires_grad], lr=cfg.LEARNING_RATE, momentum=cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY)

    all_train_losses = []
    all_val_losses = []
    flag = False

    for epoch in range(cfg.NUM_EPOCHS):
        print(f"Starting epoch {epoch+1}/{cfg.NUM_EPOCHS}...")
        train_epoch_loss = 0
        val_epoch_loss = 0
        model.train()

        # Training loop with tqdm progress bar
        train_pbar = tqdm(train_dl, total=len(train_dl), desc=f"Epoch {epoch+1}/{cfg.NUM_EPOCHS} - Training")
        for imgs, targets in train_pbar:
            imgs = [img.to(cfg.DEVICE) for img in imgs]
            targets = [{k: v.to(cfg.DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(imgs, targets)

            if isinstance(loss_dict, dict):
                losses = sum(loss for loss in loss_dict.values())
            elif isinstance(loss_dict, list):
                print("Unexpected list output in training! Check model mode.")
                losses = torch.tensor(0.0, device=cfg.DEVICE)
            else:
                raise TypeError(f"Unexpected loss_dict type: {type(loss_dict)}")

            train_epoch_loss += losses.item()

            losses = sum(loss for loss in loss_dict.values())
            train_epoch_loss += losses.item()

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            train_pbar.set_postfix(loss=train_epoch_loss / len(train_dl))

        all_train_losses.append(train_epoch_loss)
        print(f"Epoch {epoch+1}/{cfg.NUM_EPOCHS} - Train Loss: {train_epoch_loss:.4f}")

        # Validation loop
        print("Starting validation...")
        with torch.no_grad():
            val_pbar = tqdm(val_dl, total=len(val_dl), desc=f"Epoch {epoch+1}/{cfg.NUM_EPOCHS} - Validation")
            for imgs, targets in val_pbar:
                imgs = [img.to(cfg.DEVICE) for img in imgs]
                targets = [{k: v.to(cfg.DEVICE) for k, v in t.items()} for t in targets]

                loss_dict = model(imgs, targets)

                if isinstance(loss_dict, dict):
                    losses = sum(loss for loss in loss_dict.values())
                elif isinstance(loss_dict, list):
                    print("Unexpected list output in validation! Model might be in eval mode.")
                    losses = torch.tensor(0.0, device=cfg.DEVICE)
                else:
                    raise TypeError(f"Unexpected loss_dict type: {type(loss_dict)}")

                val_epoch_loss += losses.item()

                val_pbar.set_postfix(loss=val_epoch_loss / len(val_dl))

        all_val_losses.append(val_epoch_loss)
        print(f"Epoch {epoch+1}/{cfg.NUM_EPOCHS} - Validation Loss: {val_epoch_loss:.4f}")

    if cfg.SAVE_MODEL:
        print("Saving model...")
        torch.save(model.state_dict(), os.path.join(SAVE_PATH, "maskrcnn_weights.pth"))
        print("Model saved successfully.")

if __name__ == "__main__":
    print("Starting training process...")
    train()
    print("Training complete.")
