import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor, MaskRCNN_ResNet50_FPN_Weights
import matplotlib.pyplot as plt
import numpy as np
import cv2
from src.BATC.training.dataset import BuildingDataset
import argparse
import src.configs.maskrcnn_config as cfg
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Training Configuration")
parser.add_argument("--train_images_dir", type=str, required=True)
parser.add_argument("--train_annotations_dir", type=str, required=True)
parser.add_argument("--val_images_dir", type=str, required=True)
parser.add_argument("--val_annotations_dir", type=str, required=True)
args = parser.parse_args()

TRAIN_IMAGES_DIR = args.train_images_dir
TRAIN_ANNOTATIONS_DIR = args.train_annotations_dir
VAL_IMAGES_DIR = args.val_images_dir
VAL_ANNOTATIONS_DIR = args.val_annotations_dir

train_dataset = BuildingDataset(TRAIN_IMAGES_DIR, TRAIN_ANNOTATIONS_DIR)
val_dataset = BuildingDataset(VAL_IMAGES_DIR, VAL_ANNOTATIONS_DIR)

train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS)

def get_model(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = MaskRCNNPredictor(in_features, num_classes=num_classes)

    return model

num_classes = 2  # Background + Buildings
model = get_model(num_classes)

device = cfg.DEVICE
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

best_loss = float('inf')
train_losses = []

for epoch in range(cfg.NUM_EPOCHS):
    model.train()
    total_loss = 0

    # Add tqdm progress bar for training loop
    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.NUM_EPOCHS}", unit="batch")

    for images, targets in train_loader_tqdm:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Update tqdm description with current loss
        train_loader_tqdm.set_postfix(loss=total_loss / len(train_loader))

    lr_scheduler.step()

    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Epoch {epoch+1}/{cfg.NUM_EPOCHS}, Loss: {avg_loss:.4f}")

    # Save the best model
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), "best_mask_rcnn_buildings.pth")

model.eval()
with torch.no_grad():
    # Add tqdm progress bar for validation loop
    val_loader_tqdm = tqdm(val_loader, desc="Validation", unit="batch")

    for images, targets in val_loader_tqdm:
        images = [img.to(device) for img in images]
        predictions = model(images)

        # Visualize Predictions
        for img, pred in zip(images, predictions):
            img = img.permute(1, 2, 0).cpu().numpy()
            plt.figure(figsize=(10, 10))
            plt.imshow(img)

            for mask in pred["masks"]:
                mask = mask[0].cpu().numpy()
                plt.imshow(mask, alpha=0.5, cmap="jet")

            plt.show()

# Plot the training loss graph
plt.figure()
plt.plot(range(1, cfg.NUM_EPOCHS + 1), train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.show()